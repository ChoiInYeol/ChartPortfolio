import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import Optional, List


class PortfolioTCN(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_output: int,
        num_channels: List[int],
        kernel_size: int,
        n_dropout: float,
        n_timestep: int,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        TCN 기반 포트폴리오 최적화 모델
        
        Args:
            n_feature: 입력 특성 수 (n_stocks와 동일)
            n_output: 출력 차원 (n_stocks와 동일)
            num_channels: TCN 채널 수 리스트 (예: [64, 64, 64])
            kernel_size: 컨볼루션 커널 크기
            n_dropout: 드롭아웃 비율
            n_timestep: 시계열 길이
            lb: 최소 포트폴리오 비중
            ub: 최대 포트폴리오 비중
            n_select: 선택할 종목 수 (None인 경우 n_stocks 사용)
        """
        super().__init__()
        self.input_size = n_feature
        self.n_stocks = n_output
        self.lb = lb
        self.ub = ub
        self.n_select = n_select if n_select is not None else n_output
        
        # Score Block (h1) - TCN 인코더
        self.tcn = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=n_dropout
        )
        
        # 종목 선택을 위한 score 생성 레이어
        self.score_layer = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(num_channels[-1], n_output)
        )
        
        self.tempmaxpool = nn.MaxPool1d(n_timestep)

    def forward(self, x: torch.Tensor, x_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, n_stocks]
            x_probs: 상승확률 (사용하지 않음)
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # Score Block (h1)
        output = self.tcn(x.transpose(1, 2))
        features = self.tempmaxpool(output).squeeze(-1)
        
        # 종목 선택 scores 생성
        scores = self.score_layer(features)
        
        # Portfolio Block (h2) - 제약조건을 만족하는 가중치 생성
        if self.n_select < self.n_stocks:  # Cardinality 제약
            topk_values, topk_indices = torch.topk(scores, self.n_select, dim=1)
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            scores = scores * mask
        
        # Long-only + Maximum Position 제약
        weights = F.softmax(scores, dim=-1)
        weights = torch.clamp(weights, self.lb, self.ub)
        
        # 정규화
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights

    def rebalance(self, weight: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
        """
        포트폴리오 비중 재조정
        
        Args:
            weight: 초기 비중
            lb: 최소 비중
            ub: 최대 비중
            
        Returns:
            재조정된 비중
        """
        # 선택된 종목만 재조정
        selected_mask = (weight > 0).float()
        weight = weight * selected_mask
        
        weight_clamped = torch.clamp(weight, lb, ub)
        total_excess = weight_clamped.sum() - 1.0

        while abs(total_excess) > 1e-6:
            if total_excess > 0:
                # 초과분 처리
                adjustable = (weight_clamped > lb) & (selected_mask == 1)
                if not adjustable.any():
                    break
                adjustment = total_excess / adjustable.sum()
                weight_clamped[adjustable] -= adjustment
            else:
                # 부족분 처리
                adjustable = (weight_clamped < ub) & (selected_mask == 1)
                if not adjustable.any():
                    break
                adjustment = -total_excess / adjustable.sum()
                weight_clamped[adjustable] += adjustment
            
            weight_clamped = torch.clamp(weight_clamped, lb, ub)
            total_excess = weight_clamped.sum() - 1.0

        return weight_clamped


class PortfolioTCNWithProb(PortfolioTCN):
    def __init__(
        self,
        n_feature: int,
        n_output: int,
        num_channels: List[int],
        kernel_size: int,
        n_dropout: float,
        n_timestep: int,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        상승확률을 활용하는 TCN 기반 포트폴리오 최적화 모델
        
        Args:
            n_feature: 입력 특성 수 (n_stocks와 동일)
            n_output: 출력 차원 (n_stocks와 동일)
            num_channels: TCN 채널 수 리스트
            kernel_size: 컨볼루션 커널 크기
            n_dropout: 드롭아웃 비율
            n_timestep: 시계열 길이
            lb: 최소 포트폴리오 비중
            ub: 최대 포트폴리오 비중
            n_select: 선택할 종목 수 (None인 경우 n_stocks 사용)
        """
        super().__init__(
            n_feature, n_output, num_channels,
            kernel_size, n_dropout, n_timestep,
            lb, ub, n_select
        )
        
        # 상승확률 인코딩을 위한 레이어
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_output, num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.SiLU(),
            nn.Dropout(n_dropout)
        )
        
        # 결합된 특성을 처리하기 위한 레이어 수정
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(num_channels[-1], n_output)
        )
        
        self.score_layers = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(num_channels[-1], n_output)
        )

    def forward(self, x_returns: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x_returns: 수익률 시퀀스 [batch_size, seq_len, n_stocks]
            x_probs: 상승확률 [batch_size, pred_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # TCN으로 수익률 시퀀스 처리
        output = self.tcn(x_returns.transpose(1, 2))
        output = self.tempmaxpool(output).squeeze(-1)
        
        # 상승확률 처리
        # 확률 데이터가 2차원이면 그대로 사용
        if len(x_probs.shape) == 2:
            prob_features = self.prob_encoder(x_probs)  # [batch_size, n_stocks]
        # 3차원이면 마지막 시점의 확률 사용
        else:
            prob_features = self.prob_encoder(x_probs[:, -1, :])  # [batch_size, n_stocks]
        
        # 특성 결합
        combined = torch.cat([output, prob_features], dim=1)
        
        # 종목 선택
        attention_scores = self.attention(combined)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Top-k 종목 선택
        topk_values, topk_indices = torch.topk(attention_weights, self.n_select, dim=1)
        
        # 마스크 생성
        mask = torch.zeros_like(attention_weights).scatter_(1, topk_indices, 1.0)
        
        # 선택된 종목에 대한 비중 계산
        scores = self.score_layers(combined)
        weights = F.softmax(scores, dim=-1)
        
        # 선택되지 않은 종목은 0으로 마스킹
        masked_weights = weights * mask
        
        # 비중 재조정
        normalized_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 최소/최대 비중 제약 적용
        final_weights = torch.stack([
            self.rebalance(w, self.lb, self.ub) 
            for w in normalized_weights
        ])
        
        return final_weights


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)