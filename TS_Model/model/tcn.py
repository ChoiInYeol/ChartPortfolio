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
            num_channels: TCN 채널 수 리스트
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
        
        # Score Block (h1)
        # 1. TCN을 통한 시계열 특성 추출
        self.tcn = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=n_dropout
        )
        
        # 2. 종목별 score 생성을 위한 레이어
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
        # Score Block (h1): 시계열 특성 추출
        # 1. TCN을 통한 특성 추출
        output = self.tcn(x.transpose(1, 2))  # [batch_size, channels, seq_len]
        features = self.tempmaxpool(output).squeeze(-1)  # [batch_size, channels]
        
        # 2. 종목별 score 생성
        scores = self.score_layer(features)
        
        # Portfolio Block (h2): 제약조건을 만족하는 가중치 생성
        
        # 1. Cardinality 제약 처리
        if self.n_select < self.n_stocks:
            topk_values, topk_indices = torch.topk(scores, self.n_select, dim=1)
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            scores = scores * mask
        
        # 2. Long-only 제약을 위한 softmax
        weights = F.softmax(scores, dim=-1)
        
        # 3. Maximum Position 제약
        weights = torch.clamp(weights, self.lb, self.ub)
        
        # 4. 정규화 (sum to 1)
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
        """
        super().__init__(
            n_feature, n_output, num_channels,
            kernel_size, n_dropout, n_timestep,
            lb, ub, n_select
        )
        
        # Score Block (h1)
        # 1. 상승확률을 위한 TCN
        self.tcn_prob = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=n_dropout
        )
        
        # 2. 결합된 특성으로부터 score 생성을 위한 레이어
        self.score_layer = nn.Sequential(
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
        # Score Block (h1): 시계열 특성 추출
        # 1. 수익률 시퀀스 처리
        returns_out = self.tcn(x_returns.transpose(1, 2))  # [batch_size, channels, seq_len]
        
        # 2. 상승확률 처리
        if len(x_probs.shape) == 2:
            x_probs = x_probs.unsqueeze(1)  # [batch_size, 1, n_stocks]
            
        # 상승확률 데이터의 시퀀스 길이가 1인 경우 처리
        if x_probs.shape[1] == 1:
            # TCN 출력을 직접 사용
            prob_out = self.tcn_prob(x_probs.transpose(1, 2))  # [batch_size, channels, 1]
            h_prob = prob_out.squeeze(-1)  # [batch_size, channels]
        else:
            prob_out = self.tcn_prob(x_probs.transpose(1, 2))
            h_prob = self.tempmaxpool(prob_out).squeeze(-1)
        
        # 수익률 특성도 동일한 방식으로 처리
        if returns_out.shape[-1] == 1:
            h_returns = returns_out.squeeze(-1)
        else:
            h_returns = self.tempmaxpool(returns_out).squeeze(-1)
        
        # 3. 특성 결합
        combined = torch.cat([h_returns, h_prob], dim=1)
        
        # 4. 종목별 score 생성
        scores = self.score_layer(combined)
        
        # Portfolio Block (h2): 제약조건을 만족하는 가중치 생성
        
        # 1. Cardinality 제약 처리
        if self.n_select < self.n_stocks:
            topk_values, topk_indices = torch.topk(scores, self.n_select, dim=1)
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            scores = scores * mask
        
        # 2. Long-only 제약을 위한 softmax
        weights = F.softmax(scores, dim=-1)
        
        # 3. 희소성 처리: 임계값보다 작은 가중치는 0으로 설정
        threshold = 0.02  # 2% 임계값
        weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))
        
        # 4. Maximum Position 제약
        weights = torch.clamp(weights, self.lb, self.ub)
        
        # 5. 정규화 (sum to 1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights


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