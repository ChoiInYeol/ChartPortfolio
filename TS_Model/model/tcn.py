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
        
        # 1. 더 깊고 넓은 TCN 구조
        self.tcn = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=[64, 128, 256, 128, 64],  # 점진적으로 확장했다가 축소
            kernel_size=8,  # 더 넓은 수용영역
            dropout=n_dropout
        )
        
        hidden_dim = 64  # 마지막 채널 크기
        
        # 2. 시간적 특성을 더 잘 포착하기 위한 구조
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 3. 종목 간 상관관계를 위한 Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=n_dropout,
            batch_first=True
        )
        
        # 4. 종목 선택을 위한 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 시간 특성과 교차 특성 결합
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(hidden_dim, n_output)
        )
        
        # 5. 비중 결정을 위한 레이어
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(hidden_dim, n_output)
        )

    def forward(self, x: torch.Tensor, x_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파"""
        # 1. TCN 특성 추출
        tcn_out = self.tcn(x.transpose(1, 2))  # [batch_size, channels, seq_len]
        
        # 2. 시간적 특성 추출
        time_features = tcn_out.transpose(1, 2)  # [batch_size, seq_len, channels]
        time_weights = F.softmax(self.time_attention(time_features), dim=1)
        time_context = torch.sum(time_features * time_weights, dim=1)
        
        # 3. 종목 간 상관관계 모델링
        cross_features, _ = self.cross_attention(
            time_features,
            time_features,
            time_features
        )
        cross_context = cross_features[:, -1, :]  # 마지막 시점의 특성
        
        # 4. 특성 결합
        combined = torch.cat([time_context, cross_context], dim=1)
        
        # 5. 종목 선택
        attention_scores = self.attention(combined)
        attention_weights = torch.sigmoid(attention_scores)
        
        # 6. Top-k 종목 선택
        topk_values, topk_indices = torch.topk(attention_weights, self.n_select, dim=1)
        mask = torch.zeros_like(attention_weights).scatter_(1, topk_indices, 1.0)
        
        # 7. 비중 계산
        scores = self.score_layers(combined)
        weights = F.softmax(scores, dim=-1)
        
        # 8. 마스킹 및 정규화
        masked_weights = weights * mask
        normalized_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 9. 최종 제약 적용
        final_weights = torch.stack([
            self.rebalance(w, self.lb, self.ub) 
            for w in normalized_weights
        ])
        
        return final_weights

    def rebalance(self, weight: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
        """포폴리오 비중 재조정"""
        selected_mask = (weight > 0).float()
        weight = weight * selected_mask
        
        weight_clamped = torch.clamp(weight, lb, ub)
        total_excess = weight_clamped.sum() - 1.0

        while abs(total_excess) > 1e-6:
            if total_excess > 0:
                adjustable = (weight_clamped > lb) & (selected_mask == 1)
                if not adjustable.any():
                    break
                adjustment = total_excess / adjustable.sum()
                weight_clamped[adjustable] -= adjustment
            else:
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
        super().__init__(
            n_feature, n_output, num_channels,
            kernel_size, n_dropout, n_timestep,
            lb, ub, n_select
        )
        
        # 1. 상승확률을 위한 TCN
        self.tcn_prob = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=[64, 128, 256, 128, 64],  # 동일한 채널 구조 사용
            kernel_size=kernel_size,
            dropout=n_dropout
        )
        
        hidden_dim = 64  # 마지막 채널 크기
        
        # 2. 상승확률을 위한 시간적 특성 추출기
        self.prob_time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 3. 상승확률을 위한 Cross-attention
        self.prob_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=n_dropout,
            batch_first=True
        )
        
        # 4. 결합된 특성을 위한 Fusion 레이어
        self.combined_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 수익률과 확률 특성 결합
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        # 5. 종목 선택을 위한 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(hidden_dim, n_output)
        )
        
        # 6. 비중 결정을 위한 레이어
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(hidden_dim * 2, n_output)
        )

    def forward(self, x_returns: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
        # 1. 수익률 특성 추출
        returns_out = self.tcn(x_returns.transpose(1, 2))  # [batch_size, channels, seq_len]
        returns_features = returns_out.transpose(1, 2)  # [batch_size, seq_len, channels]
        
        # 2. 수익률 시간적 특성
        returns_weights = F.softmax(self.time_attention(returns_features), dim=1)
        returns_context = torch.sum(returns_features * returns_weights, dim=1)
        
        # 3. 수익률 Cross-attention
        returns_cross, _ = self.cross_attention(
            returns_features,
            returns_features,
            returns_features
        )
        returns_cross_context = returns_cross[:, -1, :]
        
        # 4. 상승확률 특성 추출
        if len(x_probs.shape) == 2:
            x_probs = x_probs.unsqueeze(1)
        prob_out = self.tcn_prob(x_probs.transpose(1, 2))
        prob_features = prob_out.transpose(1, 2)
        
        # 5. 상승확률 시간적 특성
        prob_weights = F.softmax(self.prob_time_attention(prob_features), dim=1)
        prob_context = torch.sum(prob_features * prob_weights, dim=1)
        
        # 6. 상승확률 Cross-attention
        prob_cross, _ = self.prob_cross_attention(
            prob_features,
            prob_features,
            prob_features
        )
        prob_cross_context = prob_cross[:, -1, :]
        
        # 7. 모든 특성 결합
        combined = torch.cat([
            returns_context, returns_cross_context,
            prob_context, prob_cross_context
        ], dim=1)
        fused_features = self.combined_fusion(combined)
        
        # 8. 종목 선택
        attention_scores = self.attention(fused_features)
        attention_weights = torch.sigmoid(attention_scores)
        
        # 9. Top-k 종목 선택
        topk_values, topk_indices = torch.topk(attention_weights, self.n_select, dim=1)
        mask = torch.zeros_like(attention_weights).scatter_(1, topk_indices, 1.0)
        
        # 10. 비중 계산
        scores = self.score_layers(fused_features)
        weights = F.softmax(scores, dim=-1)
        
        # 11. 마스킹 및 정규화
        masked_weights = weights * mask
        normalized_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 12. 최종 제약 적용
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