import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PortfolioGRU(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        GRU 기반 포트폴리오 최적화 모델
        
        Args:
            n_layers: GRU 레이어 수
            hidden_dim: 은닉층 차원
            n_stocks: 전체 종목 수
            dropout_p: 드롭아웃 비율
            bidirectional: 양방향 GRU 여부
            lb: 종목별 최소 비중
            ub: 종목별 최대 비중
            n_select: 선택할 종목 수 (None인 경우 n_stocks 사용)
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lb = lb
        self.ub = ub
        self.n_select = n_select if n_select is not None else n_stocks
        self.n_stocks = n_stocks
        
        # Score Block (h1)
        self.gru = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.scale = 2 if bidirectional else 1
        
        # 종목 선택을 위한 attention 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.scale, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
        )
        
        # 선택된 종목의 비중 결정을 위한 레이어
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.scale, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
        )

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
        output, _ = self.gru(x)
        h_t = output[:, -1, :]  # 마지막 타임스텝의 특성
        
        # 종목 선택
        attention_scores = self.attention(h_t)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Top-k 종목 선택
        topk_values, topk_indices = torch.topk(attention_weights, self.n_select, dim=1)
        
        # 마스크 생성
        mask = torch.zeros_like(attention_weights).scatter_(1, topk_indices, 1.0)
        
        # 선택된 종목에 대한 비중 계산
        scores = self.score_layers(h_t)
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

class PortfolioGRUWithProb(PortfolioGRU):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        상승확률을 활용하는 GRU 기반 포트폴리오 최적화 모델
        
        Args:
            n_layers: GRU 레이어 수
            hidden_dim: 은닉층 차원
            n_stocks: 전체 종목 수
            dropout_p: 드롭아웃 비율
            bidirectional: 양방향 GRU 여부
            lb: 종목별 최소 비중
            ub: 종목별 최대 비중
            n_select: 선택할 종목 수 (None인 경우 n_stocks 사용)
        """
        super().__init__(n_layers, hidden_dim, n_stocks, dropout_p, bidirectional, lb, ub, n_select)
        
        # 상승확률 인코딩을 위한 GRU
        self.gru_prob = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 결합된 특성을 처리하기 위한 레이어 수정
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.scale * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
        )
        
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.scale * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
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
        # 수익률과 확률 시퀀스 처리
        returns_out, _ = self.gru(x_returns)
        prob_out, _ = self.gru_prob(x_probs)
        
        # 마지막 타임스텝의 특성 추출
        h_returns = returns_out[:, -1, :]
        h_prob = prob_out[:, -1, :]
        
        # 특성 결합
        combined = torch.cat([h_returns, h_prob], dim=1)
        
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