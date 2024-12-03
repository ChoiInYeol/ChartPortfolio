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
        # GRU를 통한 시계열 특성 추출
        self.gru = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.scale = 2 if bidirectional else 1
        
        # 종목별 score 생성을 위한 레이어
        self.score_layer = nn.Sequential(
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
        # Score Block (h1): 시계열 특성 추출
        output, _ = self.gru(x)
        h_t = output[:, -1, :]  # 마지막 타임스텝의 특성
        
        # 종목별 score 생성
        scores = self.score_layer(h_t)
        
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
        
        # Score Block (h1)
        # 1. 상승확률을 위한 GRU
        self.gru_prob = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 2. 결합된 특성으로부터 score 생성을 위한 레이어
        self.score_layer = nn.Sequential(
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
        # Score Block (h1): 시계열 특성 추출
        # 1. 수익률 시퀀스 처리
        returns_out, _ = self.gru(x_returns)
        h_returns = returns_out[:, -1, :]
        
        # 2. 상승확률 처리
        # 확률 데이터가 단일 시점이면 차원 추가
        if len(x_probs.shape) == 2:
            x_probs = x_probs.unsqueeze(1)  # [batch_size, 1, n_stocks]
        prob_out, _ = self.gru_prob(x_probs)
        h_prob = prob_out[:, -1, :]
        
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
        
        # 3. Maximum Position 제약
        weights = torch.clamp(weights, self.lb, self.ub)
        
        # 4. 정규화 (sum to 1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights