import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class PortfolioGRU(nn.Module):
    """포트폴리오 최적화를 위한 GRU 모델"""
    
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        constraints: Dict[str, Any] = None
    ):
        """
        Args:
            n_layers: GRU 레이어 수
            hidden_dim: 은닉층 차원
            n_stocks: 전체 종목 수
            dropout_p: 드롭아웃 비율
            bidirectional: 양방향 GRU 여부
            constraints: 포트폴리오 제약조건
                - long_only: bool
                - max_position: float
                - cardinality: int
                - leverage: float
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_stocks = n_stocks
        self.constraints = constraints or {}
        
        # Score Block (h1)
        self.gru = nn.GRU(
            n_stocks, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout_p)
        self.scale = 2 if bidirectional else 1
        self.score_layer = nn.Linear(hidden_dim * self.scale, n_stocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, n_stocks]
            
        Returns:
            포트폴리오 가중치 [batch_size, n_stocks]
        """
        # Score Block (h1)
        batch_size = x.size(0)
        init_h = torch.zeros(
            self.n_layers * self.scale,
            batch_size,
            self.hidden_dim
        ).to(x.device)
        
        x, _ = self.gru(x, init_h)
        h_t = x[:, -1, :]  # 마지막 시점의 은닉 상태
        
        # 자산별 점수 생성
        scores = self.score_layer(self.dropout(h_t))
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights
    
    def convert_scores_to_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """점수를 포트폴리오 가중치로 변환합니다."""
        if self.constraints.get('long_only', True):
            # Long-only constraint
            weights = torch.softmax(scores, dim=-1)
            
        else:
            # General case allowing short positions
            weights = torch.tanh(scores)  # [-1, 1] 범위로 제한
            
            # Normalize to satisfy leverage constraint
            leverage = self.constraints.get('leverage', 1.0)
            weights = leverage * weights / weights.abs().sum(dim=-1, keepdim=True)
        
        if 'max_position' in self.constraints:
            # Maximum position constraint
            u = self.constraints['max_position']
            a = (1 - u) / (self.n_stocks * u - 1)
            
            # Generalized sigmoid
            def phi_a(x):
                return (a + 1) / (1 + torch.exp(-x))
            
            weights = torch.sign(scores) * phi_a(scores.abs())
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        if 'cardinality' in self.constraints:
            # Cardinality constraint
            k = self.constraints['cardinality']
            values, indices = torch.topk(scores.abs(), k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights

class PortfolioGRUWithProb(PortfolioGRU):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        constraints: Dict[str, Any] = None
    ):
        """상승확률을 활용하는 GRU 기반 포트폴리오 최적화 모델"""
        super().__init__(
            n_layers, hidden_dim, n_stocks,
            dropout_p, bidirectional, constraints
        )
        
        # 상승확률 인코딩을 위한 레이어
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_stocks, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )
        
        # 결합된 특성을 처리하기 위한 레이어
        self.score_layer = nn.Linear(hidden_dim * (self.scale + 1), n_stocks)

    def forward(self, x_returns: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x_returns: 수익률 시퀀스 [batch_size, seq_len, n_stocks]
            x_probs: 상승확률 [batch_size, pred_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # GRU로 수익률 시퀀스 처리
        batch_size = x_returns.size(0)
        init_h = torch.zeros(
            self.n_layers * self.scale,
            batch_size,
            self.hidden_dim
        ).to(x_returns.device)
        
        x, _ = self.gru(x_returns, init_h)
        h_returns = x[:, -1, :]
        
        # 상승확률 처리 (첫 번째 예측 시점의 확률 사용)
        h_probs = self.prob_encoder(x_probs[:, 0, :])
        
        # 특성 결합
        combined = torch.cat([h_returns, h_probs], dim=1)
        
        # 자산별 점수 생성
        scores = self.score_layer(self.dropout(combined))
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights

if __name__ == "__main__":
    pass