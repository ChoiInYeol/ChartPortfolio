import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class PortfolioGRU(nn.Module):
    """
    GRU 기반 포트폴리오 최적화 모델
    
    Score Block (h1)과 Portfolio Block (h2)로 구성된 end-to-end 프레임워크
    """
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
            n_stocks: 주식 종목 수
            dropout_p: 드롭아웃 비율
            bidirectional: 양방향 GRU 사용 여부
            constraints: 포트폴리오 제약조건 딕셔너리
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_stocks = n_stocks
        self.constraints = constraints or {}
        
        # Score Block (h1): GRU + FC layers
        self.gru = nn.GRU(
            n_stocks, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        self.scale = 2 if bidirectional else 1
        self.score_generator = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * self.scale, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_stocks)  # Generate scores for each asset
        )
        
    def h1_score_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score Block: 각 자산에 대한 점수 생성
        
        Args:
            x: 입력 시계열 데이터 [batch_size, seq_len, n_stocks]
            
        Returns:
            scores: 각 자산에 대한 점수 [batch_size, n_stocks]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state
        init_h = torch.zeros(
            self.n_layers * self.scale,
            batch_size,
            self.hidden_dim
        ).to(device)
        
        # Get final hidden state
        x, _ = self.gru(x, init_h)
        h_t = x[:, -1, :]  # Use final timestep output
        
        # Generate scores
        scores = self.score_generator(h_t)
        return scores
    
    def h2_portfolio_block(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Portfolio Block: 점수를 포트폴리오 가중치로 변환
        
        Args:
            scores: 각 자산에 대한 점수 [batch_size, n_stocks]
            
        Returns:
            weights: 포트폴리오 가중치 [batch_size, n_stocks]
        """
        # Get constraints
        max_pos = self.constraints.get('MAX_POSITION', None)
        min_pos = self.constraints.get('MIN_POSITION', 0.0)
        k = self.constraints.get('CARDINALITY', self.n_stocks)
        leverage = self.constraints.get('LEVERAGE', 1.0)
        
        if max_pos and k < self.n_stocks:
            # Maximum Position + Cardinality + Leverage constraints
            weights = self._apply_max_card_lev_constraints(
                scores, max_pos, k, leverage
            )
        elif max_pos:
            # Maximum Position constraint
            weights = self._apply_max_position_constraint(
                scores, max_pos, leverage
            )
        elif k < self.n_stocks:
            # Cardinality constraint
            weights = self._apply_cardinality_constraint(
                scores, k, leverage
            )
        else:
            # Basic long-only constraint with leverage
            weights = leverage * F.softmax(scores, dim=-1)
        
        return weights
    
    def _apply_max_position_constraint(
        self, 
        scores: torch.Tensor,
        max_pos: float,
        leverage: float
    ) -> torch.Tensor:
        """Maximum Position 제약조건 적용"""
        a = (1 - max_pos) / (self.n_stocks * max_pos - 1)
        phi_a = lambda x: (a + 1) / (1 + torch.exp(-x))
        
        abs_scores = torch.abs(scores)
        transformed_scores = phi_a(abs_scores)
        weights = leverage * transformed_scores / transformed_scores.sum(dim=-1, keepdim=True)
        return weights
    
    def _apply_cardinality_constraint(
        self,
        scores: torch.Tensor,
        k: int,
        leverage: float
    ) -> torch.Tensor:
        """Cardinality 제약조건 적용"""
        abs_scores = torch.abs(scores)
        
        # Get top-k threshold
        topk_values, _ = torch.topk(abs_scores, k, dim=-1)
        threshold = topk_values[:, -1].unsqueeze(-1)
        
        # Create mask for top-k scores
        mask = (abs_scores >= threshold).float()
        
        # Apply mask and normalize
        masked_scores = torch.exp(abs_scores) * mask
        weights = leverage * masked_scores / masked_scores.sum(dim=-1, keepdim=True)
        return weights
    
    def _apply_max_card_lev_constraints(
        self,
        scores: torch.Tensor,
        max_pos: float,
        k: int,
        leverage: float
    ) -> torch.Tensor:
        """Maximum Position + Cardinality + Leverage 제약조건 통합 적용"""
        abs_scores = torch.abs(scores)
        
        # Maximum position transformation
        a = (1 - max_pos) / (self.n_stocks * max_pos - 1)
        phi_a = lambda x: (a + 1) / (1 + torch.exp(-x))
        transformed_scores = phi_a(abs_scores)
        
        # Cardinality mask
        topk_values, _ = torch.topk(abs_scores, k, dim=-1)
        threshold = topk_values[:, -1].unsqueeze(-1)
        mask = (abs_scores >= threshold).float()
        
        # Combine constraints
        final_scores = transformed_scores * mask
        weights = leverage * final_scores / final_scores.sum(dim=-1, keepdim=True)
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass
        
        Args:
            x: 입력 시계열 데이터 [batch_size, seq_len, n_stocks]
            
        Returns:
            weights: 포트폴리오 가중치 [batch_size, n_stocks]
        """
        # Score Block (h1)
        scores = self.h1_score_block(x)
        
        # Portfolio Block (h2)
        weights = self.h2_portfolio_block(scores)
        
        return weights

class PortfolioGRUWithProb(PortfolioGRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Simple probability encoder
        self.prob_encoder = nn.Sequential(
            nn.Linear(self.n_stocks, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(kwargs.get('dropout_p', 0.3))
        )
        
        # Update fc layer for combined features
        self.fc = nn.Linear(self.hidden_dim * (self.scale + 1), self.n_stocks)

    def forward(self, x_returns: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
        batch_size = x_returns.size(0)
        init_h = torch.zeros(
            self.n_layers * self.scale,
            batch_size,
            self.hidden_dim
        ).to(x_returns.device)
        
        # Process returns
        x, _ = self.gru(x_returns, init_h)
        h_returns = x[:, -1, :]
        
        # Process probabilities
        h_probs = self.prob_encoder(x_probs.mean(dim=1))
        
        # Combine features
        combined = torch.cat([h_returns, h_probs], dim=1)
        
        # Generate weights
        logit = self.fc(self.dropout(combined))
        logit = self.swish(logit)
        weights = F.softmax(logit, dim=-1)
        
        # Apply constraints
        if self.constraints:
            weights = self.apply_constraints(weights)
        
        return weights

if __name__ == "__main__":
    pass