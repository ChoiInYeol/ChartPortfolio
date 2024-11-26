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
        """
        점수를 포트폴리오 가중치로 변환합니다.
        모든 제약조건을 순차적으로 적용하되, 각 단계에서 제약조건들이 유지되도록 합니다.
        
        Args:
            scores: 자산별 점수 [batch_size, n_stocks]
            
        Returns:
            제약조건을 만족하는 포트폴리오 가중치 [batch_size, n_stocks]
        """
        device = scores.device
        
        # 1. Cardinality 제약 적용 (상위 k개 종목 선택)
        if 'CARDINALITY' in self.constraints:
            k = self.constraints['CARDINALITY']
            values, indices = torch.topk(scores.abs(), k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
            scores = scores * mask
        
        # 2. Long-only 제약 적용
        if self.constraints.get('LONG_ONLY', True):
            scores = scores.clone()
            scores[scores < 0] = -float('inf')  # 음수 점수를 -inf로 설정하여 softmax 후 0이 되도록
        
        # 3. Maximum position 제약 적용
        if 'MAX_POSITION' in self.constraints:
            max_pos = self.constraints['MAX_POSITION']
            max_log = torch.log(torch.tensor(max_pos, device=device)) + 1
            scores = torch.clamp(scores, max=max_log)
        
        # 4. 가중치 변환 및 정규화
        weights = torch.softmax(scores, dim=-1)
        
        # 5. Maximum position 추가 검증 및 조정
        if 'MAX_POSITION' in self.constraints:
            max_pos = self.constraints['MAX_POSITION']
            weights = torch.clamp(weights, max=max_pos)
            # 정규화
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 6. Minimum position 제약 적용
        if 'MIN_POSITION' in self.constraints:
            min_pos = self.constraints['MIN_POSITION']
            weights[weights < min_pos] = 0
            # 다시 정규화
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 7. Leverage 제약 확인
        if 'LEVERAGE' in self.constraints:
            leverage = self.constraints['LEVERAGE']
            leverage_tensor = torch.tensor(leverage, device=device)
            if not torch.allclose(weights.sum(dim=-1), leverage_tensor, rtol=1e-3):
                weights = weights * leverage
        
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
        
        # 상승확률 처리
        h_probs = self.prob_encoder(x_probs)
        
        # 특성 결합
        combined = torch.cat([h_returns, h_probs], dim=1)
        
        # 자산별 점수 생성
        scores = self.score_layer(self.dropout(combined))
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights

if __name__ == "__main__":
    pass