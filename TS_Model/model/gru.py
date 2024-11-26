import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class PortfolioGRU(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int = 50,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        constraints: Dict[str, Any] = None,
        lb: float = 0.0,
        ub: float = 0.1
    ):
        """
        Score Block과 Portfolio Block으로 구성된 GRU 기반 포트폴리오 최적화 모델
        
        Args:
            n_layers: GRU 레이어 수
            hidden_dim: 은닉층 차원
            n_stocks: 자산 수
            dropout_p: 드롭아웃 비율
            bidirectional: 양방향 GRU 여부
            constraints: 포트폴리오 제약조건
            lb: 최소 비중
            ub: 최대 비중
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lb = lb
        self.ub = ub
        self.constraints = constraints or {}
        
        # Score Block (h1)
        self.gru = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.scale = 2 if bidirectional else 1
        
        # 점수 생성을 위한 레이어
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.scale, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # Score Block (h1)
        output, _ = self.gru(x)
        h_t = output[:, -1, :]  # 마지막 타임스텝의 특성
        
        # 자산별 점수 생성
        scores = self.score_layers(h_t)
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights
    
    def convert_scores_to_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """
        점수를 포트폴리오 가중치로 변환
        
        Args:
            scores: 자산별 점수 [batch_size, n_stocks]
            
        Returns:
            제약조건을 만족하는 포트폴리오 비중 [batch_size, n_stocks]
        """
        device = scores.device
        
        # 1. Cardinality 제약 적용
        if 'CARDINALITY' in self.constraints:
            k = self.constraints['CARDINALITY']
            values, indices = torch.topk(scores.abs(), k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
            scores = scores * mask
        
        # 2. Long-only 제약 적용
        if self.constraints.get('LONG_ONLY', True):
            scores = scores.clone()
            scores[scores < 0] = -float('inf')
        
        # 3. Maximum position 제약 적용
        if 'MAX_POSITION' in self.constraints:
            max_pos = self.constraints['MAX_POSITION']
            max_log = torch.log(torch.tensor(max_pos, device=device)) + 1
            scores = torch.clamp(scores, max=max_log)
        
        # 4. 가중치 변환 및 정규화
        weights = F.softmax(scores, dim=-1)
        
        # 5. 최종 리밸런싱
        weights = torch.stack([
            self.rebalance(w, self.lb, self.ub)
            for w in weights
        ])
        
        return weights

    def rebalance(self, weight: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
        """기존 리밸런싱 로직 유지"""
        weight_clamped = weight.clamp(lb, ub)
        total_excess = weight_clamped.sum() - 1.0

        while total_excess > 1e-6:
            surplus = weight_clamped - ub
            surplus[surplus < 0] = 0
            total_surplus = surplus.sum()

            if total_surplus > 0:
                weight_clamped -= surplus / total_surplus * total_excess
            else:
                weight_clamped -= total_excess / len(weight_clamped)

            weight_clamped = weight_clamped.clamp(lb, ub)
            total_excess = weight_clamped.sum() - 1.0

        return weight_clamped

class PortfolioGRUWithProb(PortfolioGRU):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_stocks: int = 50,
        dropout_p: float = 0.3,
        bidirectional: bool = False,
        constraints: Dict[str, Any] = None,
        lb: float = 0.0,
        ub: float = 0.1
    ):
        """상승확률을 활용하는 GRU 기반 포트폴리오 최적화 모델"""
        super().__init__(
            n_layers, hidden_dim, n_stocks,
            dropout_p, bidirectional, constraints,
            lb, ub
        )
        
        # 상승확률 인코딩을 위한 GRU
        self.gru_prob = nn.GRU(
            n_stocks, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 결합된 특성을 처리하기 위한 점수 레이어
        self.score_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.scale * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_stocks)
        )

    def forward(self, x: torch.Tensor, prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 수익률 시퀀스 [batch_size, seq_len, n_stocks]
            prob: 상승확률 [batch_size, seq_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        if prob is None:
            prob = torch.zeros_like(x)
            
        # 수익률과 확률 시퀀스 처리
        returns_out, _ = self.gru(x)
        prob_out, _ = self.gru_prob(prob)
        
        # 마지막 타임스텝의 특성 추출
        h_returns = returns_out[:, -1, :]
        h_prob = prob_out[:, -1, :]
        
        # 특성 결합
        combined = torch.cat([h_returns, h_prob], dim=1)
        
        # 자산별 점수 생성
        scores = self.score_layers(combined)
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights