import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_stocks, dropout_p=0.3, bidirectional=False,
                 lb=0, ub=0.2, n_select=50):
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
            n_select: 선택할 종목 수
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_select = n_select
        self.lb = lb
        self.ub = ub
        self.n_stocks = n_stocks
        
        # GRU 레이어
        self.gru = nn.GRU(
            n_stocks, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout_p)
        self.scale = 2 if bidirectional else 1
        
        # 종목 선택을 위한 레이어
        self.selection_layer = nn.Linear(hidden_dim * self.scale, n_stocks)
        
        # 선택된 종목의 비중 결정을 위한 레이어
        self.weight_layer = nn.Linear(hidden_dim * self.scale, n_select)
        self.swish = nn.SiLU()

    def forward(self, x, x_probs=None):
        """
        순전파
        
        Args:
            x: 입력 시퀀스 (batch_size, seq_len, n_stocks)
            x_probs: 상승확률 (사용하지 않음)
            
        Returns:
            포트폴리오 비중 (batch_size, n_stocks)
        """
        batch_size = x.size(0)
        init_h = torch.zeros(self.n_layers * self.scale, batch_size, self.hidden_dim).to(x.device)
        
        # GRU 통과
        x, _ = self.gru(x, init_h)
        h_t = x[:, -1, :]  # 마지막 타임스텝의 은닉 상태
        
        # 종목 선택
        selection_scores = self.selection_layer(self.dropout(h_t))
        selection_scores = torch.sigmoid(selection_scores)
        
        # Top-k 종목 선택
        _, top_indices = torch.topk(selection_scores, self.n_select, dim=1)
        
        # 선택된 종목에 대한 비중 계산
        weights = self.weight_layer(self.dropout(h_t))
        weights = self.swish(weights)
        weights = F.softmax(weights, dim=-1)
        
        # 최종 포트폴리오 비중 생성
        final_weights = torch.zeros(batch_size, self.n_stocks).to(x.device)
        for i in range(batch_size):
            final_weights[i].scatter_(0, top_indices[i], weights[i])
        
        # 최소/최대 비중 제약 적용
        final_weights = torch.stack([
            self.rebalance(batch, self.lb, self.ub) 
            for batch in final_weights
        ])
        
        return final_weights

    def rebalance(self, weight, lb, ub):
        """
        선택된 종목에 대해서만 rebalancing 수행
        
        Args:
            weight: 초기 비중
            lb: 최소 비중
            ub: 최대 비중
            
        Returns:
            재조정된 비중
        """
        selected_mask = (weight > 0).float()
        old = weight * selected_mask
        
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where((weight_clamped != ub) & (selected_mask == 1))[0]]
            if len(nominees) == 0:
                break
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where((weight_clamped != ub) & (selected_mask == 1))[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        
        return weight_clamped

class GRUWithProb(GRU):
    def __init__(self, n_layers, hidden_dim, n_stocks, dropout_p=0.3, bidirectional=False,
                 lb=0, ub=0.2, n_select=50):
        """
        상승확률을 활용하는 GRU 기반 포트폴리오 최적화 모델
        """
        super().__init__(n_layers, hidden_dim, n_stocks, dropout_p, bidirectional, lb, ub, n_select)
        
        # 상승확률 인코딩을 위한 레이어
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_stocks, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )
        
        # 결합된 특성을 처리하기 위한 레이어 수정
        self.selection_layer = nn.Linear(hidden_dim * (self.scale + 1), n_stocks)
        self.weight_layer = nn.Linear(hidden_dim * (self.scale + 1), n_select)

    def forward(self, x_returns, x_probs):
        """
        순전파
        
        Args:
            x_returns: 수익률 시퀀스 (batch_size, seq_len, n_stocks)
            x_probs: 상승확률 (batch_size, pred_len, n_stocks)
            
        Returns:
            포트폴리오 비중 (batch_size, n_stocks)
        """
        batch_size = x_returns.size(0)
        init_h = torch.zeros(self.n_layers * self.scale, batch_size, self.hidden_dim).to(x_returns.device)
        
        # GRU로 수익률 시퀀스 처리
        x_returns, _ = self.gru(x_returns, init_h)
        h_returns = x_returns[:, -1, :]
        
        # 상승확률 처리 (첫 번째 예측 시점의 확률 사용)
        h_probs = self.prob_encoder(x_probs[:, 0, :])
        
        # 특성 결합
        combined = torch.cat([h_returns, h_probs], dim=1)
        
        # 종목 선택
        selection_scores = self.selection_layer(self.dropout(combined))
        selection_scores = torch.sigmoid(selection_scores)
        
        # Top-k 종목 선택
        _, top_indices = torch.topk(selection_scores, self.n_select, dim=1)
        
        # 선택된 종목에 대한 비중 계산
        weights = self.weight_layer(self.dropout(combined))
        weights = self.swish(weights)
        weights = F.softmax(weights, dim=-1)
        
        # 최종 포트폴리오 비중 생성
        final_weights = torch.zeros(batch_size, self.n_stocks).to(x_returns.device)
        for i in range(batch_size):
            final_weights[i].scatter_(0, top_indices[i], weights[i])
        
        # 최소/최대 비중 제약 적용
        final_weights = torch.stack([
            self.rebalance(batch, self.lb, self.ub) 
            for batch in final_weights
        ])
        
        return final_weights

if __name__ == "__main__":
    pass