import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List, Tuple, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def creatMask(batch, sequence_length):
    mask = torch.zeros(batch, sequence_length, sequence_length)
    for i in range(sequence_length):
        mask[:, i, :i + 1] = 1
    return mask


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None, returnWeights=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # print("Scores in attention itself",torch.sum(scores))
    if (returnWeights):
        return output, scores

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, returnWeights=False):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next

        if (returnWeights):
            scores, weights = attention(q, k, v, self.d_k, mask, self.dropout, returnWeights=returnWeights)
            # print("scores",scores.shape,"weights",weights.shape)
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)
        # print("Attention output", output.shape,torch.min(output))
        if (returnWeights):
            return output, weights
        else:
            return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=400, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnWeights=False):
        x2 = self.norm_1(x)
        # print(x2[0,0,0])
        # print("attention input.shape",x2.shape)
        if (returnWeights):
            attenOutput, attenWeights = self.attn(x2, x2, x2, mask, returnWeights=returnWeights)
        else:
            attenOutput = self.attn(x2, x2, x2, mask)
        # print("attenOutput",attenOutput.shape)
        x = x + self.dropout_1(attenOutput)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        if (returnWeights):
            return x, attenWeights
        else:
            return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)

        pe = Variable(self.pe[:, :seq_len], requires_grad=False)

        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, input_size, seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(input_size, seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(input_size, heads, dropout), N)
        self.norm = Norm(input_size)

    def forward(self, src, mask=None, returnWeights=False):
        """
        인코더 forward pass
        
        Args:
            src (torch.Tensor): 입력 시퀀스
            mask (torch.Tensor): 어텐션 마스크
            returnWeights (bool): 어텐션 가중치 반환 여부
        """
        x = self.pe(src)
        weights_list = []
        
        for i in range(self.N):
            if returnWeights:
                x, weights = self.layers[i](x, mask, returnWeights=True)
                weights_list.append(weights)
            else:
                x = self.layers[i](x, mask)
        
        x = self.norm(x)
        
        if returnWeights:
            return x, weights_list
        return x


class PortfolioTransformer(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_timestep: int,
        n_layer: int,
        n_head: int,
        n_dropout: float,
        n_output: int,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        Transformer 기반 포트폴리오 최적화 모델
        
        Args:
            n_feature: 입력 특성 수 (n_stocks와 동일)
            n_timestep: 시계열 길이
            n_layer: Transformer 레이어 수
            n_head: 어텐션 헤드 수
            n_dropout: 드롭아웃 비율
            n_output: 출력 차원 (n_stocks와 동일)
            lb: 최소 포트폴리오 비중
            ub: 최대 포트폴리오 비중
            n_select: 선택할 종목 수 (None인 경우 n_stocks 사용)
        """
        super().__init__()
        self.n_stocks = n_output
        self.lb = lb
        self.ub = ub
        self.n_select = n_select if n_select is not None else n_output
        
        # Score Block (h1)
        # 1. Transformer를 통한 시계열 특성 추출
        self.encoder = Encoder(
            input_size=n_feature,
            seq_len=n_timestep,
            N=n_layer,
            heads=n_head,
            dropout=n_dropout
        )
        
        # 2. 종목별 score 생성을 위한 레이어
        self.score_layer = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.LayerNorm(n_feature),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(n_feature, n_output)
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
        # 1. Transformer를 통한 특성 추출
        mask = torch.ones(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        encoded = self.encoder(x, mask)  # [batch_size, seq_len, n_feature]
        features = self.tempmaxpool(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, n_feature]
        
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
        
        # 3. 희소성 처리: 임계값보다 작은 가중치는 0으로 설정
        threshold = 0.02  # 2% 임계값
        weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))
        
        # 4. Maximum Position 제약
        weights = torch.clamp(weights, self.lb, self.ub)
        
        # 5. 정규화 (sum to 1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights


class PortfolioTransformerWithProb(PortfolioTransformer):
    def __init__(
        self,
        n_feature: int,
        n_timestep: int,
        n_layer: int,
        n_head: int,
        n_dropout: float,
        n_output: int,
        lb: float = 0.0,
        ub: float = 0.1,
        n_select: Optional[int] = None
    ):
        """
        상승확률을 활용하는 Transformer 기반 포트폴리오 최적화 모델
        """
        super().__init__(
            n_feature, n_timestep, n_layer, n_head,
            n_dropout, n_output, lb, ub, n_select
        )
        
        # Score Block (h1)
        # 1. 상승확률을 위한 Transformer
        self.encoder_prob = Encoder(
            input_size=n_feature,
            seq_len=n_timestep,
            N=n_layer,
            heads=n_head,
            dropout=n_dropout
        )
        
        # 2. 결합된 특성으로부터 score 생성을 위한 레이어
        self.score_layer = nn.Sequential(
            nn.Linear(n_feature * 2, n_feature),
            nn.LayerNorm(n_feature),
            nn.SiLU(),
            nn.Dropout(n_dropout),
            nn.Linear(n_feature, n_output)
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
        returns_mask = torch.ones(x_returns.shape[0], x_returns.shape[1], x_returns.shape[1]).to(x_returns.device)
        returns_encoded = self.encoder(x_returns, returns_mask)  # [batch_size, seq_len, n_feature]
        
        # 2. 상승확률 처리
        if len(x_probs.shape) == 2:
            x_probs = x_probs.unsqueeze(1)  # [batch_size, 1, n_stocks]
            
        # 상승확률 데이터의 시퀀스 길이가 1인 경우 처리
        if x_probs.shape[1] == 1:
            # Transformer 출력을 직접 사용
            prob_mask = torch.ones(x_probs.shape[0], 1, 1).to(x_probs.device)
            prob_encoded = self.encoder_prob(x_probs, prob_mask)  # [batch_size, 1, n_feature]
            h_prob = prob_encoded.squeeze(1)  # [batch_size, n_feature]
        else:
            prob_mask = torch.ones(x_probs.shape[0], x_probs.shape[1], x_probs.shape[1]).to(x_probs.device)
            prob_encoded = self.encoder_prob(x_probs, prob_mask)
            h_prob = self.tempmaxpool(prob_encoded.transpose(1, 2)).squeeze(-1)
        
        # 수익률 특성도 동일한 방식으로 처리
        if returns_encoded.shape[1] == 1:
            h_returns = returns_encoded.squeeze(1)
        else:
            h_returns = self.tempmaxpool(returns_encoded.transpose(1, 2)).squeeze(-1)
        
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