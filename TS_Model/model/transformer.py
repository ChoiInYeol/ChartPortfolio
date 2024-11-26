import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple, Union, Any, Dict

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
        constraints: Dict[str, Any] = None
    ):
        """
        Transformer 기반 포트폴리오 최적화 모델
        
        Args:
            n_feature: 입력 특성 수
            n_timestep: 시계열 길이
            n_layer: Transformer 레이어 수
            n_head: Attention 헤드 수
            n_dropout: 드롭아웃 비율
            n_output: 출력 차원 (종목 수)
            constraints: 포트폴리오 제약조건
                - long_only: bool
                - max_position: float
                - cardinality: int
                - leverage: float
        """
        super().__init__()
        self.n_stocks = n_output
        self.constraints = constraints or {}
        
        # Score Block (h1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_feature,
            nhead=n_head,
            dim_feedforward=4*n_feature,
            dropout=n_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )
        
        self.score_layer = nn.Linear(n_feature, n_output)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # Score Block (h1)
        output = self.transformer(x)
        output = output[:, -1, :]  # 마지막 시점의 출력 사용
        
        # 자산별 점수 생성
        scores = self.score_layer(output)
        
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


class PortfolioTransformerWithProb(PortfolioTransformer):
    def __init__(
        self,
        n_feature: int,
        n_timestep: int,
        n_layer: int,
        n_head: int,
        n_dropout: float,
        n_output: int,
        constraints: Dict[str, Any] = None
    ):
        """상승확률을 활용하는 Transformer 기반 포트폴리오 최적화 모델"""
        super().__init__(
            n_feature, n_timestep, n_layer,
            n_head, n_dropout, n_output, constraints
        )
        
        # 상승확률 인코딩을 위한 레이어
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_output, n_feature),
            nn.SiLU(),
            nn.Dropout(n_dropout)
        )
        
        # 결합된 특성을 처리하기 위한 레이어
        self.score_layer = nn.Linear(n_feature * 2, n_output)

    def forward(self, x_returns: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x_returns: 수익률 시퀀스 [batch_size, seq_len, n_stocks]
            x_probs: 상승확률 [batch_size, pred_len, n_stocks]
            
        Returns:
            포트폴리오 비중 [batch_size, n_stocks]
        """
        # Transformer로 수익률 시퀀스 처리
        output = self.transformer(x_returns)
        output = output[:, -1, :]  # 마지막 시점의 출력 사용
        
        # 상승확률 처리 (첫 번째 예측 시점의 확률 사용)
        prob_features = self.prob_encoder(x_probs[:, 0, :])
        
        # 특성 결합
        combined = torch.cat([output, prob_features], dim=1)
        
        # 자산별 점수 생성
        scores = self.score_layer(combined)
        
        # Portfolio Block (h2)
        weights = self.convert_scores_to_weights(scores)
        
        return weights