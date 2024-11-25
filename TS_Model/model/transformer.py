import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable

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


class Transformer(nn.Module):
    def __init__(self, n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub):
        """
        트랜스포머 기반 포트폴리오 최적화 모델
        
        Args:
            n_feature: 입력 특성 수
            n_timestep: 시계열 길이
            n_layer: 인코더 레이어 수
            n_head: 어텐션 헤드 수
            n_dropout: 드롭아웃 비율
            n_output: 출력 차원 (종목 수)
            lb: 종목별 최소 비중
            ub: 종목별 최대 비중
        """
        super().__init__()
        self.encoder = Encoder(
            input_size=n_feature,
            seq_len=n_timestep,
            N=n_layer,
            heads=n_head,
            dropout=n_dropout
        )
        self.out = nn.Linear(n_feature, n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)
        self.lb = lb
        self.ub = ub
        self.swish = nn.SiLU()

    def forward(self, x, returnWeights=False):
        """
        순전파
        
        Args:
            x: 입력 시퀀스 (batch_size, seq_len, n_stocks)
            returnWeights: 어텐션 가중치 반환 여부
            
        Returns:
            포트폴리오 비중 (batch_size, n_stocks)
            어텐션 가중치 (returnWeights=True인 경우)
        """
        mask = creatMask(x.shape[0], x.shape[1]).to(device)
        
        if returnWeights:
            e_outputs, weights = self.encoder(x, mask, returnWeights=True)
        else:
            e_outputs = self.encoder(x, mask)

        e_outputs = self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        
        # 비중 계산
        logit = self.out(e_outputs)
        logit = self.swish(logit)
        logit = F.softmax(logit, dim=1)
        
        # 최소/최대 비중 제약 적용
        final_weights = torch.stack([
            self.rebalance(batch, self.lb, self.ub) 
            for batch in logit
        ])
        
        if returnWeights:
            return final_weights, weights
        return final_weights

    def rebalance(self, weight, lb, ub):
        """
        포트폴리오 비중 재조정
        
        Args:
            weight: 초기 비중
            lb: 최소 비중
            ub: 최대 비중
            
        Returns:
            재조정된 비중
        """
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            if len(nominees) == 0:
                break
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped


class TransformerWithProb(Transformer):
    def __init__(self, n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub):
        """
        상승확률을 활용하는 트랜스포머 기반 포트폴리오 최적화 모델
        """
        super().__init__(n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub)
        
        # 상승확률 인코딩을 위한 레이어
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_output, n_feature),
            nn.SiLU(),
            nn.Dropout(n_dropout)
        )
        
        # 결합된 특성을 처리하기 위한 레이어
        self.out = nn.Linear(n_feature * 2, n_output)

    def forward(self, x_returns, x_probs, returnWeights=False):
        """
        순전파
        
        Args:
            x_returns: 수익률 시퀀스 (batch_size, seq_len, n_stocks)
            x_probs: 상승확률 (batch_size, pred_len, n_stocks)
            returnWeights: 어텐션 가중치 반환 여부
            
        Returns:
            포트폴리오 비중 (batch_size, n_stocks)
            어텐션 가중치 (returnWeights=True인 경우)
        """
        mask = creatMask(x_returns.shape[0], x_returns.shape[1]).to(device)
        
        if returnWeights:
            e_outputs, weights = self.encoder(x_returns, mask, returnWeights=True)
        else:
            e_outputs = self.encoder(x_returns, mask)

        e_outputs = self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        
        # 상승확률 처리 (첫 번째 예측 시점의 확률 사용)
        prob_features = self.prob_encoder(x_probs[:, 0, :])
        
        # 특성 결합
        combined = torch.cat([e_outputs, prob_features], dim=1)
        
        # 비중 계산
        logit = self.out(combined)
        logit = self.swish(logit)
        logit = F.softmax(logit, dim=1)
        
        # 최소/최대 비중 제약 적용
        final_weights = torch.stack([
            self.rebalance(batch, self.lb, self.ub) 
            for batch in logit
        ])
        
        if returnWeights:
            return final_weights, weights
        return final_weights