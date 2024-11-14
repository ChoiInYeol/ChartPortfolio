import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, n_feature, n_output, num_channels,
                 kernel_size, n_dropout, n_timestep, lb, ub, n_select=5):
        super(TCN, self).__init__()
        self.input_size = n_feature
        
        # TCN의 채널 수를 점진적으로 증가
        self.tcn = TemporalConvNet(
            num_inputs=n_feature,
            num_channels=num_channels,  # [64, 128, 256] 등으로 설정
            kernel_size=kernel_size,
            dropout=n_dropout
        )
        
        # 특성 추출을 위한 projection layer 추가
        self.proj = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.ReLU(),
            nn.Dropout(n_dropout)
        )
        
        # 종목 선택과 비중 계산을 위한 레이어
        self.attention = nn.Linear(num_channels[-1], n_output)
        self.fc = nn.Linear(num_channels[-1], n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)
        
        self.lb = lb
        self.ub = ub
        self.n_select = n_select

    def forward(self, x, x_probs=None):
        # Feature extraction
        output = self.tcn(x.transpose(1, 2))  # [batch, channel, time]
        output = self.tempmaxpool(output)     # [batch, channel, 1]
        output = output.squeeze(-1)           # [batch, channel]
        
        # Feature projection
        output = self.proj(output)
        
        # Stock selection
        attention = self.attention(output)
        attention = torch.sigmoid(attention)
        _, top_indices = torch.topk(attention, self.n_select, dim=1)
        mask = torch.zeros_like(attention).scatter_(1, top_indices, 1.0)
        
        # Weight calculation
        out = self.fc(output)
        out = F.softmax(out, dim=1)
        out = out * mask
        out = out / (out.sum(dim=1, keepdim=True) + 1e-8)
        
        # Rebalancing with constraints
        out = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in out])
        return out

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped


class TCNWithProb(TCN):
    def __init__(self, n_feature, n_output, num_channels,
                 kernel_size, n_dropout, n_timestep, lb, ub, n_select=5):
        super().__init__(n_feature, n_output, num_channels,
                        kernel_size, n_dropout, n_timestep, lb, ub, n_select)
        
        # Probability encoder
        self.prob_encoder = nn.Sequential(
            nn.Linear(n_output, num_channels[-1]),
            nn.ReLU(),
            nn.Dropout(n_dropout)
        )
        
        # Modify attention and fc layers for combined features
        self.attention = nn.Linear(num_channels[-1] * 2, n_output)
        self.fc = nn.Linear(num_channels[-1] * 2, n_output)

    def forward(self, x, x_probs):
        # Feature extraction from returns
        output = self.tcn(x.transpose(1, 2))
        output = self.tempmaxpool(output).squeeze(-1)
        
        # Process probability features
        prob_features = self.prob_encoder(x_probs[:, 0, :])
        
        # Combine features
        combined = torch.cat([output, prob_features], dim=1)
        
        # Stock selection
        attention = self.attention(combined)
        attention = torch.sigmoid(attention)
        _, top_indices = torch.topk(attention, self.n_select, dim=1)
        mask = torch.zeros_like(attention).scatter_(1, top_indices, 1.0)
        
        # Weight calculation
        out = self.fc(combined)
        out = F.softmax(out, dim=1)
        out = out * mask
        out = out / (out.sum(dim=1, keepdim=True) + 1e-8)
        
        # Rebalancing with constraints
        out = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in out])
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)