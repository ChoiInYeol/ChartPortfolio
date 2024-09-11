import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, n_stocks, n_output, kernel_size, n_dropout, n_timestep, hidden_size, level, lb, ub, multimodal=False, cnn_output_dim=0, verbose=False):
        super(TCN, self).__init__()
        self.verbose = verbose
        self.input_size = n_stocks
        self.lb = lb
        self.ub = ub
        self.multimodal = multimodal

        input_dim = n_stocks + n_stocks * cnn_output_dim if multimodal else n_stocks
        num_channels = [hidden_size] * (level - 1) + [n_timestep]
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=n_dropout)
        self.fc = nn.Linear(num_channels[-1], n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)

    def forward(self, x):
        if self.verbose:
            print(f"TCN input shape: {x.shape}")
        output = self.tcn(x.transpose(1, 2))
        if self.verbose:
            print(f"After TCN shape: {output.shape}")
        output = self.tempmaxpool(output).squeeze(-1)
        if self.verbose:
            print(f"After MaxPool shape: {output.shape}")
        out = self.fc(output)
        if self.verbose:
            print(f"After FC shape: {out.shape}")
        out = F.softmax(out, dim=1)
        out = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in out])
        if self.verbose:
            print(f"Final output shape: {out.shape}")
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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        """
        chomp_size: zero padding size
        """
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