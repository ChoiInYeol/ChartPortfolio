import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gru import GRU
from model.tcn import TCN
from model.transformer import Transformer

class CNN(nn.Module):
    def __init__(self, img_height, img_width):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc_input_dim = self._get_fc_input_dim(img_height, img_width)
        self.fc = nn.Linear(self.fc_input_dim, 1)

    def _get_fc_input_dim(self, img_height, img_width):
        x = torch.randn(1, 1, img_height, img_width)
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
    
class Multimodal(nn.Module):
    def __init__(self, model_type, model_params, lb, ub, img_height, img_width):
        super().__init__()
        self.model_type = model_type
        self.lb = lb
        self.ub = ub

        if model_type.lower() == 'gru':
            self.model = GRU(**model_params)
        elif model_type.lower() == 'tcn':
            self.model = TCN(**model_params)
        elif model_type.lower() == 'transformer':
            self.model = Transformer(**model_params)
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")

        self.cnn = CNN(img_height, img_width)
        self.combine = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x_num, x_img):
        batch_size, seq_len, n_stocks = x_num.shape
        numerical_features = self.model(x_num)
        cnn_features = self.cnn(x_img.view(-1, 1, x_img.shape[2], x_img.shape[3])).view(batch_size, n_stocks)
        
        portfolio_weights = self.combine[0] * numerical_features + self.combine[1] * cnn_features
        portfolio_weights = F.softmax(portfolio_weights, dim=1)
        portfolio_weights = self.rebalance_batch(portfolio_weights, self.lb, self.ub)
        
        return portfolio_weights, cnn_features

    def rebalance_batch(self, weights, lb, ub):
        weights_clamped = torch.clamp(weights, lb, ub)
        leftover = (weights - weights_clamped).sum(dim=1, keepdim=True)
        nominees = torch.where(weights_clamped != ub, weights_clamped, torch.zeros_like(weights_clamped))
        nominees_sum = nominees.sum(dim=1, keepdim=True)
        gift = torch.where(nominees_sum > 0, leftover * (nominees / nominees_sum), torch.zeros_like(nominees))
        weights_clamped = torch.clamp(weights_clamped + gift, lb, ub)
        return weights_clamped