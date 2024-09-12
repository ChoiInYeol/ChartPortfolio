import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gru import GRU
from model.tcn import TCN
from model.transformer import Transformer

class CNN(nn.Module):
    def __init__(self, img_height, img_width):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_input_dim = self._get_fc_input_dim(img_height, img_width)
    
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        
        self.fc_binary = nn.Linear(64, 1)

    def _get_fc_input_dim(self, img_height, img_width):
        x = torch.randn(1, 1, img_height, img_width)
        x = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x))))))
        return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        features = F.relu(self.fc1(x))
        
        binary_output = torch.sigmoid(self.fc_binary(features))
        cnn_features = self.fc2(features)
        return binary_output, cnn_features
    
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
        
        # 수치 특성과 CNN 특성의 크기를 확인하고 fusion 레이어의 입력 크기를 조정합니다.
        numerical_feature_size = model_params['n_stocks']  # TCN의 출력 크기
        cnn_feature_size = 128  # CNN의 출력 크기
        
        # 수치 특성과 CNN 특성을 결합하기 위한 선형 레이어
        self.fusion = nn.Linear(numerical_feature_size + cnn_feature_size, 64)
        self.final_fc = nn.Linear(64, model_params['n_stocks'])

    def forward(self, x_num, x_img):
        batch_size, seq_len, n_stocks = x_num.shape
        
        # 수치 데이터 처리
        numerical_features = self.model(x_num)  # (batch_size, n_stocks)
        
        # 이미지 데이터 처리
        x_img_reshaped = x_img.view(batch_size * n_stocks, 1, x_img.shape[2], x_img.shape[3])
        binary_pred, cnn_features = self.cnn(x_img_reshaped)
        cnn_features = cnn_features.view(batch_size, n_stocks, -1)  # (batch_size, n_stocks, cnn_feature_dim)
        cnn_features = cnn_features.mean(1)  # (batch_size, cnn_feature_dim)
        
        print(numerical_features.shape)
        print(cnn_features.shape)
        
        # 특성 융합
        fused_features = F.relu(self.fusion(torch.cat([numerical_features, cnn_features], dim=1)))
        
        # 최종 포트폴리오 가중치 예측
        portfolio_weights = F.softmax(self.final_fc(fused_features), dim=1)
        
        # 리밸런싱 적용
        portfolio_weights = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in portfolio_weights])
        
        # binary_pred 재구성
        binary_pred = binary_pred.view(batch_size, n_stocks)
        
        return portfolio_weights, binary_pred

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            if nominees.numel() == 0:
                break
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped
