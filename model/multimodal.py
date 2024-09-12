import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gru import GRU
from model.tcn import TCN
from model.transformer import Transformer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_input_dim = self._get_fc_input_dim() # 동적으로 FC 레이어 크기 계산
    
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        
        self.fc_binary = nn.Linear(128, 1) # 이진 분류를 위한 추가 레이어

    def _get_fc_input_dim(self):
        # 예시 입력으로 계산
        x = torch.randn(1, 1, 96, 180)
        x = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x))))))
        return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        features = F.relu(self.fc1(x))
        output = self.fc2(features)
        
        # 이진 분류 출력
        binary_output = torch.sigmoid(self.fc_binary(features))
        return output, binary_output, features
    
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)
    
    def forward(self, numerical_features, cnn_features):
        # numerical_features: (batch_size, feature_dim)
        # cnn_features: (batch_size, seq_len, feature_dim)
        
        # 유사도 점수 계산
        scores = torch.bmm(cnn_features, numerical_features.unsqueeze(2)).squeeze(2)
        
        # 소프트맥스로 정규화
        attention_weights = F.softmax(scores, dim=1)
        
        # 가중치 합 계산
        attended_features = torch.bmm(attention_weights.unsqueeze(1), cnn_features).squeeze(1)
        
        return attended_features, attention_weights
    
class Multimodal(nn.Module):
    def __init__(self, model_type, model_params, lb, ub, verbose=False):
        super().__init__()
        self.cnn = CNN()
        self.model_type = model_type
        self.verbose = verbose
        self.lb = lb
        self.ub = ub

        if model_type.lower() == 'gru':
            self.model = GRU(**model_params)
        elif model_type.lower() == 'tcn':
            self.model = TCN(**model_params)
        elif model_type.lower() == 'transformer':
            self.model = Transformer(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # CNN의 출력 차원을 동적으로 결정
        self.cnn_output_dim = self.cnn.fc2.out_features
        self.numerical_output_dim = model_params.get('hidden_dim', 64)  # 기본값 64로 설정

        self.attention = Attention(self.cnn_output_dim)
        self.fusion = nn.Linear(self.cnn_output_dim + self.numerical_output_dim, self.numerical_output_dim)
        self.final_fc = nn.Linear(self.numerical_output_dim, model_params['n_stocks'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x_num, x_img):
        if self.verbose:
            print(f"Multimodal input shapes: x_num={x_num.shape}, x_img={x_img.shape}")
        
        batch_size, seq_len, n_stocks = x_num.shape
        
        # 수치 데이터 처리
        numerical_features = self.model(x_num)
        
        # 이미지 데이터 처리
        x_img_reshaped = x_img.view(batch_size * n_stocks, 1, x_img.shape[2], x_img.shape[3])
        cnn_output, binary_pred, cnn_features = self.cnn(x_img_reshaped)
        cnn_features = cnn_features.view(batch_size, n_stocks, -1)
        
        # 어텐션 메커니즘 적용
        attended_features, attention_weights = self.attention(numerical_features, cnn_features)
        
        # 특성 융합
        fused_features = self.fusion(torch.cat([numerical_features, attended_features], dim=1))
        
        # 최종 포트폴리오 가중치 예측
        portfolio_weights = F.softmax(self.final_fc(fused_features), dim=1)
        
        # 리밸런싱 적용
        portfolio_weights = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in portfolio_weights])
        
        if self.verbose:
            print(f"Portfolio weights shape: {portfolio_weights.shape}")
            print(f"Binary prediction shape: {binary_pred.shape}")
        
        return portfolio_weights, binary_pred, attention_weights

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
