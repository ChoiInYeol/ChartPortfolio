import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gru import GRU
from model.tcn import TCN
from model.transformer import Transformer

class CNN(nn.Module):
    def __init__(self, output_dim, verbose=False):
        super(CNN, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 입력 이미지 크기에 맞게 계산된 값
        self.fc1 = nn.Linear(64 * 16 * 29, 128)
        self.fc2 = nn.Linear(128, output_dim)  # 여기서 output_dim을 50 이하로 설정

    def forward(self, x):
        if self.verbose:
            print(f"4. CNN input shape: {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))
        if self.verbose:
            print(f"5. After first conv and pool: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        if self.verbose:
            print(f"6. After second conv and pool: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        if self.verbose:
            print(f"7. After flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.verbose:
            print(f"8. CNN output shape: {x.shape}")
        return x

class Multimodal(nn.Module):
    def __init__(self, model_type, model_params, cnn_output_dim, verbose=False):
        super().__init__()
        self.cnn = CNN(cnn_output_dim, verbose)
        self.model_type = model_type
        self.verbose = verbose

        if model_type.lower() == 'gru':
            self.model = GRU(**model_params)
        elif model_type.lower() == 'tcn':
            self.model = TCN(**model_params)
        elif model_type.lower() == 'transformer':
            self.model = Transformer(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x_num, x_img):
        if self.verbose:
            print(f"1. Multimodal input shapes: x_num={x_num.shape}, x_img={x_img.shape}")
        batch_size, seq_len, n_stocks = x_num.shape

        # CNN 입력을 batch_size * n_stocks로 reshape
        x_img_reshaped = x_img.view(batch_size * n_stocks, 1, x_img.shape[2], x_img.shape[3])
        cnn_output = self.cnn(x_img_reshaped)

        # CNN 출력의 크기를 다시 batch_size, n_stocks로 맞춤
        cnn_output = cnn_output.view(batch_size, n_stocks, -1)  # CNN output shape (batch_size, n_stocks, cnn_output_dim)
        if self.verbose:
            print(f"2. CNN output shape: {cnn_output.shape}")

        # CNN 출력의 시계열 길이(seq_len)에 맞게 확장
        cnn_output = cnn_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # Shape: (batch_size, seq_len, n_stocks, cnn_output_dim)
        cnn_output = cnn_output.reshape(batch_size, seq_len, -1)  # Flatten to shape: (batch_size, seq_len, n_stocks * cnn_output_dim)
        if self.verbose:
            print(f"Adjusted CNN output shape: {cnn_output.shape}")

        # x_num과 CNN 출력 결합
        x_combined = torch.cat([x_num, cnn_output], dim=2)  # Concatenate along feature dimension
        if self.verbose:
            print(f"3. Combined input shape: {x_combined.shape}")

        return self.model(x_combined)