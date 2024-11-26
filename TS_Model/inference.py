import os
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from model.gru import PortfolioGRU
from model.transformer import Transformer
from model.tcn import TCN
from typing import Dict, Any, Optional
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback
import fcntl
import time
from pathlib import Path
import json
from torch import nn

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Inference:
    def __init__(self, config: Dict[str, Any], model_path: str, local_rank: int = -1):
        """
        추론 클래스 초기화
        
        Args:
            config: 설정 딕셔너리
            model_path: 모델 가중치 파일 경로
            local_rank: 분산 학습을 위한 로컬 랭크
        """
        self.config = config
        self.model_path = model_path
        self.local_rank = local_rank
        self.distributed = local_rank != -1

        # 디바이스 설정
        if self.distributed:
            if not dist.is_initialized():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend='nccl')
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device("cuda" if config["USE_CUDA"] else "cpu")

        # 모델 초기화
        self.model = self._load_model()
        
        # 결과 저장 디렉토리 생성
        self.result_dir = os.path.join(config['RESULT_DIR'], config["MODEL"])
        os.makedirs(self.result_dir, exist_ok=True)

    def _load_model(self) -> torch.nn.Module:
        """모델 로드 및 초기화"""
        model_type = self.config["MODEL"]
        model_class = None
        
        if model_type == "GRU":
            model_class = PortfolioGRUWithProb if self.use_prob else PortfolioGRU
            model = model_class(
                n_layers=self.config["N_LAYER"],
                hidden_dim=self.config["HIDDEN_DIM"],
                n_stocks=self.config["N_STOCK"],
                dropout_p=self.config["DROPOUT"],
                bidirectional=self.config["BIDIRECTIONAL"],
                constraints={
                    'long_only': self.config.get("LONG_ONLY", True),
                    'max_position': self.config.get("MAX_POSITION"),
                    'cardinality': self.config.get("CARDINALITY"),
                    'leverage': self.config.get("LEVERAGE", 1.0)
                }
            )
        elif model_type == "TCN":
            model_class = PortfolioTCNWithProb if self.use_prob else PortfolioTCN
            model = model_class(
                n_feature=self.config["N_FEATURE"],
                n_output=self.config["N_STOCK"],
                num_channels=self.config["NUM_CHANNELS"],
                kernel_size=self.config["KERNEL_SIZE"],
                n_dropout=self.config["DROPOUT"],
                n_timestep=self.config["SEQ_LEN"],
                constraints={
                    'long_only': self.config.get("LONG_ONLY", True),
                    'max_position': self.config.get("MAX_POSITION"),
                    'cardinality': self.config.get("CARDINALITY"),
                    'leverage': self.config.get("LEVERAGE", 1.0)
                }
            )
        elif model_type == "TRANSFORMER":
            model_class = PortfolioTransformerWithProb if self.use_prob else PortfolioTransformer
            model = model_class(
                n_feature=self.config["N_FEATURE"],
                n_timestep=self.config["SEQ_LEN"],
                n_layer=self.config["N_LAYER"],
                n_head=self.config["N_HEAD"],
                n_dropout=self.config["DROPOUT"],
                n_output=self.config["N_STOCK"],
                constraints={
                    'long_only': self.config.get("LONG_ONLY", True),
                    'max_position': self.config.get("MAX_POSITION"),
                    'cardinality': self.config.get("CARDINALITY"),
                    'leverage': self.config.get("LEVERAGE", 1.0)
                }
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(self.device)

        # 모델 가중치 로드
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        return model

    def infer(self) -> np.ndarray:
        """추론 수행"""
        try:
            self._load_test_data()
            logger.info(f"Test data loaded: {len(self.test_x)} samples")
            
            weights_list = []
            with torch.no_grad():
                for i in range(len(self.test_x)):
                    x = torch.from_numpy(self.test_x[i]).float().unsqueeze(0).to(self.device)
                    if self.use_prob:
                        probs = torch.from_numpy(self.test_prob[i]).float().unsqueeze(0).to(self.device)
                        weights = self.model(x, probs)
                    else:
                        weights = self.model(x)
                    weights_list.append(weights.cpu().numpy().squeeze(0))
            
            return np.array(weights_list)
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([])

    def save_weights(self, weights_array: np.ndarray):
        """가중치 저장"""
        if weights_array.size == 0:
            logger.error("Empty weights array received")
            return
        
        try:
            # 주식 이름 로드
            tickers = pd.read_csv(
                os.path.join(self.config['DATA_DIR'], 'return_df.csv'),
                index_col=0
            ).columns[:self.config["N_STOCK"]]
            
            # 날짜 인덱스 사용
            date_index = pd.DatetimeIndex(self.test_dates)
            
            # 차원 확인
            if weights_array.shape != (len(date_index), len(tickers)):
                logger.error("Dimension mismatch!")
                return
            
            # 가중치 파일명 생성
            weights_filename = (
                f"portfolio_weights_"
                f"{self.config['MODEL']}_"
                f"loss{self.config['LOSS_FUNCTION']}.csv"
            )
            
            weights_path = os.path.join(self.result_dir, weights_filename)
            
            # DataFrame 생성 및 저장
            df_weights = pd.DataFrame(
                weights_array,
                columns=tickers,
                index=date_index
            )
            df_weights.to_csv(weights_path)
            
            logger.info(f"Weights saved to {weights_path}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            raise

    def calculate_portfolio_metrics(self, weights_array: np.ndarray):
        """포트폴리오 성과 지표 계산"""
        try:
            # 수익률 데이터 로드
            returns = pd.read_csv(
                os.path.join(self.config['DATA_DIR'], 'return_df.csv'),
                index_col=0,
                parse_dates=True
            ).loc[self.test_dates]
            
            # 포트폴리오 수익률 계산
            portfolio_returns = (weights_array * returns).sum(axis=1)
            
            # 성과 지표 계산
            metrics = {
                'Total Return': (1 + portfolio_returns).prod() - 1,
                'Annual Return': portfolio_returns.mean() * 252,
                'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
                'Sharpe Ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
                'Max Drawdown': (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min(),
                'Turnover': np.abs(weights_array[1:] - weights_array[:-1]).sum(axis=1).mean()
            }
            
            # 결과 저장
            metrics_path = os.path.join(
                self.result_dir,
                f"portfolio_metrics_{self.config['MODEL']}.json"
            )
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Portfolio metrics saved to {metrics_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise

    def __del__(self):
        """소멸자에서 process group 정리"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

def cleanup_inf():
    """분산 처리 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

class ModelInference:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_prob: bool = False
    ):
        """
        모델 추론을 위한 클래스
        
        Args:
            model: 학습된 모델
            device: 추론 디바이스
            use_prob: 확률 기반 모델 사용 여부
        """
        self.model = model
        self.device = device
        self.use_prob = use_prob
        self.model.eval()

    def predict(
        self,
        x_returns: torch.Tensor,
        x_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        포트폴리오 가중치를 예측합니다.
        
        Args:
            x_returns: 수익률 시퀀스 [batch_size, seq_len, n_stocks]
            x_probs: 상승확률 [batch_size, pred_len, n_stocks] (확률 기반 모델인 경우)
            
        Returns:
            예측된 포트폴리오 가중치 [batch_size, n_stocks]
        """
        self.model.eval()
        with torch.no_grad():
            x_returns = x_returns.to(self.device)
            
            if self.use_prob:
                if x_probs is None:
                    raise ValueError("확률 기반 모델에는 x_probs가 필요합니다.")
                x_probs = x_probs.to(self.device)
                pred = self.model(x_returns, x_probs)
            else:
                pred = self.model(x_returns)
                
        return pred