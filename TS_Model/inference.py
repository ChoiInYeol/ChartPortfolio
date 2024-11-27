# inference.py
import os
import torch
import numpy as np
import pandas as pd
import logging
from model.gru import PortfolioGRU, PortfolioGRUWithProb
from model.transformer import PortfolioTransformer, PortfolioTransformerWithProb
from model.tcn import PortfolioTCN, PortfolioTCNWithProb
from typing import Dict, Any, Optional
import torch.distributed as dist
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, config: Dict[str, Any]):
        """
        추론 클래스 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device("cuda" if config["USE_CUDA"] else "cpu")
        
        # 모델 저장 디렉토리 설정
        self.model_dir = f"/home/indi/codespace/ImagePortOpt/TS_Model/Result/{config['MODEL']['TYPE']}"
        
        # 가장 최신의 체크포인트 찾기
        self.model_path = self._find_best_checkpoint()
            
        # 모델 초기화
        self.model = self._load_model()
        
        # 결과 저장 디렉토리 생성
        self.result_dir = Path(config['PATHS']['RESULTS']) / config['MODEL']['TYPE']
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로드
        self._load_data()

    def _find_best_checkpoint(self) -> str:
        """가장 높은 epoch의 체크포인트를 찾습니다."""
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {self.model_dir}")
            
            # 체크포인트 파일 목록 가져오기
            checkpoints = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
            if not checkpoints:
                raise FileNotFoundError("체크포인트 파일이 없습니다.")
            
            # 현재 설정 로깅
            logger.info("현재 설정:")
            logger.info(f"- 모델 타입: {self.config['MODEL']['TYPE']}")
            logger.info(f"- 확률 사용: {self.config['MODEL']['USEPROB']}")
            logger.info(f"- 손실 함수: {self.config['PORTFOLIO']['OBJECTIVE']}")
            
            # 체크포인트 목록 로깅
            logger.info("발견된 체크포인트 목록:")
            for ckpt in checkpoints:
                logger.info(f"- {ckpt}")
            
            # 현재 설정과 일치하는 체크포인트 필터링
            matching_checkpoints = []
            for ckpt in checkpoints:
                # 파일명에서 직접 prob 설정 확인
                if '_prob_' in ckpt:
                    prob_setting = 'prob'
                elif '_noprob_' in ckpt:
                    prob_setting = 'noprob'
                else:
                    continue
                
                model_type = ckpt.split('_')[1]
                objective = ckpt.split('_')[-1].split('.')[0]
                
                # 설정과 일치하는지 확인
                if (model_type.upper() == self.config['MODEL']['TYPE'].upper() and
                    prob_setting == ('prob' if self.config['MODEL']['USEPROB'] else 'noprob') and
                    objective == self.config['PORTFOLIO']['OBJECTIVE']):
                    matching_checkpoints.append(ckpt)
                    logger.info(f"매칭된 체크포인트: {ckpt}")
                
            if not matching_checkpoints:
                logger.error("현재 설정과 일치하는 체크포인트가 없습니다.")
                logger.error(f"검색 조건: {self.config['MODEL']['TYPE']}, "
                          f"{'prob' if self.config['MODEL']['USEPROB'] else 'noprob'}, "
                          f"{self.config['PORTFOLIO']['OBJECTIVE']}")
                raise FileNotFoundError("현재 설정과 일치하는 체크포인트를 찾을 수 없습니다.")
            
            # epoch 기준으로 정렬하여 가장 높은 것 선택
            best_checkpoint = max(matching_checkpoints, key=lambda x: int(x.split('_')[0]))
            checkpoint_path = str(Path(self.model_dir) / best_checkpoint)
            
            logger.info(f"선택된 체크포인트: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error finding best checkpoint: {str(e)}")
            raise

    def _load_data(self):
        """테스트 데이터를 로드합니다."""
        try:
            # DEEP 경로 사용
            data_dir = str(self.config['PATHS']['DATA']['DEEP'])
            
            # 데이터셋 경로 설정
            dataset_path = Path(data_dir) / "dataset.pkl"
            dates_path = Path(data_dir) / "dates.pkl"
            
            logger.info(f"Data directory: {data_dir}")
            logger.info(f"Loading dataset from {dataset_path}")
            logger.info(f"Loading dates from {dates_path}")
            
            # 데이터셋 로드
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            with open(dates_path, 'rb') as f:
                dates = pickle.load(f)
            
            # 테스트 데이터 추출
            self.test_x, self.test_y, self.test_prob = dataset['test']
            self.test_dates = dates['test']
            
            logger.info(f"Test data loaded - x shape: {self.test_x.shape}")
            logger.info(f"Test dates length: {len(self.test_dates)}")
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            logger.error(f"Config PATHS: {self.config['PATHS']}")
            raise

    def _load_model(self) -> torch.nn.Module:
        """모델 로드 및 초기화"""
        model_type = self.config['MODEL']['TYPE']
        use_prob = self.config['MODEL']['USEPROB']
        
        if model_type == "GRU":
            model_class = PortfolioGRUWithProb if use_prob else PortfolioGRU
            model = model_class(
                n_layers=self.config['MODEL']['N_LAYER'],
                hidden_dim=self.config['MODEL']['HIDDEN_DIM'],
                n_stocks=self.config['DATA']['N_STOCKS'],
                dropout_p=self.config['MODEL']['DROPOUT'],
                bidirectional=self.config['MODEL']['BIDIRECTIONAL'],
                lb=self.config['PORTFOLIO']['CONSTRAINTS']['MIN_POSITION'],
                ub=self.config['PORTFOLIO']['CONSTRAINTS']['MAX_POSITION'],
                n_select=self.config['PORTFOLIO']['CONSTRAINTS'].get('CARDINALITY')
            )
        elif model_type == "TCN":
            # 체크포인트 파일명에서 prob 설정 확인
            is_prob_checkpoint = '_prob_' in self.model_path
            
            # 체크포인트와 현재 설정이 일치하는 모델 클래스 선택
            if is_prob_checkpoint:
                model_class = PortfolioTCNWithProb
            else:
                model_class = PortfolioTCN
            
            model = model_class(
                n_feature=self.config['DATA']['N_STOCKS'],
                n_output=self.config['DATA']['N_STOCKS'],
                num_channels=self.config['MODEL']['TCN']['num_channels'],
                kernel_size=self.config['MODEL']['TCN']['kernel_size'],
                n_dropout=self.config['MODEL']['TCN']['n_dropout'],
                n_timestep=self.config['MODEL']['TCN']['n_timestep'],
                lb=self.config['PORTFOLIO']['CONSTRAINTS']['MIN_POSITION'],
                ub=self.config['PORTFOLIO']['CONSTRAINTS']['MAX_POSITION'],
                n_select=self.config['PORTFOLIO']['CONSTRAINTS'].get('CARDINALITY')
            )
        elif model_type == "TRANSFORMER":
            model_class = PortfolioTransformerWithProb if use_prob else PortfolioTransformer
            model = model_class(
                n_feature=self.config['DATA']['N_STOCKS'],
                n_timestep=self.config['MODEL']['TRANSFORMER']['n_timestep'],
                n_layer=self.config['MODEL']['TRANSFORMER']['n_layer'],
                n_head=self.config['MODEL']['TRANSFORMER']['n_head'],
                n_dropout=self.config['MODEL']['TRANSFORMER']['n_dropout'],
                n_output=self.config['DATA']['N_STOCKS'],
                lb=self.config['PORTFOLIO']['CONSTRAINTS']['MIN_POSITION'],
                ub=self.config['PORTFOLIO']['CONSTRAINTS']['MAX_POSITION'],
                n_select=self.config['PORTFOLIO']['CONSTRAINTS'].get('CARDINALITY')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(self.device)

        # 모델 가중치 로드
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        return model

    def predict(self) -> torch.Tensor:
        """
        포트폴리오 가중치를 예측합니다.
        
        Returns:
            예측된 포트폴리오 가중치 [batch_size, n_stocks]
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(self.test_x)):
                x_returns = torch.tensor(self.test_x[i:i+1], dtype=torch.float32).to(self.device)
                
                if self.config['MODEL']['USEPROB']:
                    x_probs = torch.tensor(self.test_prob[i:i+1], dtype=torch.float32).to(self.device)
                    pred = self.model(x_returns, x_probs)
                else:
                    pred = self.model(x_returns)
                    
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        self.save_weights(predictions)
        
        return predictions

    def save_weights(self, weights_array: np.ndarray):
        """가중치 저장"""
        if weights_array.size == 0:
            logger.error("Empty weights array received")
            return
        
        try:
            # filtered_returns.csv 파일에서 칼럼명(티커) 가져오기
            tickers = pd.read_csv(
                "/home/indi/codespace/ImagePortOpt/TS_Model/data/filtered_returns.csv",
                index_col=0
            ).columns[:self.config['DATA']['N_STOCKS']]
            
            # 날짜 인덱스 사용
            date_index = pd.DatetimeIndex([dates[0] for dates in self.test_dates])
            
            # 가중치 파일명 생성
            weights_filename = (
                f"portfolio_weights_"
                f"{self.config['MODEL']['TYPE']}_"
                f"{'prob' if self.config['MODEL']['USEPROB'] else 'noprob'}_"
                f"{self.config['PORTFOLIO']['OBJECTIVE']}.csv"
            )
            
            weights_path = self.result_dir / weights_filename
            
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
            logger.error(f"Config PATHS: {self.config['PATHS']}")
            raise

def cleanup():
    """분산 처리 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()