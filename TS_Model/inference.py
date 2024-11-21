import os
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from typing import Dict, Any
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback
import fcntl
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Inference:
    def __init__(self, config: Dict[str, Any], model_path: str, use_prob: bool = False, local_rank: int = -1):
        """
        추론 클래스 초기화
        
        Args:
            config: 설정 딕셔너리
            model_path: 모델 가중치 파일 경로
            use_prob: 상승확률 데이터 사용 여부
            local_rank: 분산 학습을 위한 로컬 랭크
        """
        self.config = config
        self.model_path = model_path
        self.use_prob = use_prob
        self.local_rank = local_rank
        self.distributed = local_rank != -1

        if self.distributed:
            if not dist.is_initialized():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend='nccl')
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device("cuda" if config["USE_CUDA"] else "cpu")

        self.model_name = config["MODEL"]
        self.len_train = config['TRAIN_LEN']
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_STOCK']
        self.lb = config['LB']
        self.ub = config['UB']

        # Initialize logging to file
        if not self.distributed or self.local_rank == 0:
            model_identifier = f"{self.model_name}_{self.config['LOSS_FUNCTION']}_{self.len_train}_{self.len_pred}"
            log_filename = os.path.join(config['RESULT_DIR'], config["MODEL"], f"inference_{model_identifier}.log")
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.info(f"Configuration: {self.config}")
            logger.info(f"Model Path: {self.model_path}")

        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """모델 로드 및 초기화"""
        if self.model_name.lower() == "gru":
            if self.use_prob:
                from model.gru import GRUWithProb
                model = GRUWithProb(
                    n_layers=self.config["N_LAYER"],
                    hidden_dim=self.config["HIDDEN_DIM"],
                    n_stocks=self.n_stock,
                    dropout_p=self.config["DROPOUT"],
                    bidirectional=self.config["BIDIRECTIONAL"],
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
            else:
                from model.gru import GRU
                model = GRU(
                    n_layers=self.config["N_LAYER"],
                    hidden_dim=self.config["HIDDEN_DIM"],
                    n_stocks=self.n_stock,
                    dropout_p=self.config["DROPOUT"],
                    bidirectional=self.config["BIDIRECTIONAL"],
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
        elif self.model_name.lower() == "transformer":
            if self.use_prob:
                from model.transformer import TransformerWithProb
                model = TransformerWithProb(
                    n_feature=self.n_stock,
                    n_timestep=self.len_train,
                    n_layer=self.config["TRANSFORMER"]["n_layer"],
                    n_head=self.config["TRANSFORMER"]["n_head"],
                    n_dropout=self.config["TRANSFORMER"]["n_dropout"],
                    n_output=self.n_stock,
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
            else:
                from model.transformer import Transformer
                model = Transformer(
                    n_feature=self.n_stock,
                    n_timestep=self.len_train,
                    n_layer=self.config["TRANSFORMER"]["n_layer"],
                    n_head=self.config["TRANSFORMER"]["n_head"],
                    n_dropout=self.config["TRANSFORMER"]["n_dropout"],
                    n_output=self.n_stock,
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
        elif self.model_name.lower() == "tcn":
            if self.use_prob:
                from model.tcn import TCNWithProb
                model = TCNWithProb(
                    n_feature=self.n_stock,
                    n_output=self.n_stock,
                    num_channels=self.config["TCN"]["channels"],
                    kernel_size=self.config["TCN"]["kernel_size"],
                    n_dropout=self.config["TCN"]["n_dropout"],
                    n_timestep=self.config["TCN"]["n_timestep"],
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
            else:
                from model.tcn import TCN
                model = TCN(
                    n_feature=self.n_stock,
                    n_output=self.n_stock,
                    num_channels=self.config["TCN"]["channels"],
                    kernel_size=self.config["TCN"]["kernel_size"],
                    n_dropout=self.config["TCN"]["n_dropout"],
                    n_timestep=self.config["TCN"]["n_timestep"],
                    lb=self.lb,
                    ub=self.ub,
                    n_select=self.config['N_SELECT']
                )
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        model = model.to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

        # 파일 목록 가져오기 시 파일 잠금 사용
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with open(self.model_path + "/.lock", "w") as lockfile:
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # 모델 파일 목록 가져오기
                    pth_files = [f for f in os.listdir(self.model_path) if f.endswith('.pth')]
                    if not pth_files:
                        fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
                        raise FileNotFoundError("No .pth files found")
                    
                    # 가장 최근 모델 선택
                    latest_model = sorted(pth_files, key=lambda x: int(x.split('_')[3]))[-1]
                    model_path = os.path.join(self.model_path, latest_model)
                    
                    # 모델 로드
                    logger.info(f"Loading model: {latest_model}")
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                    
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
                    break
                    
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model

    def _load_test_data(self):
        """테스트 데이터 로드"""
        logger.info("Loading test data...")
        
        try:
            # dataset.pkl 파일에서 데이터 로드
            with open(self.config['DATASET_PATH'], "rb") as f:
                data_dict = pickle.load(f)
            
            # date.pkl 파일에서 날짜 정보 로드
            with open(self.config['DATES_PATH'], "rb") as f:
                dates_dict = pickle.load(f)
            
            # 데이터 스케일링
            scale = self.config["PRED_LEN"]
            
            # Test 데이터 추출 및 스케일링
            test_data = data_dict['test']
            self.test_x = test_data[0].astype("float32") * scale
            self.test_y = test_data[1].astype("float32") * scale
            self.test_prob = test_data[2].astype("float32")
            
            # 각 시퀀스의 마지막 날짜만 추출
            try:
                # 각 리스트의 마지막 Timestamp만 선택
                self.test_dates = [dates[-1] for dates in dates_dict['test']]
                logger.info(f"First few dates after processing: {self.test_dates[:5]}")
                logger.info(f"Total dates: {len(self.test_dates)}")
            except Exception as e:
                logger.error(f"Error processing dates: {str(e)}")
                logger.error(f"Date data type: {type(dates_dict['test'])}")
                logger.error(f"First few raw dates: {dates_dict['test'][:5]}")
                raise
            
            # 데이터 검증
            logger.info("Validating test data...")
            logger.info(f"Test data shape - X: {self.test_x.shape}, Y: {self.test_y.shape}, Prob: {self.test_prob.shape}")
            logger.info(f"Test dates length: {len(self.test_dates)}")
            
            # NaN 체크
            if (np.isnan(self.test_x).any() or 
                np.isnan(self.test_y).any() or 
                np.isnan(self.test_prob).any()):
                raise ValueError("NaN values detected in test data")
            
            logger.info("Test data loading completed")
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _load_best_model(self):
        """
        설정과 일치하는 가장 좋은 성능의 모델 로드
        
        Returns:
            str: 선택된 모델의 전체 경로
        """
        # 모델 파일 리스트 가져오기
        pth_files = [f for f in os.listdir(self.model_path) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError("No .pth files found in the specified model path.")
        
        # 현재 설정과 일치하는 모델 필터링
        valid_models = []
        for pth_file in pth_files:
            try:
                # 파일명 파싱
                # format: {use_prob}_{model_type}_{n_select}_{epoch}_loss_{loss}.pth
                parts = pth_file.split('_')
                use_prob = parts[0].lower() == 'true'
                model_type = parts[1]
                n_select = int(parts[2])
                loss = float(parts[-1].replace('.pth', ''))
                
                # 현재 설정과 일치하는지 확인
                if (model_type == self.config["MODEL"] and 
                    n_select == self.config["N_SELECT"] and 
                    use_prob == self.use_prob):
                    valid_models.append((loss, pth_file))
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping invalid model file {pth_file}: {str(e)}")
                continue
        
        if not valid_models:
            raise FileNotFoundError(
                f"No matching model found for current configuration:\n"
                f"Model: {self.config['MODEL']}\n"
                f"N_SELECT: {self.config['N_SELECT']}\n"
                f"use_prob: {self.use_prob}"
            )
        
        # loss가 가장 작은 모델 선택
        best_model = min(valid_models, key=lambda x: x[0])[1]
        model_path = os.path.join(self.model_path, best_model)
        
        logger.info(f"Selected best model: {best_model}")
        logger.info(f"Model loss: {min(valid_models)[0]:.4f}")
        
        return model_path

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
                        # prob 차원을 [batch_size, pred_len, n_stocks]로 맞춤
                        prob = torch.from_numpy(self.test_prob[i]).float().to(self.device)
                        prob = prob.unsqueeze(0)  # [1, n_stocks] -> [1, 1, n_stocks]
                        
                        # 입력 데이터 shape 로깅
                        logger.debug(f"Input shapes - x: {x.shape}, prob: {prob.shape}")
                        
                        # 차원 검증
                        if x.dim() != 3 or prob.dim() != 3:
                            raise ValueError(
                                f"Invalid input dimensions. "
                                f"Expected 3D tensors, got x: {x.dim()}D, prob: {prob.dim()}D"
                            )
                        
                        outputs = self.model(x, prob)
                    else:
                        outputs = self.model(x)
                    
                    weights = outputs.cpu().numpy().squeeze(0)
                    weights_list.append(weights)
            
            return np.array(weights_list)
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Input shapes - x: {x.shape if 'x' in locals() else 'N/A'}, "
                        f"prob: {prob.shape if 'prob' in locals() else 'N/A'}")
            return np.array([])

    def save_weights(self, weights_array: np.ndarray):
        """가중치 저장"""
        if weights_array.size == 0:
            logger.error("Empty weights array received")
            return
        
        try:
            # 프로젝트 루트 경로 설정
            current_path = Path(__file__).resolve()
            project_root = current_path.parent.parent
            data_dir = project_root / 'Data'
            
            # 주식 이름 로드
            tickers = pd.read_csv(data_dir / 'return_df.csv', index_col=0).columns[:self.n_stock]
            logger.info(f"Loaded {len(tickers)} tickers from {data_dir / 'return_df.csv'}")
            
            # 날짜 인덱스 사용
            if not isinstance(self.test_dates, pd.DatetimeIndex):
                logger.warning(f"Converting dates from type {type(self.test_dates)} to DatetimeIndex")
                try:
                    date_index = pd.DatetimeIndex(self.test_dates)
                except Exception as e:
                    logger.error(f"Error converting dates: {str(e)}")
                    logger.error(f"First few dates: {self.test_dates[:5]}")
                    raise
            else:
                date_index = self.test_dates
            
            # 차원 확인
            logger.info(f"Weights array shape: {weights_array.shape}")
            logger.info(f"Expected shape: ({len(date_index)}, {len(tickers)})")
            
            if weights_array.shape != (len(date_index), len(tickers)):
                logger.error("Dimension mismatch!")
                logger.error(f"Got shape {weights_array.shape}, expected {(len(date_index), len(tickers))}")
                return
            
            # 사용된 모델 파일명에서 loss 값 추출
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pth')]
            used_model = None
            for file in model_files:
                parts = file.split('_')
                if (parts[0].lower() == str(self.use_prob).lower() and 
                    parts[1] == self.model_name and 
                    int(parts[2]) == self.config["N_SELECT"]):
                    used_model = file
                    break
            
            if used_model:
                model_loss = float(used_model.split('_loss_')[-1].replace('.pth', ''))
            else:
                model_loss = 0.0
            
            # 가중치 파일명 생성
            weights_filename = (
                f"weights_"
                f"{self.use_prob}_{self.model_name}_"
                f"n{self.config['N_SELECT']}_"
                f"t{self.len_train}_p{self.len_pred}_"
                f"loss{model_loss:.4f}.csv"
            )
            
            weights_path = os.path.join(
                self.config['RESULT_DIR'],
                self.config["MODEL"],
                weights_filename
            )
            
            # DataFrame 생성 전 데이터 검증
            if np.isnan(weights_array).any():
                logger.warning("NaN values found in weights array")
            if np.isinf(weights_array).any():
                logger.warning("Inf values found in weights array")
            
            # CSV 파일에 헤더와 인덱스 추가
            df_weights = pd.DataFrame(weights_array, columns=tickers, index=date_index)
            df_weights.to_csv(weights_path)
            
            logger.info(f"Weights saved to {weights_path}")
            logger.info(f"Model parameters:")
            logger.info(f"- Use prob: {self.use_prob}")
            logger.info(f"- Model: {self.model_name}")
            logger.info(f"- N_select: {self.config['N_SELECT']}")
            logger.info(f"- Train length: {self.len_train}")
            logger.info(f"- Pred length: {self.len_pred}")
            logger.info(f"- Model loss: {model_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            logger.error(f"Weights array info: shape={weights_array.shape}, dtype={weights_array.dtype}")
            raise
    
    def set_data(self):
        """데이터 로드"""
        try:
            # 설정된 경로에서 데이터 로드
            with open(self.config['DATASET_PATH'], "rb") as f:
                data_dict = pickle.load(f)
            
            with open(self.config['DATES_PATH'], "rb") as f:
                dates_dict = pickle.load(f)
            
            # Test 데이터만 사용
            test_data = data_dict['test']
            self.test_x = test_data[0].astype("float32")
            self.test_y = test_data[1].astype("float32")
            self.test_prob = test_data[2].astype("float32")
            self.test_dates = dates_dict['test']
            
            logger.info("Data loading completed")
            logger.info(f"Test data shape - X: {self.test_x.shape}, Y: {self.test_y.shape}, Prob: {self.test_prob.shape}")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def __del__(self):
        """소멸자에서 process group 정리"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

def cleanup_inf():
    """분산 처리 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()