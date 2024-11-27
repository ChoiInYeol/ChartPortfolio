# train.py
import os
import pickle
import numpy as np
import pandas as pd
import logging

from model.gru import PortfolioGRU, PortfolioGRUWithProb
from model.transformer import PortfolioTransformer, PortfolioTransformerWithProb
from model.tcn import PortfolioTCN, PortfolioTCNWithProb

from model.loss import sharpe_ratio_loss, mean_variance_loss, minimum_variance_loss
from model.sam import SAM

from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

import traceback

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, config: Dict[str, Any], use_prob: bool = False, local_rank: int = -1):
        """학습을 위한 Trainer 클래스 초기화"""
        self.config = config
        self.use_prob = config['MODEL']['USE_PROB']  # config에서 직접 가져오기
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
            
        # 손실 함수 설정
        self.criterion = self._get_loss_function()
        
        # 결과 저장 경로 설정
        self.model_dir = Path(self.config['PATHS']['RESULTS']) / self.config['MODEL']['TYPE']
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        
        # 모델 초기화
        self.model = self._create_model()
        if self.distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        
        # 옵티마이저 초기화
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(
            self.model.parameters(),
            base_optimizer,
            lr=self.config['TRAINING']['LEARNING_RATE'],
            momentum=self.config['TRAINING']['MOMENTUM']
        )
        
        # Early Stopping 관련 속성 추가
        self.patience = config['TRAINING']['EARLY_STOPPING']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 학습 이력 저장을 위한 리스트
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def _get_loss_function(self):
        """손실 함수를 반환합니다."""
        loss_type = self.config['PORTFOLIO']['OBJECTIVE']
        if loss_type == "meanvar":
            return lambda returns, weights: mean_variance_loss(
                returns, weights, 
                risk_aversion=self.config['PORTFOLIO']['RISKAVERSION']
            )
        elif loss_type == "minvar":
            return minimum_variance_loss
        elif loss_type == "maxsharpe":
            return lambda returns, weights: sharpe_ratio_loss(
                returns, weights,
                risk_free_rate=self.config['PORTFOLIO']['RISKFREERATE']
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def _create_model(self) -> torch.nn.Module:
        """모델을 생성하고 이전 체크포인트를 로드합니다."""
        model_type = self.config['MODEL']['TYPE']  # 수정: MODEL 딕셔너리에서 TYPE만 가져오기
        use_prob = self.config['MODEL']['USE_PROB']
        
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
            model_class = PortfolioTCNWithProb if use_prob else PortfolioTCN
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
        
        # 이전 체크포인트 로드
        if os.path.exists(self.model_dir):
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        self.best_loss = checkpoint.get('loss', float('inf'))
                        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                    else:
                        logger.warning(f"Checkpoint at {checkpoint_path} does not contain model state dict")
                except Exception as e:
                    logger.error(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
            else:
                logger.info("No checkpoint found, starting from scratch")
        else:
            logger.info(f"Model directory {self.model_dir} does not exist, starting from scratch")
        
        return model

    def train(self):
        """모델 학습을 수행합니다."""
        if not hasattr(self, 'train_x'):
            self.set_data()
            
        train_loader = self.dataloader(self.train_x, self.train_y, self.train_prob, is_train=True)
        val_loader = self.dataloader(self.val_x, self.val_y, self.val_prob, is_train=False)
        
        epochs = tqdm(range(self.config["TRAINING"]["EPOCHS"]), desc="Training")
        
        try:
            for epoch in epochs:
                if self.distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                # Training
                self.model.train()
                train_loss = self._run_epoch(train_loader, is_training=True)
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_loss = self._run_epoch(val_loader, is_training=False)
                
                # 학습 이력 저장
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.epochs.append(epoch)
                
                # 로깅
                epochs.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
                
                # Early Stopping 체크
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, self.model, self.optimizer, None, val_loss)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                    break
                    
            # 학습 종료 후 학습 이력 저장
            self._save_training_history()
            
            # 최고 성능의 체크포인트만 남기고 나머지 삭제
            self._cleanup_checkpoints()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _run_epoch(self, dataloader, is_training=True):
        """한 에폭을 실행합니다."""
        total_loss = 0
        
        for batch in dataloader:
            returns = batch[0].to(self.device)
            probs = batch[2].to(self.device) if self.use_prob else None
            
            if is_training:
                self.optimizer.zero_grad()
                
                # First forward-backward step
                if self.use_prob:
                    weights = self.model(returns, probs)
                else:
                    weights = self.model(returns)
                    
                loss = self.criterion(returns, weights)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                
                # Second forward-backward step
                if self.use_prob:
                    weights = self.model(returns, probs)
                else:
                    weights = self.model(returns)
                    
                self.criterion(returns, weights).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                if self.use_prob:
                    weights = self.model(returns, probs)
                else:
                    weights = self.model(returns)
                loss = self.criterion(returns, weights)
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def _save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler, loss: float):
        """체크포인트를 저장합니다."""
        prob_setting = 'prob' if self.config['MODEL']['USEPROB'] else 'noprob'
        
        checkpoint_name = (
            f"{epoch}_{self.config['MODEL']['TYPE']}_"
            f"{loss:.4f}_{prob_setting}_{self.config['PORTFOLIO']['OBJECTIVE']}.pth"
        )
        
        checkpoint_path = self.model_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장됨: {checkpoint_path}")

    def _find_latest_checkpoint(self):
        """현재 설정과 일치하는 체크포인트 중 loss가 가장 낮은 것을 찾습니다."""
        try:
            checkpoints = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
            if not checkpoints:
                logger.info("체크포인트를 찾을 수 없습니다.")
                return None
            
            # 현재 설정과 일치하는 체크포인트 필터링
            matching_checkpoints = []
            for ckpt in checkpoints:
                try:
                    parts = ckpt.split('_')
                    if len(parts) < 5:  # 최소 필요한 부분이 있는지 확인
                        continue
                        
                    # 형식: {epoch}_{model}_{loss}_{use_prob}_{loss_type}.pth
                    if (parts[1] == self.config['MODEL']['TYPE'] and  # 모델 타입
                        parts[3] == ('prob' if self.use_prob else 'no_prob') and  # prob 사용 여부
                        parts[4].split('.')[0] == self.config['PORTFOLIO']['OBJECTIVE']):  # loss 타입
                        
                        # loss 값이 유효한 float인지 확인
                        float(parts[2])  
                        matching_checkpoints.append(ckpt)
                        
                except (IndexError, ValueError) as e:
                    logger.warning(f"체크포인트 {ckpt} 파싱 중 오류 발생: {str(e)}")
                    continue
            
            if not matching_checkpoints:
                logger.info("현재 설정과 일치하는 체크포인트가 없습니다.")
                return None
            
            # loss 값을 기준으로 정렬하여 가장 낮은 것 선택
            best_checkpoint = min(matching_checkpoints, 
                                key=lambda x: float(x.split('_')[2]))
            
            checkpoint_path = os.path.join(self.model_dir, best_checkpoint)
            if not os.path.exists(checkpoint_path):
                logger.error(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
                return None
                
            logger.info(f"최적의 체크포인트를 찾았습니다: {best_checkpoint}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"체크포인트 검색 중 오류 발생: {str(e)}")
            return None

    def set_data(self):
        """데이터 로드 및 전처리"""
        logging.info("Loading datasets...")
        
        try:
            # 설정된 경로에서 데이터 로드
            dataset_path = Path(self.config['PATHS']['DATA']['DEEP']) / 'dataset.pkl'
            dates_path = Path(self.config['PATHS']['DATA']['DEEP']) / 'dates.pkl'
            
            with open(dataset_path, "rb") as f:
                data_dict = pickle.load(f)
            
            with open(dates_path, "rb") as f:
                self.dates_dict = pickle.load(f)
            
            # 스케일링 팩터 설정 (config에서 가져오거나 기본값 사용)
            scale_factor = self.config.get('DATA', {}).get('SCALING_FACTOR', 20)
            logging.info(f"Using scaling factor: {scale_factor}")
            
            def process_data(data, normalize=True):
                """
                데이터 스케일링 및 정규화
                Args:
                    data: 원본 데이터
                    normalize: 정규화 여부
                """
                # 스케일링
                scaled_data = data.astype("float32") * scale_factor
                
                if normalize:
                    # 시계열별 정규화 (각 시퀀스를 독립적으로 정규화)
                    mean = np.mean(scaled_data, axis=1, keepdims=True)
                    std = np.std(scaled_data, axis=1, keepdims=True)
                    normalized_data = (scaled_data - mean) / (std + 1e-8)
                    return normalized_data
                return scaled_data
            
            # Train 데이터 처리
            train_data = data_dict['train']
            self.train_x = process_data(train_data[0])  # 입력 시퀀스는 정규화
            self.train_y = process_data(train_data[1], normalize=False)  # 타겟은 스케일링만
            self.train_prob = train_data[2].astype("float32")  # 확률은 그대로 사용
            self.train_dates = self.dates_dict['train']
            
            # Validation 데이터 처리
            val_data = data_dict['val']
            self.val_x = process_data(val_data[0])
            self.val_y = process_data(val_data[1], normalize=False)
            self.val_prob = val_data[2].astype("float32")
            self.val_dates = self.dates_dict['val']
            
            # Test 데터 처리
            test_data = data_dict['test']
            self.test_x = process_data(test_data[0])
            self.test_y = process_data(test_data[1], normalize=False)
            self.test_prob = test_data[2].astype("float32")
            self.test_dates = self.dates_dict['test']
            
            # 데이터 통계 확인
            logging.info("Data loading, scaling, and normalization completed")
            logging.info(f"Train data statistics:")
            logging.info(f"X - mean: {self.train_x.mean():.4f}, std: {self.train_x.std():.4f}")
            logging.info(f"Y - mean: {self.train_y.mean():.4f}, std: {self.train_y.std():.4f}")
            logging.info(f"X range: [{self.train_x.min():.4f}, {self.train_x.max():.4f}]")
            logging.info(f"Y range: [{self.train_y.min():.4f}, {self.train_y.max():.4f}]")
            
            logging.info(f"Train data shape - X: {self.train_x.shape}, Y: {self.train_y.shape}, Prob: {self.train_prob.shape}")
            logging.info(f"Val data shape - X: {self.val_x.shape}, Y: {self.val_y.shape}, Prob: {self.val_prob.shape}")
            logging.info(f"Test data shape - X: {self.test_x.shape}, Y: {self.test_y.shape}, Prob: {self.test_prob.shape}")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def dataloader(self, x, y, probs, is_train=True):
        """
        데이터로더 생성
        
        Args:
            x: 입력 데이터
            y: 타겟 데이터
            probs: 확률 데이터
            is_train: 학습용 데이터로더인지 여부
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(probs, dtype=torch.float32)
        )
        
        if self.distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=self.local_rank,
                shuffle=is_train  # 학습 시에만 shuffle
            )
        else:
            sampler = None
            
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["TRAINING"]["BATCH_SIZE"],
            shuffle=(sampler is None and is_train),  # 학습 시에만 shuffle
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=is_train,  # 학습 시에만 마지막 배치 drop
        )

    def infer_and_save_weights(self):
        """학습 완료 후 train/val 데이터에 대해 가중치 추론 및 저장"""
        logger.info("Inferring weights for train and validation data...")

        # 모델 로드
        model_path = os.path.join(self.model_dir, f'best_{self.config["MODEL"]}_epoch_{self.best_loss:.4f}.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Train 데이터에 대한 가중치 추론
        train_weights = self._infer_weights(self.train_x, self.train_prob)
        self.save_weights(train_weights, "train")

        # Validation 데이터에 대한 가중치 추론
        val_weights = self._infer_weights(self.val_x, self.val_prob)
        self.save_weights(val_weights, "val")

    def _infer_weights(self, x, probs):
        """가중치 추론"""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            probs_tensor = torch.tensor(probs, dtype=torch.float32).to(self.device)
            weights = self.model(x_tensor, probs_tensor)
            return weights.cpu().numpy()

    def save_weights(self, weights_array: np.ndarray, data_type: str):
        """가중치 저장"""
        if not self.distributed or self.local_rank == 0:
            # 주식 이름 로드
            try:
                tickers = pd.read_csv("/home/indi/codespace/ImagePortOpt/TS_Model/data/filtered_returns.csv", index_col=0).columns[:self.config['DATA']['N_STOCKS']]
            except Exception as e:
                logger.error(f"Error loading tickers: {e}")
                tickers = [f"Stock_{i}" for i in range(self.config['DATA']['N_STOCKS'])]

            # 날짜 인덱스 가져오기
            if data_type == "train":
                dates = self.train_dates
            elif data_type == "val":
                dates = self.val_dates
            else:  # test
                dates = self.test_dates
                
            # 날짜 리스트를 평탄화 (중첩된 리스트 처리)
            flat_dates = [date for sublist in dates for date in sublist]
            
            weights_path = os.path.join(
                self.config['RESULT_DIR'],
                self.config["MODEL"],
                f"weights_{data_type}.csv"
            )
            
            # CSV 파일에 헤더와 인덱스 추가
            df_weights = pd.DataFrame(weights_array, columns=tickers, index=pd.to_datetime(flat_dates))
            df_weights.to_csv(weights_path)
            logger.info(f"Weights for {data_type} data saved to {weights_path}")

    def _save_training_history(self):
        """학습 이력을 CSV 파일로 저장합니다."""
        if not self.distributed or self.local_rank == 0:  # 분산 학습 시 마스터 노드에서만 저장
            history_df = pd.DataFrame({
                'epoch': self.epochs,
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            })
            
            # 모델 설정에 따른 파일명 생성
            model_type = self.config["MODEL"]["TYPE"]
            if model_type == "GRU":
                model_config = f"{model_type}_L{self.config['MODEL']['N_LAYER']}_H{self.config['MODEL']['HIDDEN_DIM']}"
                if self.config['MODEL']['BIDIRECTIONAL']:
                    model_config += "_bi"
            elif model_type == "TCN":
                model_config = f"{model_type}_L{self.config['MODEL']['TCN']['level']}_H{self.config['MODEL']['TCN']['hidden_size']}"
            elif model_type == "TRANSFORMER":
                model_config = f"{model_type}_L{self.config['MODEL']['TRANSFORMER']['n_layer']}_H{self.config['MODEL']['TRANSFORMER']['n_head']}"
            
            # 목적함수 추가
            objective = self.config['PORTFOLIO']['OBJECTIVE']
            model_config += f"_{objective}"
            
            # 확률 사용 여부 추가 
            if self.config['MODEL']['USE_PROB']:
                model_config += "_prob"
                
            # Loss와 Epoch 추가
            model_config += f"_e{len(self.epochs)}_loss{self.best_val_loss:.4f}"
            
            # 저장 경로 설정
            history_path = os.path.join(
                self.model_dir,
                f'training_history_{model_config}.csv'
            )
            
            history_df.to_csv(history_path, index=False)
            logger.info(f"Training history saved to {history_path}")
            
            # 학습 결과 요약 로깅
            logger.info(f"Training Summary:")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Total epochs: {len(self.epochs)}")
            logger.info(f"Final train loss: {self.train_losses[-1]:.4f}")
            logger.info(f"Final validation loss: {self.val_losses[-1]:.4f}")

    def __del__(self):
        """소멸자에서 process group 정리"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def predict(self, data_type: str = 'train') -> np.ndarray:
        """���정된 데이터셋에 대한 포트폴리오 가중치를 예측합니다."""
        self.model.eval()
        
        # 데이터셋 선택
        if data_type == 'train':
            x_data = self.train_x
            prob_data = self.train_prob if self.use_prob else None
        elif data_type == 'val':
            x_data = self.val_x
            prob_data = self.val_prob if self.use_prob else None
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # 예측을 위한 데이터로더 생성 (shuffle=False)
        loader = self.dataloader(x_data, 
                               np.zeros_like(x_data),  # dummy y values
                               prob_data if self.use_prob else np.zeros_like(x_data),
                               is_train=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                x_returns = batch[0].to(self.device)
                if self.use_prob:
                    x_probs = batch[2].to(self.device)
                    pred = self.model(x_returns, x_probs)
                else:
                    pred = self.model(x_returns)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

    def save_weights(self, weights_array: np.ndarray, data_type: str = 'train'):
        """
        가중치를 저장합니다.
        
        Args:
            weights_array (np.ndarray): 저장할 가중치 배열
            data_type (str): 데이터 타입 ('train' 또는 'val')
        """
        if weights_array.size == 0:
            logger.error("Empty weights array received")
            return
        
        try:
            # 티커 가져오기
            tickers = pd.read_csv(
                "/home/indi/codespace/ImagePortOpt/TS_Model/data/filtered_returns.csv",
                index_col=0
            ).columns[:self.config['DATA']['N_STOCKS']]
            
            # 날짜 인덱스 사용
            dates = self.train_dates if data_type == 'train' else self.val_dates
            date_index = pd.DatetimeIndex([date[0] for date in dates])
            
            # 가중치 파일명 생성
            weights_filename = (
                f"portfolio_weights_"
                f"{self.config['MODEL']['TYPE']}_"
                f"{'prob' if self.use_prob else 'no_prob'}_"
                f"{self.config['PORTFOLIO']['OBJECTIVE']}_{data_type}.csv"
            )
            
            weights_path = self.model_dir / weights_filename
            
            # DataFrame 생성 및 저장
            df_weights = pd.DataFrame(
                weights_array,
                columns=tickers,
                index=date_index
            )
            df_weights.to_csv(weights_path)
            
            logger.info(f"{data_type.capitalize()} weights saved to {weights_path}")
            
        except Exception as e:
            logger.error(f"Error saving {data_type} weights: {str(e)}")
            raise

    def _cleanup_checkpoints(self):
        """
        학습 종료 후 최고 성능의 체크포인트만 남기고 나머지 삭제
        """
        if not self.distributed or self.local_rank == 0:  # 마스터 노드에서만 실행
            try:
                # 체크포인트 파일 목록 가져오기
                checkpoints = []
                for f in os.listdir(self.model_dir):
                    if f.endswith('.pth'):
                        try:
                            # 체크포인트 파일명 파싱 시도
                            parts = f.split('_')
                            if len(parts) >= 5:  # 최소 필요한 부분이 있는지 확인
                                checkpoints.append(f)
                        except:
                            continue
                            
                if not checkpoints:
                    logger.info("No checkpoints found to cleanup")
                    return
                
                # 현재 설정과 일치하는 체크포인트 필터링
                matching_checkpoints = []
                for ckpt in checkpoints:
                    try:
                        parts = ckpt.split('_')
                        model_type = parts[1]
                        prob_setting = parts[3]
                        loss_type = parts[4].split('.')[0]
                        
                        if (model_type == self.config['MODEL']['TYPE'] and
                            prob_setting == ('prob' if self.use_prob else 'no_prob') and
                            loss_type == self.config['PORTFOLIO']['OBJECTIVE']):
                            matching_checkpoints.append(ckpt)
                    except:
                        continue
                
                if not matching_checkpoints:
                    logger.info("No matching checkpoints found")
                    return
                
                # loss 값을 기준으로 정렬하여 가장 좋은 것 선택
                try:
                    best_checkpoint = min(matching_checkpoints,
                                        key=lambda x: float(x.split('_')[2]))
                except ValueError as e:
                    logger.error(f"Error parsing loss values: {str(e)}")
                    return
                
                # 나머지 체크포인트 삭제
                deleted_count = 0
                for ckpt in matching_checkpoints:
                    if ckpt != best_checkpoint:
                        try:
                            checkpoint_path = os.path.join(self.model_dir, ckpt)
                            os.remove(checkpoint_path)
                            deleted_count += 1
                            logger.info(f"Removed checkpoint: {checkpoint_path}")
                        except OSError as e:
                            logger.error(f"Error removing {ckpt}: {str(e)}")
                            continue
                
                logger.info(f"Cleanup complete - Kept: {best_checkpoint}, Removed: {deleted_count} files")
                
            except Exception as e:
                logger.error(f"Error during checkpoint cleanup: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")