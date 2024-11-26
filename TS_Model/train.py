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
        if loss_type == "mean_variance":
            return lambda returns, weights: mean_variance_loss(
                returns, weights, 
                risk_aversion=self.config['PORTFOLIO']['RISK_AVERSION']
            )
        elif loss_type == "minimum_variance":
            return minimum_variance_loss
        elif loss_type == "sharpe_ratio":
            return lambda returns, weights: sharpe_ratio_loss(
                returns, weights,
                risk_free_rate=self.config['PORTFOLIO']['RISK_FREE_RATE']
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
                constraints=self.config['PORTFOLIO']['CONSTRAINTS']
            )
        elif model_type == "TCN":
            model_class = PortfolioTCNWithProb if use_prob else PortfolioTCN
            model = model_class(
                n_timestep=self.config['MODEL']['TCN']['n_timestep'],
                n_output=self.config['MODEL']['TCN']['n_output'],
                kernel_size=self.config['MODEL']['TCN']['kernel_size'],
                n_dropout=self.config['MODEL']['TCN']['n_dropout'],
                hidden_size=self.config['MODEL']['TCN']['hidden_size'],
                level=self.config['MODEL']['TCN']['level'],
                channels=self.config['MODEL']['TCN']['channels'],
                constraints=self.config['PORTFOLIO']['CONSTRAINTS']
            )
        elif model_type == "TRANSFORMER":
            model_class = PortfolioTransformerWithProb if use_prob else PortfolioTransformer
            model = model_class(
                n_timestep=self.config['MODEL']['TRANSFORMER']['n_timestep'],
                n_output=self.config['MODEL']['TRANSFORMER']['n_output'],
                n_layer=self.config['MODEL']['TRANSFORMER']['n_layer'],
                n_head=self.config['MODEL']['TRANSFORMER']['n_head'],
                n_dropout=self.config['MODEL']['TRANSFORMER']['n_dropout'],
                constraints=self.config['PORTFOLIO']['CONSTRAINTS']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(self.device)
        
        # 이전 체크포인트 로드
        if os.path.exists(self.model_dir):
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.best_loss = checkpoint['loss']
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return model

    def train(self):
        """모델 학습을 수행합니다."""
        if not hasattr(self, 'train_x'):
            self.set_data()
            
        train_loader = self.dataloader(self.train_x, self.train_y, self.train_prob)
        val_loader = self.dataloader(self.val_x, self.val_y, self.val_prob)
        
        epochs = tqdm(range(self.config["TRAINING"]["EPOCHS"]), desc="Training")
        
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
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
        # 학습 종료 후 학습 이력 저장
        self._save_training_history()

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

    def _save_checkpoint(self, epoch: int, loss: float):
        """체크포인트를 저장합니다."""
        if not self.distributed or self.local_rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epochs': self.epochs
            }
            
            # 새로운 파일명 형식 적용
            filename = f"{epoch}_{self.config['MODEL']['TYPE']}_{loss:.4f}_"\
                    f"{'prob' if self.use_prob else 'no_prob'}_"\
                    f"{self.config['PORTFOLIO']['OBJECTIVE']}.pth"
            
            checkpoint_path = os.path.join(self.model_dir, filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _find_latest_checkpoint(self):
        """현재 설정과 일치하는 체크포인트 중 loss가 가장 낮은 것을 찾습니다."""
        checkpoints = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None
        
        # 현재 설정과 일치하는 체크포인트 필터링
        matching_checkpoints = []
        for ckpt in checkpoints:
            parts = ckpt.split('_')
            # 형식: {epoch}_{model}_{loss}_{use_prob}_{loss_type}.pth
            if (parts[1] == self.config['MODEL']['TYPE'] and  # 모델 타입
                parts[3] == ('prob' if self.use_prob else 'no_prob') and  # prob 사용 여부
                parts[4].split('.')[0] == self.config['PORTFOLIO']['OBJECTIVE']):  # loss 타입
                matching_checkpoints.append(ckpt)
        
        if not matching_checkpoints:
            return None
        
        # loss 값을 기준으로 정렬하여 가장 낮은 것 선택
        best_checkpoint = min(matching_checkpoints, 
                            key=lambda x: float(x.split('_')[2]))
        
        logger.info(f"Found best matching checkpoint: {best_checkpoint}")
        return os.path.join(self.model_dir, best_checkpoint)

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
            # Train 데이터
            train_data = data_dict['train']
            self.train_x = train_data[0].astype("float32")
            self.train_y = train_data[1].astype("float32")
            self.train_prob = train_data[2].astype("float32")
            self.train_dates = self.dates_dict['train']
            
            # Validation 데이터
            val_data = data_dict['val']
            self.val_x = val_data[0].astype("float32")
            self.val_y = val_data[1].astype("float32")
            self.val_prob = val_data[2].astype("float32")
            self.val_dates = self.dates_dict['val']
            
            # Test 데이터
            test_data = data_dict['test']
            self.test_x = test_data[0].astype("float32")
            self.test_y = test_data[1].astype("float32")
            self.test_prob = test_data[2].astype("float32")
            self.test_dates = self.dates_dict['test']
            
            logging.info("Data loading completed")
            logging.info(f"Train data shape - X: {self.train_x.shape}, Y: {self.train_y.shape}, Prob: {self.train_prob.shape}")
            logging.info(f"Val data shape - X: {self.val_x.shape}, Y: {self.val_y.shape}, Prob: {self.val_prob.shape}")
            logging.info(f"Test data shape - X: {self.test_x.shape}, Y: {self.test_y.shape}, Prob: {self.test_prob.shape}")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def dataloader(self, x, y, probs):
        """데이터로더 생성"""
        # 데이터 검증
        print("\n=== Dataloader Input Validation ===")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"probs shape: {probs.shape}")
        print(f"x contains NaN: {np.isnan(x).any()}")
        print(f"y contains NaN: {np.isnan(y).any()}")
        print(f"probs contains NaN: {np.isnan(probs).any()}")
        print("=================================\n")
        
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
                shuffle=True
            )
        else:
            sampler = None
            
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["TRAINING"]["BATCH_SIZE"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
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
                tickers = pd.read_csv("data/return_df.csv", index_col=0).columns[:self.config["N_STOCK"]]
            except Exception as e:
                logger.error(f"Error loading tickers: {e}")
                tickers = [f"Stock_{i}" for i in range(self.config["N_STOCK"])]

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
            
            # 저장 경로 설정
            history_path = os.path.join(
                self.model_dir,
                f'training_history_{self.config["MODEL"]["TYPE"]}.csv'
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