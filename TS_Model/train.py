# train.py
import os
import torch
import numpy as np
import logging
from model.gru import GRU, GRUWithProb
from model.transformer import Transformer, TransformerWithProb
from model.tcn import TCN, TCNWithProb
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity
from tqdm import tqdm
from typing import Dict, Any
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, config: Dict[str, Any], use_prob: bool = False, local_rank: int = -1):
        """
        학습을 위한 Trainer 클래스 초기화
        """
        self.config = config
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
            
        # Loss function 설정
        self.criterion = max_sharpe if self.config.get("LOSS_FUNCTION", "max_sharpe") == "max_sharpe" else equal_risk_parity
        
        # 결과 저장 경로 설정
        self.model_dir = os.path.join(self.config['RESULT_DIR'], self.config["MODEL"])
        os.makedirs(self.model_dir, exist_ok=True)
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
            lr=self.config["LR"],
            momentum=self.config['MOMENTUM']
        )
        

    def _create_model(self) -> torch.nn.Module:
        """모델 생성 및 이전 체크포인트 로드"""
        # 기존 체크포인트 찾기
        model_files = []
        for file in os.listdir(self.model_dir):
            if not file.endswith('.pth'):
                continue
                
            # 파일명 파싱
            parts = file.split('_')
            use_prob = parts[0].lower() == 'true'
            model_type = parts[1]
            
            # 현재 설정과 일치하는지 확인
            if (model_type == self.config["MODEL"] and 
                use_prob == self.use_prob):
                loss = float(file.split('loss_')[1].replace('.pth', ''))
                model_files.append((loss, file))
        
        # 모델 생성
        if self.config["MODEL"] == "TCN":
            model = (TCNWithProb if self.use_prob else TCN)(
                n_feature=self.config["N_STOCK"],
                n_output=self.config["N_STOCK"],
                num_channels=[self.config["TCN"]["hidden_size"]] * self.config["TCN"]["level"],
                kernel_size=self.config["TCN"]["kernel_size"],
                n_dropout=self.config["TCN"]["n_dropout"],
                n_timestep=self.config["TCN"]["n_timestep"],
                lb=self.config["LB"],
                ub=self.config["UB"],
            )
        elif self.config["MODEL"] == "GRU":
            if self.use_prob:
                model = GRUWithProb(
                    n_layers=self.config["N_LAYER"],
                    hidden_dim=self.config["HIDDEN_DIM"],
                    n_stocks=self.config["N_STOCK"],
                    dropout_p=self.config["DROPOUT"],
                    bidirectional=self.config["BIDIRECTIONAL"],
                    lb=self.config['LB'],
                    ub=self.config['UB'],
                )
            else:
                model = GRU(
                    n_layers=self.config["N_LAYER"],
                    hidden_dim=self.config["HIDDEN_DIM"],
                    n_stocks=self.config["N_STOCK"],
                    dropout_p=self.config["DROPOUT"],
                    bidirectional=self.config["BIDIRECTIONAL"],
                    lb=self.config['LB'],
                    ub=self.config['UB'],
                )
        elif self.config["MODEL"] == "TRANSFORMER":
            if self.use_prob:
                model = TransformerWithProb(
                    n_feature=self.config["N_STOCK"],
                    n_timestep=self.config["TRAIN_LEN"],
                    n_layer=self.config["TRANSFORMER"]["n_layer"],
                    n_head=self.config["TRANSFORMER"]["n_head"],
                    n_dropout=self.config["TRANSFORMER"]["n_dropout"],
                    n_output=self.config["N_STOCK"],
                    lb=self.config['LB'],
                    ub=self.config['UB'],
                )
            else:
                model = Transformer(
                    n_feature=self.config["N_STOCK"],
                    n_timestep=self.config["TRAIN_LEN"],
                    n_layer=self.config["TRANSFORMER"]["n_layer"],
                    n_head=self.config["TRANSFORMER"]["n_head"],
                    n_dropout=self.config["TRANSFORMER"]["n_dropout"],
                    n_output=self.config["N_STOCK"],
                    lb=self.config['LB'],
                    ub=self.config['UB'],
                )
        else:
            raise ValueError(f"Unknown model type: {self.config['MODEL']}")
        
        model = model.to(self.device)
        
        # 이전 체크포인트가 있다면 로드
        if model_files:
            best_model = min(model_files, key=lambda x: x[0])[1]
            checkpoint_path = os.path.join(self.model_dir, best_model)
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.best_loss = checkpoint['loss']
            self.early_stop_count = checkpoint.get('early_stop_count', 0)
            
            # optimizer 상태 저장을 위해 checkpoint 임시 저장
            self.temp_checkpoint = checkpoint
            
            logger.info(f"Loaded checkpoint - Loss: {self.best_loss:.4f}, Early stop count: {self.early_stop_count}")
            
        else:
            logger.info("No existing checkpoint found. Starting from scratch.")
            self.best_loss = float('inf')
            self.early_stop_count = 0
            self.temp_checkpoint = None
            
        return model

    def set_data(self):
        """데이터 로드 및 전처리"""
        logging.info("Loading datasets...")
        
        try:
            # 설정된 경로에서 데이터 로드
            with open(self.config['DATASET_PATH'], "rb") as f:
                data_dict = pickle.load(f)
            
            with open(self.config['DATES_PATH'], "rb") as f:
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
            batch_size=self.config["BATCH"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def train(self):
        """모델 학습 수행"""
        train_loader = self.dataloader(self.train_x, self.train_y, self.train_prob)
        val_loader = self.dataloader(self.val_x, self.val_y, self.val_prob)
        
        # optimizer 상태 복원
        if self.temp_checkpoint is not None:
            self.optimizer.load_state_dict(self.temp_checkpoint['optimizer_state_dict'])
            history = self.temp_checkpoint['history']
            start_epoch = self.temp_checkpoint['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_batch_loss': [],
                'learning_rate': []
            }
            start_epoch = 0
        
        epochs = tqdm(range(start_epoch, self.config["EPOCHS"]), desc="Training", ncols=100)
        
        for epoch in epochs:
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            # Training
            self.model.train()
            train_epoch_loss, batch_losses = self._run_epoch(train_loader, is_training=True)
            
            # Loss 동기화
            if self.distributed:
                train_loss_tensor = torch.tensor(train_epoch_loss).to(self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_epoch_loss = train_loss_tensor.item() / dist.get_world_size()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss, _ = self._run_epoch(val_loader, is_training=False)
                
                if self.distributed:
                    val_loss_tensor = torch.tensor(val_epoch_loss).to(self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_epoch_loss = val_loss_tensor.item() / dist.get_world_size()
                    
            # Loss 기록
            history['train_loss'].append(train_epoch_loss)
            history['val_loss'].append(val_epoch_loss)
            history['train_batch_loss'].extend(batch_losses)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 로그 파일에 기록
            logger.info(f"Epoch {epoch+1}/{self.config['EPOCHS']}")
            logger.info(f"Train Loss: {train_epoch_loss:.4f}")
            logger.info(f"Val Loss: {val_epoch_loss:.4f}")
            
            # Progress bar 업데이트
            epochs.set_postfix({
                'train_loss': f'{train_epoch_loss:.4f}',
                'val_loss': f'{val_epoch_loss:.4f}'
            })
            
            # Early stopping 업데이트
            if not self.distributed or self.local_rank == 0:
                if val_epoch_loss < self.best_loss:
                    self.best_loss = val_epoch_loss
                    self.early_stop_count = 0
                    self._save_model(epoch, history)
                else:
                    self.early_stop_count += 1
                    
                if self.early_stop_count >= self.config["EARLY_STOP"]:
                    epochs.write(f"Early stopping at epoch {epoch}")
                    break

        # 학습 완료 후 추론 수행
        if self.config.get("INFER_AFTER_TRAIN", False):
            self.infer_and_save_weights()

        return self.model, history

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

    def _run_epoch(self, dataloader, is_training=True):
        """한 epoch 실행"""
        total_loss = 0
        batch_losses = []  # 배치별 loss 기록
        mode = "Train" if is_training else "Valid"
        
        batches = tqdm(
            dataloader, 
            desc=f"{mode} batches",
            leave=False,
            ncols=100,
            mininterval=1.0,
            miniters=max(1, len(dataloader) // 10)
        )
        
        for batch_idx, (x, y, probs) in enumerate(batches):
            x = x.to(self.device)
            y = y.to(self.device)
            probs = probs.to(self.device)

            if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(probs).any():
                logger.error("NaN detected in inputs")
                raise ValueError("NaN detected in inputs")
            
            if is_training:
                self.optimizer.zero_grad()
                out = self.model(x, probs)
                loss = self.criterion(y, out)
                loss.backward()
                
                # gradient norm 기록
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                logger.debug(f"Gradient norm: {grad_norm:.4f}")
                
                self.optimizer.first_step(zero_grad=True)
                self.criterion(y, self.model(x, probs)).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                with torch.no_grad():
                    out = self.model(x, probs)
                    loss = self.criterion(y, out)
            
            current_loss = loss.item()
            total_loss += current_loss
            batch_losses.append(current_loss)
            
            # tqdm 업데이트 빈도 조정
            if len(dataloader) >= 10 and batch_idx % (len(dataloader) // 10) == 0:
                batches.set_postfix({'loss': f'{current_loss:.4f}'})
        
        return total_loss / len(dataloader), batch_losses

    def _save_model(self, epoch, history):
        """모델과 학습 기록 저장"""
        # 임시 파일에 먼저 저장
        temp_save_path = os.path.join(
            self.model_dir, 
            f'temp_{self.use_prob}_{self.config["MODEL"]}_{epoch}_loss_{self.best_loss:.4f}.pth'
        )
        final_save_path = os.path.join(
            self.model_dir, 
            f'{self.use_prob}_{self.config["MODEL"]}_{epoch}_loss_{self.best_loss:.4f}.pth'
        )
        
        # 임시 파일에 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'history': history,
            'early_stop_count': self.early_stop_count,
        }, temp_save_path)
        
        # 원자적 이동 연산으로 파일 이름 변경
        os.replace(temp_save_path, final_save_path)
        
        logger.info(f"Model and training history saved to {final_save_path}")

    def __del__(self):
        """소멸자에서 process group 정리"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()