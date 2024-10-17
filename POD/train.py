# train.py
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.multimodal import Multimodal
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity, mean_variance, combined_loss
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Any
import h5py

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class PortfolioDataset(Dataset):
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = f['x'].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_file, 'r') as f:
            x = torch.from_numpy(f['x'][idx]).float()
            y = torch.from_numpy(f['y'][idx]).float()
        return x, y

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if config["USE_CUDA"] and torch.cuda.is_available() else "cpu")
        self.model_name = config["MODEL"].lower()
        self.multimodal = config.get("MULTIMODAL", False)
        self.len_train = config['TRAIN_LEN']
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_STOCK']
        self.lb = config['LB']
        self.ub = config['UB']
        self.beta = config.get("BETA", 0.2)
        self.loss_fn = config.get("LOSS_FUNCTION", "max_sharpe")
        self.loss_functions = {
            'max_sharpe': max_sharpe,
            'equal_risk_parity': equal_risk_parity,
            'mean_variance': mean_variance,
            'combined': combined_loss,
        }
        if self.loss_fn not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
        self.criterion = self.loss_functions[self.loss_fn]
        self.previous_weights = None

        # Initialize logging to file
        os.makedirs(config['RESULT_DIR'], exist_ok=True)
        model_identifier = f"{self.model_name}_{self.multimodal}_{self.loss_fn}_{self.len_train}_{self.len_pred}"
        log_filename = os.path.join(config['RESULT_DIR'], f"training_{model_identifier}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        logger.info(f"Configuration: {self.config}")
        logger.info(f"Model: {self.model_name}, Multimodal: {self.multimodal}, Loss Function: {self.loss_fn}")

        self.model = self._create_model()
        self.optimizer = self._create_optimizer()

        # Paths and best loss initialization
        self.model_dir = os.path.join(self.config['RESULT_DIR'], 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model_path = self.model_dir  # Directory to store model files
        self.best_loss = float('inf')

        # Early stopping parameters
        self.min_delta = self.config.get("MIN_DELTA", 1e-4)  # Minimum significant improvement
        self.early_stop_count = 0
        self.early_stop_threshold = self.config["EARLY_STOP"]

    def _create_model(self) -> torch.nn.Module:
        common_params = {
            'n_stocks': self.n_stock,
            'lb': self.lb,
            'ub': self.ub,
            'multimodal': self.multimodal
        }
        if self.multimodal:
            img_height, img_width = self._get_image_dimensions()
            return Multimodal(
                model_type=self.model_name,
                model_params={**self.config[self.model_name.upper()], **common_params},
                img_height=img_height,
                img_width=img_width,
                lb=self.lb,
                ub=self.ub
            ).to(self.device)
        elif self.model_name == "gru":
            gru_config = self.config['GRU']
            return GRU(
                n_layers=gru_config['n_layers'],
                hidden_dim=gru_config['hidden_dim'],
                dropout_p=gru_config['dropout_p'],
                bidirectional=gru_config['bidirectional'],
                n_output=gru_config['n_output'],
                **common_params
            ).to(self.device)
        elif self.model_name == "transformer":
            transformer_config = self.config['TRANSFORMER']
            return Transformer(
                n_timestep=transformer_config['n_timestep'],
                n_layer=transformer_config['n_layer'],
                n_head=transformer_config['n_head'],
                n_dropout=transformer_config['n_dropout'],
                n_output=transformer_config['n_output'],
                **common_params
            ).to(self.device)
        elif self.model_name == "tcn":
            tcn_config = self.config['TCN']
            return TCN(
                n_output=tcn_config['n_output'],
                kernel_size=tcn_config['kernel_size'],
                n_dropout=tcn_config['n_dropout'],
                n_timestep=tcn_config['n_timestep'],
                hidden_size=tcn_config['hidden_size'],
                level=tcn_config['level'],
                **common_params
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _create_optimizer(self) -> SAM:
        base_optimizer = torch.optim.SGD
        return SAM(
            self.model.parameters(),
            base_optimizer,
            lr=self.config["LR"],
            momentum=self.config['MOMENTUM']
        )

    def _get_image_dimensions(self) -> Tuple[int, int]:
        dimensions = {
            5: (32, 15),
            20: (64, 60),
            60: (96, 180),
            120: (128, 360)
        }
        if self.len_train in dimensions:
            return dimensions[self.len_train]
        else:
            raise ValueError(f"Unsupported TRAIN_LEN value: {self.len_train}")

    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = PortfolioDataset("data/train_dataset.h5")
        val_dataset = PortfolioDataset("data/validation_dataset.h5")

        train_loader = DataLoader(train_dataset, batch_size=self.config['BATCH'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['BATCH'], shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader

    def train(self) -> None:
        train_dataset, val_dataset = self._load_data()
        train_loader = DataLoader(train_dataset, batch_size=self.config['BATCH'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['BATCH'], shuffle=False, num_workers=4, pin_memory=True)

        valid_loss = []
        train_loss = []
        early_stop_count = self.early_stop_count
        early_stop_threshold = self.early_stop_threshold

        # Load existing checkpoint if available
        checkpoint_files = [f for f in os.listdir(self.model_dir) if f.startswith('best_model')]
        if checkpoint_files:
            # Load the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_epoch_')[1].split('_')[0]))
            checkpoint_path = os.path.join(self.model_dir, latest_checkpoint)
            logger.info(f"Resuming training from saved model: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch'] + 1
            early_stop_count = checkpoint['early_stop_count']
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.config["EPOCHS"] + 1):
            train_epoch_loss = self._run_epoch(train_loader, is_training=True)
            train_loss.append(train_epoch_loss)
            logger.info(f"Epoch {epoch}/{self.config['EPOCHS']}, Training Loss: {train_epoch_loss:.6f}")

            valid_epoch_loss = self._run_epoch(val_loader, is_training=False)
            valid_loss.append(valid_epoch_loss)
            logger.info(f"Epoch {epoch}/{self.config['EPOCHS']}, Validation Loss: {valid_epoch_loss:.6f}")

            # Check if validation loss improved significantly
            if self.best_loss - valid_epoch_loss > self.min_delta:
                self.best_loss = valid_epoch_loss
                self._save_model(epoch, self.best_loss, early_stop_count)
                early_stop_count = 0
                logger.info(f"Validation loss improved. Early stop count reset to 0.")
            else:
                early_stop_count += 1
                logger.info(f"No significant improvement in validation loss. Early stop count: {early_stop_count}")

            if early_stop_count >= early_stop_threshold:
                logger.info(f"Early stopping at epoch {epoch} after {early_stop_count} epochs without significant improvement.")
                break

        # After training, remove unnecessary parameter files if needed
        # self._cleanup_models()

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> float:
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0

        for x, y in tqdm(dataloader, desc="Training" if is_training else "Validation", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            if is_training:
                self.optimizer.zero_grad()

            loss = self._compute_loss((x, y))
            total_loss += loss.item()

            if is_training:
                loss.backward()
                self.optimizer.step()

        average_loss = total_loss / len(dataloader)
        return average_loss

    def _compute_loss(self, data) -> torch.Tensor:
        if self.multimodal:
            x, y, img, labels = [d.to(self.device) for d in data]
            portfolio_weights, binary_pred = self.model(x, img)
        else:
            x, y = [d.to(self.device) for d in data]
            portfolio_weights = self.model(x)

        if self.loss_fn == 'combined_with_turnover':
            if self.previous_weights is None:
                self.previous_weights = torch.zeros_like(portfolio_weights)
            loss = self.criterion(
                y, portfolio_weights, portfolio_weights, self.previous_weights,
                beta=self.config.get('LOSS_BETA', 0.5),
                gamma=self.config.get('TURNOVER_GAMMA', 0.1),
                transaction_cost=self.config.get('TRANSACTION_COST', 0.001)
            )
            self.previous_weights = portfolio_weights.detach()
        else:
            loss = self.criterion(y, portfolio_weights)

        return loss

    def _save_model(self, epoch, best_loss, early_stop_count) -> None:
        model_filename = f"best_model_epoch_{epoch}_loss_{best_loss:.4f}.pth"
        self.best_model_path = os.path.join(self.model_dir, model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
            'early_stop_count': early_stop_count
        }, self.best_model_path)
        logger.info(f"Best model saved to {self.best_model_path}")

    def _cleanup_models(self) -> None:
        # Optional: Implement cleanup logic if necessary
        pass

    def run(self) -> None:
        self.train()

    def get_best_model_path(self) -> str:
        return self.best_model_path
