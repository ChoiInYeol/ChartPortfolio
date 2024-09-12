import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model.multimodal import Multimodal
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity, combined_loss
from train.utils import save_model, load_model
from tqdm import tqdm
import os
import logging
from typing import Dict, Tuple, List, Any
import scienceplots
plt.style.use('science')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, img_data: np.ndarray, labels: np.ndarray, len_pred: int):
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()
        self.img_data = torch.from_numpy(img_data).float()
        self.labels = torch.from_numpy(labels).float()
        self.len_pred = len_pred

    def __len__(self) -> int:
        return len(self.x_data)  # 변경: 전체 길이 반환

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx], self.img_data[idx], self.labels[idx]  # 변경: 직접 인덱스 사용

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if config["USE_CUDA"] and torch.cuda.is_available() else "cpu")
        self.model_name = config["MODEL"].lower()
        self.multimodal = config.get("MULTIMODAL", False)
        self.len_train = config['TRAIN_LEN']
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_FEAT']
        self.lb = config['LB']
        self.ub = config['UB']
        self.best_model_count = 0
        self.alpha = config.get("ALPHA", 0.5)
        
        self.criterion = combined_loss if self.multimodal else max_sharpe
        
        logger.info(f"Model: {self.model_name}, Multimodal: {self.multimodal}, Prediction Length: {self.len_pred}")
        
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()

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
        
        if self.model_name == "gru":
            gru_config = self.config['GRU']
            return GRU(
                n_layers=gru_config['n_layers'],
                hidden_dim=gru_config['hidden_dim'],
                dropout_p=gru_config['dropout_p'],
                bidirectional=gru_config['bidirectional'],
                **common_params
            ).to(self.device)
        
        if self.model_name == "transformer":
            transformer_config = self.config['TRANSFORMER']
            return Transformer(
                n_timestep=transformer_config['n_timestep'],
                n_layer=transformer_config['n_layer'],
                n_head=transformer_config['n_head'],
                n_dropout=transformer_config['n_dropout'],
                n_output=transformer_config['n_output'],
                **common_params
            ).to(self.device)
        
        if self.model_name == "tcn":
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
        
        raise ValueError(f"지원되지 않는 모델 유형: {self.model_name}")

    def _create_optimizer(self) -> SAM:
        base_optimizer = torch.optim.SGD
        return SAM(
            self.model.parameters(),
            base_optimizer,
            lr=self.config["LR"],
            momentum=self.config['MOMENTUM']
        )

    def _get_image_dimensions(self) -> Tuple[int, int]:
        if self.len_train == 5:
            return 32, 15
        elif self.len_train == 20:
            return 64, 60
        elif self.len_train == 60:
            return 96, 180
        elif self.len_train == 120:
            return 128, 360
        else:
            raise ValueError(f"Unsupported TRAIN_LEN value: {self.len_train}")

    def _load_and_process_data(self) -> Tuple[PortfolioDataset, PortfolioDataset]:
        logger.info("Loading and processing data...")
        
        with open("data/dataset.pkl", "rb") as f:
            train_x_raw, train_y_raw, test_x_raw, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            date_info = pickle.load(f)
            
        with open("data/dataset_img.pkl", "rb") as f:
            train_img_data, test_img_data, times_tr, times_te = pickle.load(f)
        
        scale = 20
        train_x = train_x_raw * scale
        train_y = train_y_raw * scale
        test_x = test_x_raw * scale
        test_y = test_y_raw * scale
        
        train_img = train_img_data['images']
        test_img = test_img_data['images']
        train_labels = np.array(train_img_data['labels'])
        test_labels = np.array(test_img_data['labels'])

        self.train_date = date_info['train']
        self.test_date = date_info['test']
        self.N_STOCK = self.config['N_FEAT']
        self.LEN_PRED = train_y.shape[1]
        
        self.test_y_raw = test_y_raw

        train_dataset = PortfolioDataset(train_x, train_y, train_img, train_labels, self.LEN_PRED)
        test_dataset = PortfolioDataset(test_x, test_y, test_img, test_labels, self.LEN_PRED)
        return train_dataset, test_dataset

    def train(self, visualize: bool = True) -> Tuple[torch.nn.Module, List[float], List[float]]:
        train_dataset, test_dataset = self._load_and_process_data()
        train_loader = DataLoader(train_dataset, batch_size=self.config['BATCH'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['BATCH'], shuffle=False)
        
        logger.debug(f"Train data shape: x={train_dataset.x_data.shape}, y={train_dataset.y_data.shape}")
        logger.debug(f"Image data shape: {train_dataset.img_data.shape if self.multimodal else 'N/A'}")
        logger.debug(f"Labels shape: {train_dataset.labels.shape if self.multimodal else 'N/A'}")
        
        valid_loss = []
        train_loss = []
        early_stop_count = 0
        early_stop_th = self.config["EARLY_STOP"]

        for epoch in tqdm(range(self.config["EPOCHS"]), desc="Epochs"):
            train_loss.append(self._run_epoch(train_loader, is_training=True))
            valid_loss.append(self._run_epoch(test_loader, is_training=False))

            if valid_loss[-1] <= min(valid_loss):
                self._save_model(epoch, valid_loss[-1])
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count == early_stop_th:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if visualize:
            self._save_training_loss(train_loss, valid_loss)

        return self.model, train_loss, valid_loss

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> float:
        self.model.train() if is_training else self.model.eval()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Valid'} Progress", leave=False)
        for idx, data in enumerate(pbar):
            loss = self._compute_loss(data, is_training)
            running_loss += loss.item() / len(dataloader)
            pbar.set_postfix({'loss': running_loss / (idx + 1)})

        return running_loss

    def _compute_loss(self, data, is_training: bool) -> torch.Tensor:
        if self.multimodal:
            x, y, img, labels = [d.to(self.device) for d in data]
        else:
            x, y = [d.to(self.device) for d in data[:2]]
            img, labels = None, None

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            if self.multimodal:
                portfolio_weights, binary_pred = self.model(x, img)
                loss = self.criterion(y, portfolio_weights, labels, binary_pred, self.alpha)
            else:
                portfolio_weights = self.model(x)
                loss = self.criterion(y, portfolio_weights, None, None, self.alpha)

            if is_training:
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self._compute_second_order_loss(x, y, img, labels)
                self.optimizer.second_step(zero_grad=True)

        return loss

    def _compute_second_order_loss(self, x, y, img, labels):
        if self.multimodal:
            portfolio_weights, binary_pred = self.model(x, img)
            self.criterion(y, portfolio_weights, labels, binary_pred, self.alpha).backward()
        else:
            portfolio_weights = self.model(x)
            self.criterion(y, portfolio_weights, None, None, self.alpha).backward()

    def _save_model(self, epoch: int, loss: float):
        self.best_model_count += 1
        save_model(self.model, "result", f"{self.config['MODEL']}_{self.best_model_count}_{loss:.6f}")
        logger.info(f"Improved! Epoch {epoch + 1}, Loss: {loss:.6f}")

    def backtest(self, model_file: str = None, visualize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if model_file is None:
            model_file = f"best_model_weight_{self.config['MODEL']}_{self.best_model_count}_{self.config['MULTIMODAL']}"
        self.model = load_model(self.model, f"{model_file}", use_cuda=True)

        _, test_dataset = self._load_and_process_data()
        
        logger.debug(f"Test image data shape: {test_dataset.img_data.shape}")
        
        myPortfolio, equalPortfolio, myWeights = self._calculate_portfolio_performance(test_dataset)
        
        performance, stats, myWeights_df = self._create_performance_dataframe(myPortfolio, equalPortfolio, myWeights)

        self._log_backtest_results(stats)

        return performance, stats, myWeights_df

    def _calculate_portfolio_performance(self, test_dataset):
        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, test_dataset.x_data.shape[0], self.LEN_PRED):
                x = test_dataset.x_data[i][np.newaxis, :, :].to(self.device)
                if self.multimodal:
                    img = test_dataset.img_data[i][np.newaxis, :, :, :].to(self.device)
                    out, _, _ = self.model(x, img)
                else:
                    out = self.model(x.float().cuda())[0]
                
                myWeights.append(out.detach().cpu().numpy())
                m_rtn = np.sum(self.test_y_raw[i], axis=0)
                myPortfolio.append(myPortfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy(), m_rtn)))
                equalPortfolio.append(equalPortfolio[-1] * np.exp(np.dot(EWPWeights, m_rtn)))
        
        return myPortfolio, equalPortfolio, myWeights

    def _create_performance_dataframe(self, myPortfolio, equalPortfolio, myWeights):
        min_length = min(len(myPortfolio), len(equalPortfolio), len(self.test_date) // self.LEN_PRED)
        myPortfolio = myPortfolio[:min_length]
        equalPortfolio = equalPortfolio[:min_length]
        myWeights = myWeights[:min_length]
        
        date_index = pd.to_datetime(self.test_date[::self.LEN_PRED][:min_length])

        performance = pd.DataFrame(
            {"EWP": equalPortfolio, "MyPortfolio": myPortfolio},
            index=date_index
        )

        index_sp = pd.read_csv("data/snp500_index.csv", index_col="Date", parse_dates=True)["Adj Close"]
        index_sp = index_sp.loc[performance.index[0]:performance.index[-1]]
        index_sp = index_sp.reindex(performance.index)

        performance = performance.fillna(method='ffill')
        performance["index_sp"] = index_sp * (performance["MyPortfolio"].iloc[0] / index_sp.iloc[0])

        performance.to_csv(os.path.join(self.config['RESULT_DIR'], "backtest.csv"))

        result = performance.copy()
        result["EWP_Return"] = result["EWP"].pct_change()
        result["My_Return"] = result["MyPortfolio"].pct_change()
        result["Index_Return"] = result["index_sp"].pct_change()
        result = result.dropna()

        annualization_factor = 252 / self.LEN_PRED

        expectedReturn = result[["EWP_Return", "My_Return", "Index_Return"]].mean() * annualization_factor
        volatility = result[["EWP_Return", "My_Return", "Index_Return"]].std() * np.sqrt(annualization_factor)
        sharpRatio = expectedReturn / volatility

        mdd = result[["EWP", "MyPortfolio", "index_sp"]].apply(self._get_mdd)

        stats = pd.DataFrame({
            "Annualized Return": expectedReturn,
            "Annualized Volatility": volatility,
            "Annualized Sharpe Ratio": sharpRatio,
            "MDD": mdd
        })

        myWeights_df = pd.DataFrame(myWeights, index=performance.index[:len(myWeights)])

        return performance, stats, myWeights_df

    def _visualize_backtest(self, performance):
        plt.figure(figsize=(14, 7))
        for col in performance.columns:
            plt.plot(performance.index, performance[col], label=col)
        
        plt.title("Portfolio Performance Comparison", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Portfolio Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(self.config['RESULT_DIR'], f"performance_{self.config['MODEL']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_weights(self, performance, weights):
        plt.figure(figsize=(14, 7))
        weights_df = pd.DataFrame(weights, index=performance.index[:len(weights)])
        
        for col in weights_df.columns:
            plt.plot(weights_df.index, weights_df[col], label=f'Stock {col+1}')
        
        plt.title("Portfolio Weights Over Time", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Weight", fontsize=14)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(self.config['RESULT_DIR'], f"weights_{self.config['MODEL']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_returns_distribution(self, performance):
        returns = performance.pct_change().dropna()
        
        plt.figure(figsize=(14, 7))
        for col in returns.columns:
            sns.kdeplot(returns[col], label=col)
        
        plt.title("Returns Distribution", fontsize=16)
        plt.xlabel("Returns", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(self.config['RESULT_DIR'], f"returns_distribution_{self.config['MODEL']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_drawdown(self, performance):
        def calculate_drawdown(series):
            wealth_index = (1 + series.pct_change()).cumprod()
            previous_peaks = wealth_index.cummax()
            drawdowns = (wealth_index - previous_peaks) / previous_peaks
            return drawdowns

        plt.figure(figsize=(14, 7))
        for col in performance.columns:
            drawdown = calculate_drawdown(performance[col])
            plt.plot(drawdown.index, drawdown, label=col)
        
        plt.title("Portfolio Drawdown", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Drawdown", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(self.config['RESULT_DIR'], f"drawdown_{self.config['MODEL']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_rolling_sharpe(self, performance, window=252):
        returns = performance.pct_change().dropna()
        rolling_sharpe = returns.rolling(window=window).apply(lambda x: np.sqrt(252) * x.mean() / x.std())
        
        plt.figure(figsize=(14, 7))
        for col in rolling_sharpe.columns:
            plt.plot(rolling_sharpe.index, rolling_sharpe[col], label=col)
        
        plt.title(f"Rolling Sharpe Ratio (Window: {window} days)", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Sharpe Ratio", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(self.config['RESULT_DIR'], f"rolling_sharpe_{self.config['MODEL']}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _get_mdd(self, x) -> float:
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

    def save_results(self, model_name: str, training_loss: pd.DataFrame = None, performance: pd.DataFrame = None, stats: pd.DataFrame = None, weights: pd.DataFrame = None):
        result_dir = self.config['RESULT_DIR']
        os.makedirs(result_dir, exist_ok=True)
        
        if training_loss is not None:
            training_loss.to_csv(os.path.join(result_dir, f"{model_name}_training_loss.csv"), index=False)
        if performance is not None:
            performance.to_csv(os.path.join(result_dir, f"{model_name}_backtest_performance.csv"))
        if stats is not None:
            stats.to_csv(os.path.join(result_dir, f"{model_name}_backtest_stats.csv"))
        if weights is not None:
            weights.to_csv(os.path.join(result_dir, f"{model_name}_portfolio_weights.csv"))
        
        logger.info(f"Results saved for model: {model_name}")

    def run_experiment(self, train: bool = True, visualize: bool = True, model_file: str = None) -> Tuple[torch.nn.Module, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            model = None
            training_loss_df = None

            if train:
                logger.info("Starting training...")
                model, train_loss, valid_loss = self.train(visualize=visualize)
                logger.info("Training completed.")
                
                training_loss_df = pd.DataFrame({
                    'epoch': range(1, len(train_loss) + 1),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                })
            else:
                logger.info("Skipping training as per configuration.")

            logger.info("Starting backtesting...")
            performance, stats, weights = self.backtest(model_file=model_file, visualize=visualize)
            logger.info("Backtesting completed.")
            
            if visualize:
                self._visualize_backtest(performance)
                self._visualize_weights(performance, weights)
                self._visualize_returns_distribution(performance)
                self._visualize_drawdown(performance)
                self._visualize_rolling_sharpe(performance)
            
            model_name = f"{self.config['MODEL']}_{self.config['MULTIMODAL']}"
            self.save_results(model_name, training_loss_df, performance, stats, weights)
            
            logger.info("Experiment results have been saved.")
            return model, performance, stats, weights
        except Exception as e:
            logger.error(f"Error in experiment: {e}")
            raise
        
