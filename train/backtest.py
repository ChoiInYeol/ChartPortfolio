# backtest.py
import os
import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from model.multimodal import Multimodal
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from .utils import load_model
from typing import Dict, Any, Tuple
import json
import logging
from .visualize import (
    visualize_backtest,
    visualize_weights,
    visualize_returns_distribution,
    visualize_drawdown,
    visualize_rolling_sharpe,
    visualize_binary_predictions
)
from model.loss import max_sharpe, equal_risk_parity, combined_loss

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('backtest.log')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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


class Backtester:
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
        self.beta = config.get("BETA", 0.5)  # 새로운 하이퍼파라미터 추가
        self.criterion = combined_loss  # loss.py에서 정의한 combined_loss 사용
        logger.info(f"Configuration: {self.config}")
        
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

    def backtest(self, model_file: str = None, visualize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if model_file is None:
            model_file = f"best_model_weight_{self.config['MODEL']}_{self.best_model_count}_{self.config['MULTIMODAL']}"
        self.model = load_model(self.model, f"{model_file}", use_cuda=True)

        _, test_dataset = self._load_and_process_data()
        
        logger.debug(f"Test image data shape: {test_dataset.img_data.shape}")
        
        if self.multimodal:
            myPortfolio, equalPortfolio, myWeights, binary_preds = self._calculate_portfolio_performance(test_dataset)
            performance, stats, myWeights_df, binary_preds_df = self._create_performance_dataframe(myPortfolio, equalPortfolio, myWeights, binary_preds)
        else:
            myPortfolio, equalPortfolio, myWeights = self._calculate_portfolio_performance(test_dataset)
            performance, stats, myWeights_df = self._create_performance_dataframe(myPortfolio, equalPortfolio, myWeights)

        if self.multimodal:
            return performance, stats, myWeights_df, binary_preds_df
        else:
            return performance, stats, myWeights_df

    def _calculate_portfolio_performance(self, test_dataset):
        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        binary_preds = [] if self.multimodal else None
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, test_dataset.x_data.shape[0], self.LEN_PRED):
                x = test_dataset.x_data[i][np.newaxis, :, :].to(self.device)
                if self.multimodal:
                    img = test_dataset.img_data[i][np.newaxis, :, :, :].to(self.device)
                    out, binary_pred = self.model(x, img)
                    binary_preds.append(binary_pred.detach().cpu().numpy())
                else:
                    out = self.model(x.float().cuda())
                
                myWeights.append(out.detach().cpu().numpy())
                m_rtn = np.sum(self.test_y_raw[i], axis=0)
                myPortfolio.append(myPortfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy().squeeze(), m_rtn)))
                equalPortfolio.append(equalPortfolio[-1] * np.exp(np.dot(EWPWeights, m_rtn)))
        
        if self.multimodal:
            return myPortfolio, equalPortfolio, np.array(myWeights), np.array(binary_preds)
        else:
            return myPortfolio, equalPortfolio, np.array(myWeights)

    def _create_performance_dataframe(self, myPortfolio, equalPortfolio, myWeights, binary_preds=None):
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

        if self.multimodal:
            myWeights = np.squeeze(myWeights)  # (min_length, 1, 50) -> (min_length, 50)
            binary_preds = np.squeeze(binary_preds)[:min_length]  # (min_length, 1, 50) -> (min_length, 50)
            
            # performance.index의 길이에 맞춰 자르기
            myWeights = myWeights[:len(performance.index)]
            binary_preds = binary_preds[:len(performance.index)]
            
            myWeights_df = pd.DataFrame(myWeights, index=performance.index)
            binary_preds_df = pd.DataFrame(binary_preds, index=performance.index)
            return performance, stats, myWeights_df, binary_preds_df
        else:
            myWeights = np.squeeze(myWeights)[:len(performance.index)]
            myWeights_df = pd.DataFrame(myWeights, index=performance.index)
            return performance, stats, myWeights_df

# Usage example
if __name__ == "__main__":
    # Load configuration
    config = json.load(open("config/train_config.json", "r"))
    # Initialize backtester
    backtester = Backtester(config)
    # Perform backtesting
    performance, stats, weights = backtester.backtest(model_file='path_to_model.pt', visualize=True)
