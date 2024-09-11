import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.multimodal import Multimodal
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity
from train.utils import save_model, load_model
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if self.config["USE_CUDA"] else "cpu"
        self.model_name = self.config["MODEL"]
        self.multimodal = self.config["MULTIMODAL"]
        self.best_model_count = 0
        self.train_len = self.config['TRAIN_LEN']
        self.pred_len = self.config['PRED_LEN']
        
        print(f"model_name: {self.model_name}, multimodal: {self.multimodal}, pred_len: {self.pred_len}")
        
        # 모델별 파라미터 설정
        model_params = self._get_model_params()
        
        if self.multimodal:
            self.model = Multimodal(
                model_type=self.model_name,
                model_params=model_params,
                cnn_output_dim=self.config["CNN_OUTPUT_DIM"],
                pred_len=self.pred_len,
                verbose=self.config.get("VERBOSE", False)
            ).to(self.device)
        else:
            self.model = self._create_model(model_params).to(self.device)
        
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(
            self.model.parameters(), base_optimizer, lr=self.config["LR"],
            momentum=self.config['MOMENTUM']
        )
        self.criterion = max_sharpe

    def _get_model_params(self):
        common_params = {
            'lb': self.config['LB'],
            'ub': self.config['UB'],
            'multimodal': self.multimodal,
            'cnn_output_dim': self.config["CNN_OUTPUT_DIM"],
            'verbose': self.config.get("VERBOSE", False),
            'n_stocks': self.config['N_FEAT'],
            'pred_len': self.pred_len
        }
        
        if self.model_name.lower() == "gru":
            return {**self.config["GRU"], **common_params}
        elif self.model_name.lower() == "tcn":
            return {**self.config["TCN"], **common_params}
        elif self.model_name.lower() == "transformer":
            return {**self.config["TRANSFORMER"], **common_params}
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _create_model(self, model_params):
        if self.model_name.lower() == "gru":
            return GRU(**model_params)
        elif self.model_name.lower() == "tcn":
            return TCN(**model_params)
        elif self.model_name.lower() == "transformer":
            return Transformer(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _dataload(self):
        with open("data/dataset.pkl", "rb") as f:
            train_x_raw, train_y_raw, test_x_raw, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            test_date = pickle.load(f)
        self.train_x_raw = train_x_raw
        self.train_y_raw = train_y_raw
        self.test_x_raw = test_x_raw
        self.test_y_raw = test_y_raw
        self.test_date = test_date
        
    def _dataload_image(self, chart_dir="data/charts/"):
        if not self.multimodal:
            return
        
        self.all_images = {}
        self.all_meta_data = {}
        for file in os.listdir(chart_dir):
            if file.endswith(".npz"):
                ticker = file.split("_")[2]
                data = np.load(os.path.join(chart_dir, file), allow_pickle=True)
                self.all_images[ticker] = data['images']
                self.all_meta_data[ticker] = data['meta_data']

    def _scale_data(self, scale=20):
        self.train_x = torch.from_numpy(self.train_x_raw.astype("float32") * scale)
        self.train_y = torch.from_numpy(self.train_y_raw.astype("float32") * scale)
        self.test_x = torch.from_numpy(self.test_x_raw.astype("float32") * scale)
        self.test_y = torch.from_numpy(self.test_y_raw.astype("float32") * scale)

    def _set_parameter(self):
        self.LEN_TRAIN = self.train_x.shape[1]
        self.LEN_PRED = self.train_y.shape[1]
        self.N_STOCK = self.config["N_FEAT"]

    def _shuffle_data(self):
        randomized = np.arange(len(self.train_x))
        np.random.shuffle(randomized)
        self.train_x = self.train_x[randomized]
        self.train_y = self.train_y[randomized]

    def set_data(self):
        self._dataload()
        self._scale_data()
        
        # 데이터 확인
        print("train_x_raw shape after alignment:", self.train_x_raw.shape)
        print("train_y_raw shape after alignment:", self.train_y_raw.shape)
        print("test_x_raw shape after alignment:", self.test_x_raw.shape)
        print("test_y_raw shape after alignment:", self.test_y_raw.shape)
        print("test_date type:", type(self.test_date))
        print("test_date length:", len(self.test_date) if hasattr(self.test_date, '__len__') else "N/A")
        self._set_parameter()
        self._shuffle_data()

    def dataloader(self, x, y):
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["BATCH"],
            shuffle=False,
            drop_last=True,
        )

    def train(self, visualize=True):
        train_loader = self.dataloader(self.train_x, self.train_y)
        test_loader = self.dataloader(self.test_x, self.test_y)

        valid_loss = []
        train_loss = []

        early_stop_count = 0
        early_stop_th = self.config["EARLY_STOP"]

        for epoch in tqdm(range(self.config["EPOCHS"]), desc="Epochs"):
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = test_loader

                running_loss = 0.0

                pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Progress", leave=False)
                for idx, data in enumerate(pbar):
                    x, y = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        out = self.model(x)
                        loss = self.criterion(y, out)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.first_step(zero_grad=True)
                            self.criterion(y, self.model(x)).backward()
                            self.optimizer.second_step(zero_grad=True)

                    running_loss += loss.item() / len(dataloader)
                    pbar.set_postfix({'loss': running_loss / (idx + 1)})

                if phase == "train":
                    train_loss.append(running_loss)
                else:
                    valid_loss.append(running_loss)
                    if running_loss <= min(valid_loss):
                        self.best_model_count += 1
                        save_model(self.model, "result", f"{self.config['MODEL']}_{self.best_model_count}")
                        tqdm.write(f"Improved! at {epoch + 1} epochs, with {running_loss:.6f}")
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stop_count == early_stop_th:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

        if visualize:
            self._save_training_loss(train_loss, valid_loss)

        return self.model, train_loss, valid_loss

    def _save_training_loss(self, train_loss, valid_loss):
        # 에폭 번호 생성
        epochs = range(1, len(train_loss) + 1)
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': train_loss,
            'Validation Loss': valid_loss
        })
        
        # CSV 파일로 저장
        df.to_csv(f"result/training_loss_{self.config['MODEL']}_{self.config['MULTIMODAL']}.csv", index=False)
        print(f"훈련 및 검증 손실이 result/training_loss_{self.config['MODEL']}_{self.config['MULTIMODAL']}.csv에 저장되었습니다.")

    def backtest(self, visualize=True):
        self.model = load_model(self.model, f"result/best_model_weight_{self.config['MODEL']}.pt", use_cuda=True)

        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        for i in range(0, self.test_x.shape[0], self.LEN_PRED):
            x = self.test_x[i][np.newaxis, :, :]
            out = self.model(x.float().cuda())[0]
            myWeights.append(out.detach().cpu().numpy())
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            
            print(f"x 형태: {x.shape},\n"
                  f"m_rtn 형태: {m_rtn.shape},\n"
                  f"out 형태: {out.detach().cpu().numpy().shape},\n"
                  f"내적 형태: {np.dot(out.detach().cpu().numpy(), m_rtn).shape}")
            
            myPortfolio.append(
                myPortfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy(), m_rtn))
            )
            equalPortfolio.append(
                equalPortfolio[-1] * np.exp(np.dot(EWPWeights, m_rtn))
            )

        idx = np.arange(0, len(self.test_date), self.LEN_PRED)
        performance = pd.DataFrame(
            {"EWP": equalPortfolio, "MyPortfolio": myPortfolio},
            index=np.array(self.test_date)[idx],
        )
        index_sp = pd.DataFrame(
            pd.read_csv("data/snp500_index.csv", index_col="Date")["Adj Close"]
        )
        index_sp = index_sp[self.test_date[0] :]
        performance["index_sp"] = index_sp["Adj Close"] * (
            myPortfolio[0] / index_sp["Adj Close"][0]
        )
        performance.to_csv("result/backtest.csv")

        if visualize:
            self._visualize_backtest(performance)
            self._visualize_weights(performance, myWeights)

        result = performance.copy()
        result["EWP_Return"] = np.log(result["EWP"]) - np.log(result["EWP"].shift(1))
        result["My_Return"] = np.log(result["MyPortfolio"]) - np.log(
            result["MyPortfolio"].shift(1)
        )
        result["Index_Return"] = np.log(result["index_sp"]) - np.log(
            result["index_sp"].shift(1)
        )
        result = result.dropna()

        expectedReturn = result[["EWP_Return", "My_Return", "Index_Return"]].mean()
        expectedReturn *= 12
        print("Annualized Return of Portfolio")
        print(expectedReturn)
        print("-" * 20)
        volatility = result[["EWP_Return", "My_Return", "Index_Return"]].std()
        volatility *= np.sqrt(12)
        print("Annualized Volatility of Portfolio")
        print(volatility)
        print("-" * 20)
        print("Annualized Sharp Ratio of Portfolio")
        print((expectedReturn / volatility))
        print("-" * 20)
        print("MDD")
        mdd_df = result[["EWP", "MyPortfolio", "index_sp"]].apply(self._get_mdd)
        print(mdd_df)

    def _visualize_backtest(self, performance):
        performance.plot(figsize=(14, 7), fontsize=10)
        plt.legend(fontsize=10)
        plt.savefig(f"result/performance_{self.config['MODEL']}.png")

    def _visualize_weights(self, performance, weights):
        weights = np.array(weights)
        ticker = pd.read_csv("data/return_df.csv", index_col=0).columns
        n = self.N_STOCK
        plt.figure(figsize=(15, 10))
        for i in range(n):
            plt.plot(weights[:, i], label=ticker[i])
        plt.title("Weights")
        plt.xticks(
            np.arange(0, len(list(performance.index[1:]))),
            list(performance.index[1:]),
            rotation="vertical",
        )
        plt.legend()
        plt.savefig(f"result/weights_{self.config['MODEL']}.png")

    def _get_mdd(self, x):
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return (
            x.index[peak_upper],
            x.index[peak_lower],
            (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper],
        )
