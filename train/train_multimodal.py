import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from model.multimodal import MultimodalPortfolio
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity
from train.utils import save_model, load_model

class MultimodalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if self.config["USE_CUDA"] else "cpu"
        self.model = MultimodalPortfolio(
            n_layers=self.config["N_LAYER"],
            hidden_dim=self.config["HIDDEN_DIM"],
            n_stocks=self.config["N_FEAT"],
            cnn_output_dim=self.config["CNN_OUTPUT_DIM"],
            dropout_p=self.config["DROPOUT"],
            bidirectional=self.config["BIDIRECTIONAL"],
            lb=self.config['LB'],
            ub=self.config['UB'],
            verbose=self.config.get("VERBOSE", False)
        ).to(self.device)
        
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(
            self.model.parameters(), base_optimizer, lr=self.config["LR"],
            momentum=self.config['MOMENTUM']
        )
        self.criterion = max_sharpe

    def _load_numerical_data(self):
        with open("data/dataset.pkl", "rb") as f:
            train_x_raw, train_y_raw, test_x_raw, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            test_date = pickle.load(f)
        self.train_x_raw = train_x_raw
        self.train_y_raw = train_y_raw
        self.test_x_raw = test_x_raw
        self.test_y_raw = test_y_raw
        self.test_date = test_date

    def _load_image_data(self, chart_dir="data/charts/"):
        self.all_images = {}
        self.all_meta_data = {}
        for file in os.listdir(chart_dir):
            if file.endswith(".npz"):
                ticker = file.split("_")[2]
                data = np.load(os.path.join(chart_dir, file), allow_pickle=True)
                self.all_images[ticker] = data['images']
                self.all_meta_data[ticker] = data['meta_data']
                
    def _process_dates(self):
        if isinstance(self.test_date, dict):
            self.train_dates = self.test_date.get('train', [])
            self.test_dates = self.test_date.get('test', [])
        else:
            self.train_dates = self.test_date[:len(self.train_x_raw)]
            self.test_dates = self.test_date[len(self.train_x_raw):]

    def _align_data(self):
        self.aligned_train = self._align_data_helper(self.train_x_raw, self.train_y_raw, self.train_dates)
        self.aligned_test = self._align_data_helper(self.test_x_raw, self.test_y_raw, self.test_dates)
        
        self.train_x_raw = np.array([item[0] for item in self.aligned_train])
        self.train_y_raw = np.array([item[2] for item in self.aligned_train])
        self.test_x_raw = np.array([item[0] for item in self.aligned_test])
        self.test_y_raw = np.array([item[2] for item in self.aligned_test])
    
    def _align_data_helper(self, numerical_data, numerical_labels, dates):
        aligned_data = []
        n_stocks = self.config['N_FEAT']
        tickers = list(self.all_images.keys())[:n_stocks]
        
        for i, end_date in enumerate(dates):
            if i >= len(numerical_data):
                print(f"Warning: index {i} is out of range for numerical_data with length {len(numerical_data)}")
                break
            
            window_data = numerical_data[i]
            window_label = numerical_labels[i]
            img_data = []
            for ticker in tickers:
                img_index = next((idx for idx, meta in enumerate(self.all_meta_data[ticker]) if pd.Timestamp(meta['End_Date']) <= pd.Timestamp(end_date)), None)
                if img_index is not None:
                    img_data.append(self.all_images[ticker][img_index])
                else:
                    img_data.append(np.zeros((64, 117)))  # 이미지 데이터가 없는 경우 빈 이미지로 대체
            aligned_data.append((window_data, np.array(img_data), window_label))
        
        return aligned_data
    
    def _print_data_info(self):
        print("train_x_raw shape:", self.train_x_raw.shape)
        print("train_y_raw shape:", self.train_y_raw.shape)
        print("test_x_raw shape:", self.test_x_raw.shape)
        print("test_y_raw shape:", self.test_y_raw.shape)
        print("test_date type:", type(self.test_date))
        print("test_date length:", len(self.test_date) if hasattr(self.test_date, '__len__') else "N/A")
        print("aligned_train length:", len(self.aligned_train))
        print("aligned_test length:", len(self.aligned_test))
        print("train_x_raw shape after alignment:", self.train_x_raw.shape)
        print("train_y_raw shape after alignment:", self.train_y_raw.shape)
        print("test_x_raw shape after alignment:", self.test_x_raw.shape)
        print("test_y_raw shape after alignment:", self.test_y_raw.shape)

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
        self._load_numerical_data()
        self._load_image_data()
        self._process_dates()
        self._align_data()
        self._scale_data()
        self._set_parameter()
        self._shuffle_data()
        self._print_data_info()
        
    def dataloader(self, data):
        x_num = torch.tensor([item[0] for item in data]).float()
        x_img = torch.tensor([item[1] for item in data]).float()
        y = torch.tensor([item[2] for item in data]).float()
        
        dataset = torch.utils.data.TensorDataset(x_num, x_img, y)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["BATCH"],
            shuffle=False,
            drop_last=True,
        )

    def train(self, visualize=True):
        train_loader = self.dataloader(self.aligned_train)
        test_loader = self.dataloader(self.aligned_test)

        valid_loss = []
        train_loss = []

        early_stop_count = 0
        early_stop_th = self.config["EARLY_STOP"]

        for epoch in range(self.config["EPOCHS"]):
            print("Epoch {}/{}".format(epoch + 1, self.config["EPOCHS"]))
            print("-" * 10)
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = test_loader

                running_loss = 0.0

                for idx, (x_num, x_img, y) in enumerate(dataloader):
                    x_num = x_num.float().to(self.device)
                    x_img = x_img.float().to(self.device)
                    y = y.float().to(self.device)
                    
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        out = self.model(x_num, x_img)
                        loss = self.criterion(y, out)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.first_step(zero_grad=True)
                            self.criterion(y, self.model(x_num, x_img)).backward()
                            self.optimizer.second_step(zero_grad=True)

                    running_loss += loss.item() / len(dataloader)
                if phase == "train":
                    train_loss.append(running_loss)
                else:
                    valid_loss.append(running_loss)
                    if running_loss <= min(valid_loss):
                        save_model(self.model, "result", "multimodal")
                        print(f"Improved! at {epoch + 1} epochs, with {running_loss}")
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stop_count == early_stop_th:
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
        df.to_csv('result/training_loss_multimodal.csv', index=False)
        print("훈련 및 검증 손실이 'result/training_loss_multimodal.csv'에 저장되었습니다.")

    def backtest(self, visualize=True):
        self.model = load_model(self.model, "result/best_model_weight_multimodal.pt", use_cuda=True)

        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        
        for i in range(0, len(self.aligned_test) - self.LEN_PRED, self.LEN_PRED):
            x_num, x_img, _ = self.aligned_test[i]
            x_num = torch.tensor(x_num).float().unsqueeze(0).to(self.device)
            x_img = torch.tensor(x_img).float().unsqueeze(0).to(self.device)
            
            out = self.model(x_num, x_img)[0]
            myWeights.append(out.detach().cpu().numpy())
            
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            print(m_rtn.shape)
            print(out.detach().cpu().numpy().shape)
            print(np.dot(out.detach().cpu().numpy(), m_rtn).shape)
            myPortfolio.append(
                myPortfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy(), m_rtn))
            )
            equalPortfolio.append(
                equalPortfolio[-1] * np.exp(np.dot(EWPWeights, m_rtn))
            )

        idx = np.arange(0, len(self.test_date) - self.LEN_PRED, self.LEN_PRED)
        performance = pd.DataFrame(
            {"EWP": equalPortfolio, "MyPortfolio": myPortfolio},
            index=np.array(self.test_date)[idx],
        )
        index_sp = pd.DataFrame(
            pd.read_csv("data/snp500_index.csv", index_col="Date")["Adj Close"]
        )
        index_sp = index_sp[self.test_date[0] : self.test_date[-self.LEN_PRED]]
        performance["index_sp"] = index_sp["Adj Close"] * (
            myPortfolio[0] / index_sp["Adj Close"].iloc[0]
        )
        performance.to_csv("result/backtest_multimodal.csv")

        if visualize:
            self._visualize_backtest(performance)
            self._visualize_weights(performance, myWeights)

        self._calculate_performance_metrics(performance)

    def _visualize_backtest(self, performance):
        performance.plot(figsize=(14, 7), fontsize=10)
        plt.legend(fontsize=10)
        plt.savefig("result/performance_multimodal.png")
        plt.show()

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
        plt.savefig("result/weights_multimodal.png")
        plt.show()

    def _get_mdd(self, x):
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return (
            x.index[peak_upper],
            x.index[peak_lower],
            (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper],
        )

    def _calculate_performance_metrics(self, performance):
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

if __name__ == "__main__":
    config = {
        "USE_CUDA": True,
        "N_LAYER": 2,
        "HIDDEN_DIM": 64,
        "N_FEAT": 10,
        "CNN_OUTPUT_DIM": 32,
        "DROPOUT": 0.3,
        "BIDIRECTIONAL": False,
        "LB": 0,
        "UB": 0.1,
        "LR": 0.01,
        "MOMENTUM": 0.9,
        "BATCH": 32,
        "EPOCHS": 100,
        "EARLY_STOP": 10,
        "TRAIN_LEN": 60,
        "PRED_LEN": 5,
        "VERBOSE": False
    }

    trainer = MultimodalTrainer(config)
    trainer.set_data()
    model, train_loss, valid_loss = trainer.train(visualize=True)
    trainer.backtest(visualize=True)