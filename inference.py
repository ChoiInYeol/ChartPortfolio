# inference.py
import os
import torch
import numpy as np
import pickle
import logging
from model.multimodal import Multimodal
from model.gru import GRU
from model.transformer import Transformer
from model.tcn import TCN
from typing import Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class Inference:
    def __init__(self, config: Dict[str, Any], model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda" if config["USE_CUDA"] and torch.cuda.is_available() else "cpu")
        self.model_name = config["MODEL"].lower()
        self.multimodal = config.get("MULTIMODAL", False)
        self.len_train = config['TRAIN_LEN']
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_FEAT']
        self.lb = config['LB']
        self.ub = config['UB']
        self.use_top_m_assets = config.get('USE_TOP_M_ASSETS', False)
        self.top_m = config.get('TOP_M', 0)

        # Initialize logging to file
        model_identifier = f"{self.model_name}_{self.multimodal}_{self.config['LOSS_FUNCTION']}_{self.len_train}_{self.len_pred}"
        log_filename = os.path.join(config['RESULT_DIR'], f"inference_{model_identifier}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        logger.info(f"Configuration: {self.config}")
        logger.info(f"Model Path: {self.model_path}")

        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        common_params = {
            'n_stocks': self.n_stock,
            'lb': self.lb,
            'ub': self.ub,
            'multimodal': self.multimodal
        }
        if self.multimodal:
            img_height, img_width = self._get_image_dimensions()
            model = Multimodal(
                model_type=self.model_name,
                model_params={**self.config[self.model_name.upper()], **common_params},
                img_height=img_height,
                img_width=img_width,
                lb=self.lb,
                ub=self.ub
            ).to(self.device)
        elif self.model_name == "gru":
            gru_config = self.config['GRU']
            model = GRU(
                n_layers=gru_config['n_layers'],
                hidden_dim=gru_config['hidden_dim'],
                dropout_p=gru_config['dropout_p'],
                bidirectional=gru_config['bidirectional'],
                **common_params
            ).to(self.device)
        elif self.model_name == "transformer":
            transformer_config = self.config['TRANSFORMER']
            model = Transformer(
                n_timestep=transformer_config['n_timestep'],
                n_layer=transformer_config['n_layer'],
                n_head=transformer_config['n_head'],
                n_dropout=transformer_config['n_dropout'],
                n_output=transformer_config['n_output'],
                **common_params
            ).to(self.device)
        elif self.model_name == "tcn":
            tcn_config = self.config['TCN']
            model = TCN(
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

        # Load model weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

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

    def _load_test_data(self):
        logger.info("Loading test data...")

        with open("data/dataset.pkl", "rb") as f:
            _, _, _, _, test_x_raw, _ = pickle.load(f)

        if self.multimodal:
            with open("data/dataset_img.pkl", "rb") as f:
                _, _, test_img_data, _ = pickle.load(f)
            test_img = test_img_data['images']
        else:
            test_img = None

        scale = self.len_pred
        test_x = test_x_raw * scale

        self.test_x = torch.from_numpy(test_x).float().to(self.device)
        self.test_img = torch.from_numpy(test_img).float().to(self.device) if test_img is not None else None

    def infer(self):
        self._load_test_data()
        weights_list = []

        with torch.no_grad():
            for i in range(len(self.test_x)):
                x = self.test_x[i].unsqueeze(0)  # Add batch dimension
                if self.multimodal:
                    img = self.test_img[i].unsqueeze(0)
                    outputs = self.model(x, img)
                    portfolio_weights = outputs[0].cpu().numpy()
                else:
                    outputs = self.model(x)
                    portfolio_weights = outputs.cpu().numpy()
                weights_list.append(portfolio_weights.flatten())

        weights_array = np.array(weights_list)
        return weights_array

    def save_weights(self, weights_array):
        # Save weights to the model's RESULT_DIR
        weights_path = os.path.join(self.config['RESULT_DIR'], f"weights_{get_model_identifier(self.config)}.npy")
        np.save(weights_path, weights_array)
        logger.info(f"Weights saved to {weights_path}")

def get_model_identifier(config):
    return f"{config['MODEL']}_{config['MULTIMODAL']}_{config['LOSS_FUNCTION']}_{config['TRAIN_LEN']}_{config['PRED_LEN']}"