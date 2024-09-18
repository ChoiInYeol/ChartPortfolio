# main.py
import os
import json
import random
import numpy as np
import torch
import argparse
from train import Trainer
from backtest import Backtester

def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('--train', type=str, choices=['True', 'False'], default='True',
                        help='Whether to train the models (default: True)')
    parser.add_argument('--visualize', type=str, choices=['True', 'False'], default='True',
                        help='Whether to visualize the results (default: True)')

    args = parser.parse_args()
    train_flag = args.train == 'True'
    visualize_flag = args.visualize == 'True'

    base_config = json.load(open("config/train_config.json", "r"))
    set_random_seed(base_config["SEED"])

    # List of models and multimodal options to experiment with
    models = ['GRU', 'TCN', 'TRANSFORMER']
    multimodal_options = [True, False]
    model_paths = {}

    # Train all models and save their parameters
    if train_flag:
        for model_name in models:
            for multimodal in multimodal_options:
                config = base_config.copy()
                config["MODEL"] = model_name
                config["MULTIMODAL"] = multimodal
                print(f"Training model {config['MODEL']}, multimodal={config['MULTIMODAL']}")
                trainer = Trainer(config)
                trainer.run()
                model_paths[(model_name, multimodal)] = trainer.get_best_model_path()
                print(f"Model {config['MODEL']} trained and saved.")

    # Run backtests using the saved model parameters
    for model_name in models:
        for multimodal in multimodal_options:
            config = base_config.copy()
            config["MODEL"] = model_name
            config["MULTIMODAL"] = multimodal
            print(f"Backtesting model {config['MODEL']}, multimodal={config['MULTIMODAL']}")
            backtester = Backtester(config)
            model_path = model_paths.get((model_name, multimodal))
            if model_path:
                backtester.backtest(model_path, visualize=visualize_flag)
                print(f"Backtest for model {config['MODEL']} completed.")
            else:
                print(f"No model found for {config['MODEL']} with multimodal={config['MULTIMODAL']}")