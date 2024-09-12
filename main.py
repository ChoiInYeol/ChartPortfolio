import os
import json
import random
import numpy as np
import torch
import argparse
from train.train import Trainer

"""
훈련 없이 백테스트 하기: python main.py --train False
시각화 없이 백테스트 하기: python main.py --visualize False
TCN 모델 사용하기: python main.py --model TCN
다중 모달 모델 사용하기: python main.py --multimodal True
훈련 없이 백테스트 하고 시각화 없이 하기: python main.py --train False --visualize False
"""

    

def work(config, train=True, visualize=False):
    worker = Trainer(config)
    worker.set_data()
    if train:
        worker.train(visualize)
    worker.backtest(model_file='result/best_model_weight_TCN_18.pt', visualize=visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('--train', type=str, choices=['True', 'False'], default='True',
                        help='Whether to train the model (default: True)')
    parser.add_argument('--visualize', type=str, choices=['True', 'False'], default='True',
                        help='Whether to visualize the results (default: True)')
    parser.add_argument('--model', type=str, choices=['GRU', 'TCN', 'TRANSFORMER'], 
                        help='Model to use (GRU, TCN, or TRANSFORMER)')
    parser.add_argument('--multimodal', type=str, choices=['True', 'False'],
                        help='Whether to use multimodal approach')

    args = parser.parse_args()

    config = json.load(open("config/train_config.json", "r"))

    # Update config based on command line arguments
    if args.model:
        config["MODEL"] = args.model
    if args.multimodal:
        config["MULTIMODAL"] = args.multimodal == 'True'

    os.environ["PYTHONHASHSEED"] = str(config["SEED"])
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    work(config, train=args.train == 'False', visualize=args.visualize == 'True')