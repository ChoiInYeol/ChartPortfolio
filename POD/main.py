# main.py
import os
import yaml
import random
import numpy as np
import torch
from itertools import product
import argparse
from train import Trainer
from inference import Inference
from backtest import Backtester

def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Use manual_seed_all for multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_model_configs(base_config):
    param_lists = {
        'MODEL': base_config.get('MODELS', ['TCN']),
        'MULTIMODAL': base_config.get('MULTIMODAL_OPTIONS', [False]),
        'LOSS_FUNCTION': base_config.get('LOSS_FUNCTIONS', ['max_sharpe']),
        'TRAIN_LEN': base_config.get('TRAIN_LEN_OPTIONS', [60, 20]),
        'PRED_LEN': base_config.get('PRED_LEN_OPTIONS', [20])
    }
    
    keys, values = zip(*param_lists.items())
    for combination in product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combination)))
        yield config

def get_model_identifier(config):
    return f"{config['MODEL']}_{config['MULTIMODAL']}_{config['LOSS_FUNCTION']}_{config['TRAIN_LEN']}_{config['PRED_LEN']}"

def load_trained_models(results_file):
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return yaml.safe_load(f) or []
    return []

def save_trained_models(results_file, trained_models):
    with open(results_file, 'w') as f:
        yaml.safe_dump(trained_models, f)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run training, inference, and backtesting.')
    parser.add_argument('--train', action='store_true', help='Run the training step.')
    parser.add_argument('--inference', action='store_true', help='Run the inference step.')
    parser.add_argument('--backtest', action='store_true', help='Run the backtesting step.')
    args = parser.parse_args()

    # If no arguments are given, run all steps
    if not any([args.train, args.inference, args.backtest]):
        args.train = True
        args.inference = True
        args.backtest = True

    base_config = load_config("config/config.yaml")
    set_random_seed(base_config["SEED"])

    results_dir = base_config['RESULT_DIR']
    results_file = os.path.join(results_dir, 'results.yaml')
    trained_models = load_trained_models(results_file)

    model_configs = list(generate_model_configs(base_config))

    print("Order in which models will be processed:")
    for idx, config in enumerate(model_configs):
        print(f"{idx+1}: {get_model_identifier(config)}")

    if args.train:
        # Train models
        for config in model_configs:
            model_identifier = get_model_identifier(config)
            if model_identifier in trained_models:
                print(f"Skipping already trained model: {model_identifier}")
                continue

            result_subdir = os.path.join(results_dir, model_identifier)
            os.makedirs(result_subdir, exist_ok=True)
            config['RESULT_DIR'] = result_subdir

            print(f"Training model {model_identifier}")
            trainer = Trainer(config)
            trainer.run()
            print(f"Model {model_identifier} trained and saved.")

            trained_models.append(model_identifier)
            save_trained_models(results_file, trained_models)

    if args.inference:
        # Perform inference
        model_identifiers = []
        for config in model_configs:
            model_identifier = get_model_identifier(config)
            model_dir = os.path.join(results_dir, model_identifier, 'models')
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
                if model_files:
                    # Assuming the model with the lowest loss is desired
                    model_file = min(model_files, key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
                    model_path = os.path.join(model_dir, model_file)
                    print(f"Performing inference for model {model_identifier}")
                    config['RESULT_DIR'] = os.path.join(results_dir, model_identifier)
                    inference = Inference(config, model_path)
                    weights = inference.infer()
                    inference.save_weights(weights)
                    model_identifiers.append(model_identifier)
                else:
                    print(f"No model files found for {model_identifier}")
            else:
                print(f"Model directory not found: {model_dir}")
        # Save model_identifiers for backtesting
        with open(os.path.join(results_dir, 'model_identifiers.yaml'), 'w') as f:
            yaml.safe_dump(model_identifiers, f)

    if args.backtest:
        # Load model_identifiers if not defined
        if not args.inference:
            model_identifiers_file = os.path.join(results_dir, 'model_identifiers.yaml')
            if os.path.exists(model_identifiers_file):
                with open(model_identifiers_file, 'r') as f:
                    model_identifiers = yaml.safe_load(f) or []
            else:
                # If model_identifiers.yaml does not exist, attempt to generate model_identifiers from existing result subdirectories
                model_identifiers = []
                for config in model_configs:
                    model_identifier = get_model_identifier(config)
                    weights_file = os.path.join(results_dir, model_identifier, 'weights.pkl')
                    if os.path.exists(weights_file):
                        model_identifiers.append(model_identifier)
                if not model_identifiers:
                    print("No model identifiers found for backtesting.")
                    return
        backtester = Backtester(base_config)
        backtester.backtest(model_identifiers, visualize=True)
        print("Backtesting completed for all models.")

if __name__ == "__main__":
    main()
