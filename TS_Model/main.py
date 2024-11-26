import os
import json
import random
import numpy as np
import torch
import yaml
import logging
import inquirer
from train import Trainer
from inference import Inference
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from model import PortfolioGRU, PortfolioGRUWithProb, PortfolioTCN, PortfolioTCNWithProb, PortfolioTransformer, PortfolioTransformerWithProb

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def set_seed(seed: int):
    """시드 고정"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment_config() -> Dict:
    """실험 설정을 위한 사용자 입력을 받습니다."""
    questions = [
        inquirer.List('model',
                     message="Select model type",
                     choices=['GRU', 'TCN', 'TRANSFORMER'],
                     default='GRU'),
        
        inquirer.List('objective',
                     message="Select portfolio objective",
                     choices=['mean_variance', 'minimum_variance', 'sharpe_ratio'],
                     default='sharpe_ratio'),
        
        # 포트폴리오 제약조건 설정
        inquirer.Confirm('long_only',
                        message="Use long-only constraint?",
                        default=True),
        
        inquirer.Text('max_position',
                     message="Maximum position limit (0-1, enter for none)",
                     default=''),
        
        inquirer.Text('cardinality',
                     message="Cardinality constraint (enter for none)",
                     default=''),
        
        inquirer.Text('leverage',
                     message="Leverage constraint (default: 1.0)",
                     default='1.0'),
        
        # 학습 파라미터 설정
        inquirer.Text('risk_aversion',
                     message="Risk aversion parameter (for mean-variance)",
                     default='1.0'),
        
        inquirer.Text('learning_rate',
                     message="Learning rate",
                     default='0.001'),
        
        inquirer.Confirm('use_validation',
                        message="Use validation set?",
                        default=True),
    ]
    
    answers = inquirer.prompt(questions)
    
    # 숫자형 변환
    if answers['max_position']:
        answers['max_position'] = float(answers['max_position'])
    if answers['cardinality']:
        answers['cardinality'] = int(answers['cardinality'])
    answers['leverage'] = float(answers['leverage'])
    answers['risk_aversion'] = float(answers['risk_aversion'])
    answers['learning_rate'] = float(answers['learning_rate'])
    
    return answers

def get_available_datasets() -> dict:
    """
    data 폴더에서 사용 가능한 데이터셋을 검색합니다.
    
    Returns:
        dict: {dataset_name: dataset_info} 형태의 딕셔너리
    """
    current_path = Path(__file__).resolve()
    data_dir = current_path.parent / 'data'
    
    datasets = {}
    dataset_file = data_dir / 'dataset.pkl'
    dates_file = data_dir / 'dates.pkl'
    
    if dataset_file.exists() and dates_file.exists():
        key = 'default'
        datasets[key] = {
            'filename': 'dataset.pkl'
        }
    
    return datasets

def get_user_selections():
    """사용자 선택 받기"""
    # 사용 가능한 데이터셋 검색
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        logger.error("사용 가능한 데이터셋이 없습니다. make_dataset.py를 먼저 실행하세요.")
        exit(1)
    
    # 데이터셋 선택지 생성
    dataset_choices = ['default']
    
    questions = [
        inquirer.Checkbox(
            'modes',
            message='실행할 모드를 선택하세요 (스페이스바로 선택)',
            choices=['train', 'inference'],
            default=['train']
        ),
        inquirer.Checkbox(
            'models',
            message='사용할 모델을 선택하세요 (스페이스바로 선택)',
            choices=['GRU', 'TCN', 'TRANSFORMER'],
            default=['GRU']
        ),
        inquirer.List(
            'dataset',
            message='사용할 데이터셋을 선택하세요',
            choices=dataset_choices,
            default=dataset_choices[0] if dataset_choices else None
        ),
        inquirer.List(
            'use_prob',
            message='상승확률 데이터를 사용하시겠습니까?',
            choices=['Yes', 'No'],
            default='No'
        ),
        inquirer.Path(
            'config_path',
            message='설정 파일 경로를 입력하세요',
            default='config/config.yaml',
            exists=True
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers:
        logger.error("사용자가 선택을 취소했습니다.")
        exit(1)
        
    if not answers['modes']:
        logger.error("최소한 하나의 모드를 선택해야 합니다.")
        exit(1)
        
    if not answers['models']:
        logger.error("최소한 하나의 모델을 선택해야 합니다.")
        exit(1)
    
    # 선택된 데이터셋 정보 추출
    selected_dataset = answers['dataset'].split(' (')[0]
    dataset_info = available_datasets[selected_dataset]
    
    # answers 딕셔너리에 데이터셋 정보 추가
    answers['dataset_info'] = dataset_info
    answers['use_prob'] = (answers['use_prob'] == 'Yes')
    
    return answers

def work(config, selections):
    """
    학습, 추론 및 백테스트 실행
    
    Args:
        config (dict): 설정 딕셔너리
        selections (dict): 사용자 선택 옵션
    """
    try:
        modes = selections['modes']
        model_types = selections['models']
        use_prob = selections['use_prob']
        dataset_info = selections['dataset_info']
        
        # 데이터셋 파일 경로 설정
        current_path = Path(__file__).resolve()
        data_dir = current_path.parent / 'data'
        
        dataset_path = data_dir / dataset_info['filename']
        dates_path = data_dir / dataset_info['filename'].replace('dataset', 'dates')
        
        # 설정 파일에 데이터셋 정보 추가
        config.update({
            'DATASET_PATH': str(dataset_path),
            'DATES_PATH': str(dates_path)
        })
        
        for mode in modes:
            for model_type in model_types:
                config["MODEL"] = model_type
                logger.info(f"\nExecuting {mode} for {model_type}")
                logger.info(f"Dataset: {dataset_info['filename']}")
                logger.info(f"Use probability: {use_prob}")
                
                if mode == 'train':
                    worker = Trainer(config, use_prob=use_prob)
                    worker.set_data()
                    logger.info("Starting training...")
                    worker.train()
                    logger.info("Training completed")
                
                elif mode == 'inference':
                    model_path = os.path.join(config['RESULT_DIR'], model_type)
                    inference_worker = Inference(config, model_path, use_prob=use_prob)
                    inference_worker.set_data()
                    logger.info("Starting inference...")
                    weights = inference_worker.infer()
                    inference_worker.save_weights(weights)
                    logger.info("Inference completed")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """설정에 따라 모델을 생성합니다."""
    model_type = config['model']['type']
    use_prob = config['model'].get('use_prob', False)
    
    if model_type == 'gru':
        if use_prob:
            model = PortfolioGRUWithProb(
                n_layers=config['model']['n_layers'],
                hidden_dim=config['model']['hidden_dim'],
                n_stocks=config['data']['n_stocks'],
                dropout_p=config['model']['dropout'],
                bidirectional=config['model'].get('bidirectional', False),
                constraints=config['portfolio_constraints']
            )
        else:
            model = PortfolioGRU(
                n_layers=config['model']['n_layers'],
                hidden_dim=config['model']['hidden_dim'],
                n_stocks=config['data']['n_stocks'],
                dropout_p=config['model']['dropout'],
                bidirectional=config['model'].get('bidirectional', False),
                constraints=config['portfolio_constraints']
            )
    
    elif model_type == 'tcn':
        if use_prob:
            model = PortfolioTCNWithProb(
                n_feature=config['data']['n_features'],
                n_output=config['data']['n_stocks'],
                num_channels=config['model']['num_channels'],
                kernel_size=config['model']['kernel_size'],
                n_dropout=config['model']['dropout'],
                n_timestep=config['data']['seq_len'],
                constraints=config['portfolio_constraints']
            )
        else:
            model = PortfolioTCN(
                n_feature=config['data']['n_features'],
                n_output=config['data']['n_stocks'],
                num_channels=config['model']['num_channels'],
                kernel_size=config['model']['kernel_size'],
                n_dropout=config['model']['dropout'],
                n_timestep=config['data']['seq_len'],
                constraints=config['portfolio_constraints']
            )
    
    elif model_type == 'transformer':
        if use_prob:
            model = PortfolioTransformerWithProb(
                n_feature=config['data']['n_features'],
                n_timestep=config['data']['seq_len'],
                n_layer=config['model']['n_layers'],
                n_head=config['model']['n_heads'],
                n_dropout=config['model']['dropout'],
                n_output=config['data']['n_stocks'],
                constraints=config['portfolio_constraints']
            )
        else:
            model = PortfolioTransformer(
                n_feature=config['data']['n_features'],
                n_timestep=config['data']['seq_len'],
                n_layer=config['model']['n_layers'],
                n_head=config['model']['n_heads'],
                n_dropout=config['model']['dropout'],
                n_output=config['data']['n_stocks'],
                constraints=config['portfolio_constraints']
            )
    
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")
    
    return model.to(device)

def main():
    """메인 실행 함수"""
    # 설정 로드
    config = load_config()
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(config)
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config, device)
    
    # 옵티마이저와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = PortfolioLoss(config['training']['loss_weights'])
    
    # Trainer 생성 및 학습 실행
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_prob=config['model'].get('use_prob', False)
    )
    
    # 학습 수행
    trainer.set_data()
    model, history = trainer.train()
    
    # 학습 결과 저장
    model_path = result_dir / f"model_{config['objective']}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, model_path)
    
    # 추론 수행
    inferencer = Inference(config, model_path)
    weights = inferencer.infer()
    
    # 가중치 저장
    inferencer.save_weights(weights)
    
    # 포트폴리오 성과 분석
    metrics = inferencer.calculate_portfolio_metrics(weights)
    logging.info("Portfolio Metrics:")
    logging.info(metrics)
    
    # 시각화
    visualizer = PortfolioVisualizer()
    
    # 포트폴리오 가중치 시각화
    visualizer.plot_weight_evolution(
        weights,
        result_dir,
        f"{config['model']}_{config['objective']}"
    )
    
    # 포트폴리오 성과 시각화
    visualizer.plot_portfolio_performance(
        weights,
        result_dir,
        f"{config['model']}_{config['objective']}"
    )

if __name__ == "__main__":
    main()