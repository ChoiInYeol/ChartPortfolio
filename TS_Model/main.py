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

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    if config_path.endswith('.json'):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

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

if __name__ == "__main__":
    try:
        # 사용자 선택 받기
        selections = get_user_selections()
        
        # 설정 파일 로드
        config = load_config(selections['config_path'])
        
        # 결과 디렉토리 생성
        os.makedirs(config['RESULT_DIR'], exist_ok=True)
        
        # 로그 파일 설정
        log_file = os.path.join(config['RESULT_DIR'], 'main.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Using config file: {selections['config_path']}")
        
        # 시드 고정
        set_seed(config["SEED"])
        logger.info(f"Random seed set to {config['SEED']}")
        
        # 선택 사항 로깅
        logger.info("User selections:")
        logger.info(f"- Modes: {selections['modes']}")
        logger.info(f"- Models: {selections['models']}")
        logger.info(f"- Dataset: {selections['dataset']}")
        logger.info(f"- Use probability data: {selections['use_prob']}")
        
        # 실행
        work(config, selections)
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise