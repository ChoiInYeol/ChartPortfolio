import argparse
import yaml
import logging
import torch
from pathlib import Path
from dataload.make_dataset import prepare_dataset
from train import Trainer
from inference import ModelInference

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='시계열 모델 학습/추론')
    
    parser.add_argument('--mode', 
                       choices=['train', 'inference'],
                       required=True,
                       help='실행 모드 (train 또는 inference)')
    
    parser.add_argument('--model_type',
                       choices=['GRU', 'TCN', 'TRANSFORMER'],
                       required=True,
                       help='모델 타입')
    
    parser.add_argument('--use_prob',
                       type=lambda x: x.lower() == 'true',
                       required=True,
                       help='확률 기반 모델 사용 여부 (true/false)')
    
    parser.add_argument('--config',
                       type=str,
                       default='config/base_config.yaml',
                       help='설정 파일 경로')
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    # CLI 인자 파싱
    args = parse_arguments()
        
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # CLI 입력으로 config 값 업데이트
    config['MODEL']['TYPE'] = args.model_type
    config['MODEL']['USEPROB'] = args.use_prob
    
    setup_logging()
    logging.info(f"Running with model {config['MODEL']['TYPE']} (useprob={config['MODEL']['USEPROB']})")
    
    # GPU 병렬 처리 설정
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        config['TRAINING']['USE_DATA_PARALLEL'] = True
    else:
        config['TRAINING']['USE_DATA_PARALLEL'] = False
    
    try:
        if args.mode == 'train':
            # 1. 모델 학습
            logging.info("Starting training...")
            trainer = Trainer(config)
            trainer.train()
            
            # 2. 각 데이터셋에 대한 가중치 예측 및 저장
            logging.info("Generating predictions for all datasets...")
            
            # Train 데이터
            logging.info("Processing training data...")
            train_weights = trainer.predict(data_type='train')
            trainer.save_weights(train_weights, data_type='train')
            
            # Validation 데이터
            logging.info("Processing validation data...")
            val_weights = trainer.predict(data_type='val')
            trainer.save_weights(val_weights, data_type='val')
            
        else:  # inference 모드
            # Test 데이터에 대한 추론 수행
            logging.info("Starting inference for test data...")
            inference = ModelInference(config)
            test_weights = inference.predict()
            logging.info("Test data inference completed and weights saved")
            
        logging.info("All processing completed")
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    # python main.py --mode train --model_type GRU --use_prob true --config config/base_config.yaml