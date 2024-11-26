import argparse
import yaml
import logging
import torch
from pathlib import Path
from dataload.make_dataset import prepare_dataset
from train import Trainer
from inference import ModelInference

def parse_args():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization with Deep Learning')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='inference',
                       help='실행 모드 (train 또는 inference)')
    parser.add_argument('--model_path', type=str,
                       help='추론 시 사용할 모델 가중치 경로 (inference 모드에서만 사용)')
    parser.add_argument('--model_type', type=str, choices=['GRU', 'TCN', 'TRANSFORMER'],
                       help='사용할 모델 타입')
    parser.add_argument('--use_prob', type=str, choices=['true', 'false'],
                       help='확률 기반 모델 사용 여부')
    return parser.parse_args()

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """메인 실행 함수"""
    # 인자 파싱 및 설정 로드
    args = parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 커맨드라인 인자로 config 값 업데이트
    if args.model_type:
        config['MODEL']['TYPE'] = args.model_type
    if args.use_prob:
        config['MODEL']['USE_PROB'] = args.use_prob.lower() == 'true'
    
    setup_logging()
    logging.info(f"Running in {args.mode} mode with model {config['MODEL']['TYPE']} (use_prob={config['MODEL']['USE_PROB']})")
    
    # GPU 병렬 처리 설정
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        config['TRAINING']['USE_DATA_PARALLEL'] = True
    else:
        config['TRAINING']['USE_DATA_PARALLEL'] = False
    
    try:
        # 1. 데이터셋 준비 (선택)
        # logging.info("Preparing dataset...")
        # prepare_dataset(config)
        
        if args.mode == 'train':
            # 2. 모델 학습
            logging.info("Starting training...")
            trainer = Trainer(config)
            trainer.train()
            
            # 학습 데이터에 대한 가중치 예측 및 저장
            train_weights = trainer.predict(data_type='train')
            trainer.save_weights(train_weights, data_type='train')
            
            # 검증 데이터에 대한 가중치 예측 및 저장
            val_weights = trainer.predict(data_type='val')
            trainer.save_weights(val_weights, data_type='val')
            
            logging.info("Training completed")
            
        elif args.mode == 'inference':
            if not args.model_path:
                raise ValueError("Model path must be provided for inference mode")
            
            # 3. 추론 실행
            logging.info("Starting inference...")
            inference = ModelInference(
                config=config,
                model_path=args.model_path
            )
            weights = inference.predict()
            logging.info("Inference completed")
            
            # 4. 결과 저장
            inference.save_weights(weights)
            logging.info("Results saved")
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()