import argparse
import yaml
import logging
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
    
    setup_logging()
    logging.info(f"Running in {args.mode} mode")
    
    try:
        # 1. 데이터셋 준비 (선택)
        # logging.info("Preparing dataset...")
        # prepare_dataset(config)
        
        if args.mode == 'train':
            # 2. 모델 학습
            logging.info("Starting training...")
            trainer = Trainer(config)
            trainer.train()
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