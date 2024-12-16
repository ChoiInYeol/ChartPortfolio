#!/bin/bash

# 모델 타입 목록
MODEL_TYPES=("TRANSFORMER")

# 확률 사용 여부
USE_PROBS=("true" "false")

# 설정 파일 경로
CONFIG_PATH="config/base_config.yaml"

# 각 조합에 대해 학습 및 추론 실행
for MODEL in "${MODEL_TYPES[@]}"; do
    for USE_PROB in "${USE_PROBS[@]}"; do
        echo "Running training for ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --mode train \
            --model_type ${MODEL} \
            --use_prob ${USE_PROB} \
            --config ${CONFIG_PATH}
        
        echo "Running inference for ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --mode inference \
            --model_type ${MODEL} \
            --use_prob ${USE_PROB} \
            --config ${CONFIG_PATH}
    done
done