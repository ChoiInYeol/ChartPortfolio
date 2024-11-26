#!/bin/bash

CONFIG_PATH="config/base_config.yaml"

# 각 모델 타입에 대해 실행
for MODEL in "GRU" "TCN" "TRANSFORMER"; do
    # 확률 사용 여부에 대해 실행
    for USE_PROB in "true" "false"; do
        # 학습 모드
        echo "Training ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --mode train \
            --config_path ${CONFIG_PATH} \
            --model_type ${MODEL} \
            --use_prob ${USE_PROB}
        
        # 추론 모드
        echo "Inferencing ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --mode inference \
            --config_path ${CONFIG_PATH} \
            --model_type ${MODEL} \
            --use_prob ${USE_PROB}
    done
done