#!/bin/bash

CONFIG_PATH="config/base_config.yaml"

# 각 모델 타입에 대해 실행
for MODEL in "GRU" "TCN" "TRANSFORMER"; do
    # 확률 사용 여부에 대해 실행
    for USE_PROB in "true" "false"; do
        # 학습 모드
        echo "Training ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --config ${CONFIG_PATH} \
            --mode train
        
        # 추론 모드 
        echo "Inferencing ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --config ${CONFIG_PATH} \
            --mode inference \
            --model_path "models/${MODEL}_latest.pth"
    done
done