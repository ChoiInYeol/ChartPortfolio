#!/bin/bash

CONFIG_PATH="config/base_config.yaml"

# 각 모델 타입에 대해 실행
for MODEL in "GRU" "TCN" "TRANSFORMER"; do
    # 확률 사용 여부에 대해 실행
    for USE_PROB in "true" ; do
        echo "Processing ${MODEL} with use_prob=${USE_PROB}"
        python main.py \
            --config ${CONFIG_PATH} \
            --model_type ${MODEL} \
            --use_prob ${USE_PROB}
    done
done