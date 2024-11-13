#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="logs/training"
mkdir -p $LOG_DIR

# 현재 시간을 파일명에 사용
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# 학습 큐 실행 함수
run_training() {
    local model=$1
    local n_select=$2
    local use_prob=$3
    local hidden_dim=$4
    local n_layer=$5
    local dropout=$6
    
    echo "Starting training: Model=$model, N_SELECT=$n_select, USE_PROB=$use_prob, HIDDEN_DIM=$hidden_dim, N_LAYER=$n_layer, DROPOUT=$dropout" | tee -a $LOG_FILE
    
    # config.yaml 임시 수정
    sed -i "s/MODEL: .*/MODEL: '$model'/" config/config.yaml
    sed -i "s/N_SELECT: .*/N_SELECT: $n_select/" config/config.yaml
    sed -i "s/HIDDEN_DIM: .*/HIDDEN_DIM: $hidden_dim/" config/config.yaml
    sed -i "s/N_LAYER: .*/N_LAYER: $n_layer/" config/config.yaml
    sed -i "s/DROPOUT: .*/DROPOUT: $dropout/" config/config.yaml
    
    # TCN 특정 파라미터 수정
    if [ "$model" = "TCN" ]; then
        sed -i "s/hidden_size: .*/hidden_size: $hidden_dim/" config/config.yaml
        sed -i "s/n_dropout: .*/n_dropout: $dropout/" config/config.yaml
    fi
    
    # Transformer 특정 파라미터 수정
    if [ "$model" = "TRANSFORMER" ]; then
        sed -i "s/n_layer: .*/n_layer: $n_layer/" config/config.yaml
        sed -i "s/n_dropout: .*/n_dropout: $dropout/" config/config.yaml
    fi
    
    # Python 스크립트 실행
    if [ "$use_prob" = true ]; then
        python main.py --modes train --models $model --options use_prob >> $LOG_FILE 2>&1
    else
        python main.py --modes train --models $model >> $LOG_FILE 2>&1
    fi
    
    echo "Completed training: $model with N_SELECT=$n_select" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
}

# 학습 큐 설정
declare -a models=("GRU" "TCN" "TRANSFORMER")
declare -a n_selects=(30 50 100)
declare -a use_probs=(true false)
declare -a hidden_dims=(128 256)
declare -a n_layers=(1 2)
declare -a dropouts=(0.3 0.5)

# 모든 조합에 대해 순차적으로 학습 실행
for model in "${models[@]}"; do
    for n_select in "${n_selects[@]}"; do
        for use_prob in "${use_probs[@]}"; do
            for hidden_dim in "${hidden_dims[@]}"; do
                for n_layer in "${n_layers[@]}"; do
                    for dropout in "${dropouts[@]}"; do
                        run_training "$model" "$n_select" "$use_prob" "$hidden_dim" "$n_layer" "$dropout"
                    done
                done
            done
        done
    done
done

echo "All training jobs completed!" | tee -a $LOG_FILE 