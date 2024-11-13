#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="logs/training"
mkdir -p $LOG_DIR

# 현재 시간을 파일명에 사용
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# GPU 개수 확인
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs" | tee -a $LOG_FILE

# 작업 큐 생성을 위한 임시 파일
QUEUE_FILE="job_queue_${TIMESTAMP}.txt"

# 학습 설정 생성 함수
generate_jobs() {
    local queue_file=$1
    
    # 모든 파라미터 조합 생성
    for model in "GRU" "TCN" "TRANSFORMER"; do
        for n_select in 30 50 100; do
            for use_prob in true false; do
                for hidden_dim in 128 256; do
                    for n_layer in 1 2; do
                        for dropout in 0.3 0.5; do
                            echo "${model},${n_select},${use_prob},${hidden_dim},${n_layer},${dropout}" >> $queue_file
                        done
                    done
                done
            done
        done
    done
    
    # 작업 수 출력
    local total_jobs=$(wc -l < $queue_file)
    echo "Generated $total_jobs jobs" | tee -a $LOG_FILE
}

# 단일 GPU에서 학습 실행
run_training() {
    local gpu_id=$1
    local model=$2
    local n_select=$3
    local use_prob=$4
    local hidden_dim=$5
    local n_layer=$6
    local dropout=$7
    
    # GPU 설정
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    echo "[GPU $gpu_id] Starting training: Model=$model, N_SELECT=$n_select, USE_PROB=$use_prob" | tee -a $LOG_FILE
    
    # config.yaml 임시 복사본 생성
    local config_copy="config/config_gpu${gpu_id}.yaml"
    cp config/config.yaml $config_copy
    
    # 설정 파일 수정
    sed -i "s/MODEL: .*/MODEL: '$model'/" $config_copy
    sed -i "s/N_SELECT: .*/N_SELECT: $n_select/" $config_copy
    sed -i "s/HIDDEN_DIM: .*/HIDDEN_DIM: $hidden_dim/" $config_copy
    sed -i "s/N_LAYER: .*/N_LAYER: $n_layer/" $config_copy
    sed -i "s/DROPOUT: .*/DROPOUT: $dropout/" $config_copy
    
    # 모델별 특수 파라미터 설정
    if [ "$model" = "TCN" ]; then
        sed -i "s/hidden_size: .*/hidden_size: $hidden_dim/" $config_copy
        sed -i "s/n_dropout: .*/n_dropout: $dropout/" $config_copy
    fi
    
    if [ "$model" = "TRANSFORMER" ]; then
        sed -i "s/n_layer: .*/n_layer: $n_layer/" $config_copy
        sed -i "s/n_dropout: .*/n_dropout: $dropout/" $config_copy
    fi
    
    # Python 스크립트 실행
    if [ "$use_prob" = true ]; then
        python main.py --config $config_copy --modes train --models $model --options use_prob >> "${LOG_FILE}.gpu${gpu_id}" 2>&1
    else
        python main.py --config $config_copy --modes train --models $model >> "${LOG_FILE}.gpu${gpu_id}" 2>&1
    fi
    
    # 임시 설정 파일 삭제
    rm $config_copy
    
    echo "[GPU $gpu_id] Completed training: $model with N_SELECT=$n_select" | tee -a $LOG_FILE
}

# 작업 큐 생성
generate_jobs $QUEUE_FILE

# 병렬 처리를 위한 작업 분배
process_queue() {
    local gpu_id=$1
    
    while true; do
        # 다음 작업 가져오기 (atomic operation을 위해 임시 파일 사용)
        local job_file="${QUEUE_FILE}.processing"
        if ! mv "$QUEUE_FILE" "$job_file" 2>/dev/null; then
            # 더 이상 작업이 없으면 종료
            break
        fi
        
        # 첫 번째 라인 읽기
        local next_job=$(head -n 1 "$job_file")
        
        # 나머지 라인들을 다시 큐에 저장
        tail -n +2 "$job_file" > "$QUEUE_FILE"
        rm "$job_file"
        
        # 작업이 없으면 종료
        if [ -z "$next_job" ]; then
            break
        fi
        
        # 작업 파라미터 파싱
        IFS=',' read -r model n_select use_prob hidden_dim n_layer dropout <<< "$next_job"
        
        # 학습 실행
        run_training $gpu_id "$model" "$n_select" "$use_prob" "$hidden_dim" "$n_layer" "$dropout"
    done
}

# GPU 개수만큼 병렬 처리 시작
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    process_queue $gpu &
done

# 모든 프로세스 완료 대기
wait

# 임시 파일 정리
rm -f $QUEUE_FILE

echo "All training jobs completed!" | tee -a $LOG_FILE 