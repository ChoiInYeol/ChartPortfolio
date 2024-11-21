for mode in train inference; do
    for model in GRU TCN TRANSFORMER; do
        for use_prob in Yes No; do
            echo "Running $mode with $model and use_prob=$use_prob"
            python main.py --modes $mode --models $model --use_prob $use_prob --config_path config/config.yaml
        done
    done
done