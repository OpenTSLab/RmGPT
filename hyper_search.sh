#!/bin/bash

# Hyperparameter search via a W&B sweep. Set these before launching:
#   W&B_ENTITY / W&B_PROJECT: your wandb target
#   SWEEP_ID: id created with "wandb sweep sweep-config.yaml"
#   GPU_IDS: physical GPUs to use, one tmux window each
COMMAND="wandb agent ${W&B_ENTITY:-your-entity}/${W&B_PROJECT:-your-project}/${SWEEP_ID:?set SWEEP_ID first}"

GPU_IDS=(${GPU_IDS:-0 1 2 3})

# Start tmux session
tmux new-session -d -s hyperparam_search-0

idx=0
for gpu in "${GPU_IDS[@]}"; do
    tmux new-window -t hyperparam_search-0: -n "Hyperparam_Search_$idx"
    tmux send-keys -t "Hyperparam_Search_$idx" "source ~/.bashrc && conda activate time_series && CUDA_VISIBLE_DEVICES=$gpu $COMMAND" Enter
    idx=$((idx + 1))
done
