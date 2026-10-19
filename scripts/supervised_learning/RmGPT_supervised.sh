model_name=RmGPT
exp_name=RmGPT_pretrain_x512
wandb_mode=${WANDB_MODE:-disabled}
ptune_name=prompt_tuning
d_model=512

# Supervised learning
python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --prompt_tune_epoch 20\
  --train_epochs 0 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/main_result/multi_task.yaml \
  --pretrained_weight 'checkpoints/ALL_task_test_RmGPT_All_ftM_dm512_el4_test_0/pretrain_checkpoint.pth'\
  --learning_rate 3e-4\
  --batch_size 256 \
