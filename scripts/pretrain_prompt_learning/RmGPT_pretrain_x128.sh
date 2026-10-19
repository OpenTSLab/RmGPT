model_name=RmGPT
exp_name=RmGPT_pretrain_x512
wandb_mode=${WANDB_MODE:-disabled}
ptune_name=prompt_tuning
# export CUDA_LAUNCH_BLOCKING=1
d_model=512
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
# random_port=$((RANDOM % 900 + 10000))

# # Pretrain

# 分布式启动方法
# torchrun --nnodes 1 --nproc-per-node 7 --master_port $random_port run_pretrain.py \
#   --is_training 1 \
#   --model_id $exp_name \
#   --model $model_name \
#   --prompt_num 10 \
#   --patch_len 256 \
#   --stride 256 \
#   --e_layers 3 \
#   --d_model $d_model \
#   --des 'Exp' \
#   --acc_it 128 \
#   --batch_size 32 \
#   --learning_rate 5e-5 \
#   --min_lr 1e-4 \
#   --weight_decay 5e-6\
#   --train_epochs 150 \
#   --warmup_epochs 0 \
#   --min_keep_ratio 0.5 \
#   --right_prob 0.5 \
#   --min_mask_ratio 0.7 \
#   --max_mask_ratio 0.8 \
#   --debug $wandb_mode \
#   --task_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml
#   # --min_keep_ratio 0.1 \


# 单机卡启动方法
python run_pretrain.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 1 \
  --batch_size 1024 \
  --learning_rate 5e-9 \
  --min_lr 1e-4 \
  --weight_decay 5e-10\
  --train_epochs 10 \
  --warmup_epochs 0 \
  --min_keep_ratio 0.5 \
  --right_prob 0.5 \
  --min_mask_ratio 0.7 \
  --max_mask_ratio 0.8 \
  --debug $wandb_mode \
  --task_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml\
  --min_keep_ratio 0.1 \
  --device 'cuda:0'

# Prompt tuning
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
  --prompt_tune_epoch 10\
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/main_result/multi_task.yaml \
  --pretrained_weight 'checkpoints/Base_RmGPT2_pretrain_x512_RmGPT_All_ftM_dm512_el4_Exp_0/pretrain_checkpoint.pth' \
  --batch_size 256 \

  # --device cuda:0
  # --learning_rate 0.0001 \
  # --acc_it 32 \
  # --clip_grad 100 \
