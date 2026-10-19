model_name=RmGPT
wandb_mode=disabled
project_name=fewshot
exp_name=fewshot_newdata_finetune_pct05
# random_port=$((RANDOM % 9000 + 1000))
# Path to the supervised checkpoint
# get supervised checkpoint: scripts/supervised/RmGPT_supervised_x64.sh
ckpt_path=checkpoints/ALL_task_test_RmGPT_All_ftM_dm512_el4_test_0/pretrain_checkpoint.pth
python run.py \
  --is_training 1 \
  --fix_seed 2021 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 5 \
  --learning_rate 1e-6 \
  --weight_decay 1e-7\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_train.yaml

#   # --subsample_pct 0.05 \
