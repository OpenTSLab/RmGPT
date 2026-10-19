model_name=RmGPT
wandb_mode=disabled
project_name=fewshot
exp_name=fewshot_newdata_finetune_pct05
# random_port=$((RANDOM % 9000 + 1000))
# Path to the supervised checkpoint
# get supervised checkpoint: scripts/supervised/RmGPT_supervised_x64.sh
ckpt_path=checkpoints/few_shot_without_CWRU/checkpoint.pth
# ckpt_path=checkpoints/ALL_task_fewshot_newdata_finetune_pct05_RmGPT_All_ftM_dm512_el4_Exp_0/checkpoint.pth
python run.py \
  --is_training 1 \
  --device 'cuda:0' \
  --fix_seed 2021 \
  --batch_size 16\
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 0 \
  --prompt_tune_epoch 20\
  --learning_rate 1e-4 \
  --weight_decay 1e-5\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --fix_seed 2024 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_interface.yaml \
  --subsample_pct 1-shot\

python run.py \
  --is_training 1 \
  --device 'cuda:0' \
  --fix_seed 2021 \
  --batch_size 16\
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 0 \
  --prompt_tune_epoch 20\
  --learning_rate 1e-4 \
  --weight_decay 1e-5\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --fix_seed 2024 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_interface.yaml \
  --subsample_pct 4-shot\


python run.py \
  --is_training 1 \
  --device 'cuda:0' \
  --fix_seed 2021 \
  --batch_size 16\
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 0 \
  --prompt_tune_epoch 20\
  --learning_rate 1e-4 \
  --weight_decay 1e-5\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --fix_seed 2024 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_interface.yaml \
  --subsample_pct 8-shot\


python run.py \
  --is_training 1 \
  --device 'cuda:0' \
  --fix_seed 2021 \
  --batch_size 16\
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 0 \
  --prompt_tune_epoch 20\
  --learning_rate 1e-4 \
  --weight_decay 1e-5\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --fix_seed 2024 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_interface.yaml \
  --subsample_pct 16-shot\


python run.py \
  --is_training 1 \
  --device 'cuda:0' \
  --fix_seed 2021 \
  --batch_size 16\
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model 512 \
  --des 'Exp' \
  --train_epochs 0 \
  --prompt_tune_epoch 20\
  --learning_rate 1e-4 \
  --weight_decay 1e-5\
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 1 \
  --clip_grad 5 \
  --fix_seed 2024 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/data_config/few_shot/fewshot_interface.yaml \
  --subsample_pct 0.1\