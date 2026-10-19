model_name=ResNet1D
exp_name=ResNet1D
wandb_mode=${WANDB_MODE:-disabled}
ptune_name=few-shot
d_model=512
learning_rate=3e-4
e_layers=18
batch_size=16
device='cuda:0'

python run.py\
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name\
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size\
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 1-shot\


python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 4-shot\

python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 8-shot\

python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 16-shot\

python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 16-shot\

python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --train_epochs 10 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --task_data_config_path  data_provider/data_config/few_shot/fewshot_interface.yaml\
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\
  --subsample_pct 0.1\
