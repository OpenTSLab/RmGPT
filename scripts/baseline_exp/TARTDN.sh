model_name=TARTDN
exp_name=TARTDN
wandb_mode=${WANDB_MODE:-disabled}
ptune_name=supervised
d_model=512
learning_rate=3e-4
e_layers=4
batch_size=32
device='cuda:5'

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
  --task_data_config_path  data_provider/data_config/baseline/CWRU.yaml\
  --batch_size $batch_size\
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device\

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
  --task_data_config_path  data_provider/data_config/baseline/PHM_Challenge2024.yaml \
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device

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
  --task_data_config_path  data_provider/data_config/baseline/QPZZ.yaml \
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device

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
  --task_data_config_path  data_provider/data_config/baseline/RUL_XJTU.yaml \
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device

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
  --task_data_config_path  data_provider/data_config/baseline/SLIET.yaml \
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device

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
  --task_data_config_path  data_provider/data_config/baseline/SMU.yaml \
  --batch_size $batch_size \
  --input_len 2048\
  --learning_rate $learning_rate\
  --e_layers $e_layers\
  --device $device
