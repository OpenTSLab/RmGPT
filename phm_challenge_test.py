import argparse
import torch
from exp.exp_sup import  Remote_test
import random
import numpy as np
import wandb
from utils.ddp import is_main_process, init_distributed_mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RmGPT PHM Challenge 2024 test')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                         default=0, help='status')
    parser.add_argument('--model_id', type=str, required=False,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='RmGPT',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/data_config/PHM_competition/PHMchallenge_test.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')
    #device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # ddp
    parser.add_argument('--ddp',type=bool,default=False,help='whether to use ddp')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=2, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, required=True,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='PHM_Competition', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()

    if args.ddp:
        init_distributed_mode(args)
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    print('Args in experiment:')
    print(args)
    exp_name = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des)

    if int(args.prompt_tune_epoch) != 0:
        exp_name = 'Ptune'+str(args.prompt_tune_epoch)+'_'+exp_name
        print(exp_name)

    # if is_main_process():
    #     wandb.init(
    #         name=exp_name,
    #         # set the wandb project where this run will be logged
    #         project=args.project_name,
    #         # track hyperparameters and run metadata
    #         config=args,
    #         mode=args.debug,
    #     )

    Exp = Remote_test


    ii = 0
    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

    exp.predict(setting, load_pretrain=True)
    torch.cuda.empty_cache()
