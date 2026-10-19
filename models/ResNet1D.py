import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, args, configs_list, pretrain=False):
        super(Model, self).__init__()

        self.configs_list = configs_list
        self.d_model = args.d_model  # Adding d_model definition

        # Model settings
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len
        self.channel_num = configs_list[0][1]['enc_in']
        self.num_class = configs_list[0][1]['num_class']
        self.input_len = args.input_len
        self.e_layer = args.e_layers  # Adding e_layer definition
        self.d_model = args.d_model
        self.in_planes = 64
        self.planes_list = [64, 128, 256, 512]
        self.initial_conv = nn.Conv1d(self.channel_num, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)  # Initial conv layer to handle channel mismatch
        self.initial_bn = nn.BatchNorm1d(self.in_planes)
        self.layers = self._make_layers(self.e_layer)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_input_dim = self.planes_list[self.e_layer% 4-1] # Since adaptive pool outputs (B, 512, 1)

        self.classifier = nn.Linear(self.fc_input_dim, self.num_class)
        self.rul_predictor = nn.Linear(self.fc_input_dim, 1)


    def _make_layers(self, e_layer):
        layers = []
        in_planes = self.in_planes
        planes = [64, 128, 256, 512]

        for i in range(e_layer):
            stride = 1 if i == 0 else 2
            layers.append(BasicBlock1D(in_planes, planes[i % 4], stride))
            in_planes = planes[i % 4]

        return nn.Sequential(*layers)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, task_id=None, task_name=None, enable_mask=None):
        # X_enc [B, L, D]
        x = x_enc.permute(0, 2, 1)  # Adjusting to match input size [B, D, L]
        x = F.relu(self.initial_bn(self.initial_conv(x)))  # Initial conv to match channels
        out = self.layers(x)
        out = self.adaptive_pool(out)  # Apply adaptive pooling
        out = out.view(out.size(0), -1)  # Flatten the output tensor

        if 'classification' in task_name:
            dec_out = self.classifier(out)
            return dec_out  # [B, N]
        if 'RUL' in task_name:
            dec_out = self.rul_predictor(out)
            return dec_out  # [B, 1]
        return None


def test_model():
    import yaml
    import argparse

    def read_task_data_config(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        task_dataset_config = config.get('task_dataset', {})
        return task_dataset_config

    def get_task_data_config_list(task_data_config, default_batch_size=32):
        task_data_config_list = []

        for task_name, task_config in task_data_config.items():
            task_config['max_batch'] = default_batch_size
            task_data_config_list.append([task_name, task_config])

        return task_data_config_list
    #读取args
    parser = argparse.ArgumentParser(description='RmGPT Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='status') 
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
                        default='models/test.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # pretrain
    parser.add_argument('--right_prob', type=float,
                        default=1.0, help='right mask prob')
    parser.add_argument('--min_mask_ratio', type=float,
                        default=0.5, help='min right mask prob')
    parser.add_argument('--max_mask_ratio', type=float,
                        default=0.8, help='max right mask prob')
    parser.add_argument('--min_keep_ratio', type=float, default=None,
                        help='min crop ratio for various length in pretraining')

    #device
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    # ddp
    parser.add_argument('--ddp',type=bool,default=False,help='whether to use ddp')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=32,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='optimizer learning rate')
    parser.add_argument('--beta2', type=float,
                        default=0.999, help='optimizer beta2')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='eps for optimizer')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='disabled')
    parser.add_argument('--clip_grad', type=float, default=None, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=16,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--input_len", type=int, default=2048)

    args = parser.parse_args()
    # 创建一个Model实例
    config = read_task_data_config(args.task_data_config_path)
    config_list = get_task_data_config_list(config)
    model = Model(args,config_list,pretrain=True)

    

    # 创建一些假的输入数据
    x_enc = torch.randn(10, 2048, 3)  # 假设有10个样本，每个样本有50个时间步，每个时间步有100个特征
    x_mark_enc = torch.randn(10, 2048,3)
    x_dec = torch.randn(10, 2048, 3)
    x_mark_dec = torch.randn(10, 2048, 3)
    mask = torch.ones(10, 2048, dtype=torch.bool)  # 假设所有的数据都是有效的
    task_id = torch.tensor([0])  # 假设任务ID为0
    task_name = 'classification_PHM'  # 
    enable_mask = False  

    # 调用forward方法
    dec_out = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, task_id, task_name, enable_mask)

    # 打印输出
    print(dec_out.shape)

# 在脚本的最后调用测试函数
if __name__ == '__main__':
    test_model()