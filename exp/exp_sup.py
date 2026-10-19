from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate, cal_accuracy, adjustment
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.metrics import metric
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.dataloader import BalancedDataLoaderIterator
from utils.layer_decay import param_groups_lrd
from utils.ddp import get_world_size, is_main_process, gather_tensors_from_all_gpus

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import wandb
import sys
import copy

warnings.filterwarnings('ignore')


def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']

    return wrapper


# Replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def get_loss_by_name(loss_name):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss()
    else:
        print("no loss function found!")
        exit()


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args):
        super(Exp_All_Task, self).__init__()

        self.args = args
        self.ori_task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        self.ori_task_data_config_list = get_task_data_config_list(
            self.ori_task_data_config, default_batch_size=self.args.batch_size)

        self.task_data_config = self.ori_task_data_config
        self.task_data_config_list = self.ori_task_data_config_list
        if args.ddp:
            device_id = dist.get_rank() % torch.cuda.device_count()
            print("this device_id:", device_id)
        else:
            device_id = args.device
        self.device_id = device_id
        print("device id", self.device_id)
        self.model = self._build_model()

    def _build_model(self, ddp=True):
        if ddp==False: 
            ddp = False
        else:
            ddp = self.args.ddp
        import importlib
        module = importlib.import_module("models."+self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
        return model

    def _get_data(self, flag):
        ddp = self.args.ddp

        this_task_data_config = self.task_data_config
        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if 'classification' in task_config['task_name'] and flag == 'val':
                # No separate val split: evaluate on the test split for val
                flag = 'test'
            data_set, data_loader = data_provider(
                self.args, task_config, flag, ddp=ddp)
            data_set_list.append(data_set)
            data_loader_list.append(data_loader)
            print(task_data_name, len(data_set))
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        if self.args.ddp:
            world_size = get_world_size()
        else:
            world_size = 1    
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        self.real_learning_rate = min(real_learning_rate,0.0003)
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("effective batch size: %d" % eff_batch_size)
        if self.args.layer_decay is not None:
            print("layer decay: %.2f" % self.args.layer_decay)
            model_without_ddp = self.model.module
            param_groups = param_groups_lrd(model_without_ddp, self.args.weight_decay,
                                            no_weight_decay_list=[
                                                'prompts', 'mask_tokens', 'cls_tokens', 'category_tokens'],
                                            layer_decay=self.args.layer_decay
                                            )
            model_optim = optim.Adam(param_groups, lr=real_learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(
            ), lr=real_learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, config_list):
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if 'classification' in each_config[1]['task_name']:
                    loss_name = 'CE'
                elif 'RUL' in each_config[1]['task_name'] :
                    loss_name = 'MSE'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(get_loss_by_name(loss_name))

        return criterion_list

    def choose_training_parts(self, prompt_tune=False):
        #通过控制可训练参数的梯度来控制训练部分
        for name, param in self.model.named_parameters():
            if prompt_tune:
                #仅仅训练网络中保存的任务参数
                if  'cls_head' in name or 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name  or 'cls_token' in name or 'category_token' in name or 'token_head' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                #把所有的参数都训练
                param.requires_grad = True
        if not prompt_tune:
            print("all trainable.")

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        # Load pretrained weights (Optional),如果有预训练权重，加载预训练权重；读取相同名字的部分都权重
        if self.args.pretrained_weight is not None:
            if self.args.pretrained_weight == 'auto':
                pretrain_weight_path = os.path.join(
                    self.path, 'pretrain_checkpoint.pth')
            else:
                pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:',
                  pretrain_weight_path, folder=self.path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu')['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg, folder=self.path)

        # Data
        _, train_loader_list = self._get_data(flag='train')
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test')
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Model param check
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Parameters number for all {} M".format(
            pytorch_total_params/1e6), folder=self.path)
        model_param = []
        for name, param in self.model.named_parameters():
            if ('prompts' in name and 'prompt2forecat' not in name) or 'prompt_token' in name or \
                'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                print('skip this:', name)
            else:
                model_param.append(param.numel())
        model_total_params = sum(model_param)
        print("Parameters number for RmGPT {} M".format(
            model_total_params/1e6), folder=self.path)

        # Optimizer and Criterion
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler()

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle, criterion_list)
            torch.cuda.empty_cache()
        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()
        #先prompt后finetune，和args里的设置有关
        for epoch in range(self.args.train_epochs+self.args.prompt_tune_epoch):
            adjust_learning_rate(model_optim, epoch,
                                 self.real_learning_rate, self.args)
            # Prompt learning
            if (epoch+1) <= self.args.prompt_tune_epoch:
                self.choose_training_parts(prompt_tune=True)
            else:
                self.choose_training_parts(prompt_tune=False)

            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)

            # we report the results of last epoch and not find the best epoch based on val set, since some datasets do not have val set
            avg_cls_acc, avg_rul_mse, avg_rul_mae = self.test(
                setting, load_pretrain=False, test_data_list=test_data_list, test_loader_list=test_loader_list)

            # save ckpt
            if is_main_process():
                if self.args.prompt_tune_epoch >= 1:
                    torch.save(self.model.state_dict(),
                               os.path.join(path, 'ptune_checkpoint.pth'))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(path, 'checkpoint.pth'))

        if is_main_process():
            wandb.log({'Final_RUL-mse': avg_rul_mse,
                       'Final_RUL-mae': avg_rul_mae, 'Final_CLS-acc': avg_cls_acc})
            print("Final score: RUL-mse: {}, RUL-mae: {}, CLS-acc {}".format(avg_rul_mse,
                                                                             avg_rul_mae, avg_cls_acc), folder=self.path)

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = max(self.args.acc_it,1.0)
        max_norm = self.args.clip_grad

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0

        for i, (sample_init, task_id) in enumerate(data_loader_cycle):

            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            if small_batch_size != self.args.batch_size:
                sample_list = self.split_batch(
                    sample_init, small_batch_size, task_name)
                len_sample_list = len(sample_list)
            else:
                sample_list = [sample_init]
                len_sample_list = 1
            len_sample_list = max(len_sample_list, 1.0)
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if  'classification' in task_name:
                    loss = self.train_classification(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif  'RUL' in task_name:
                    loss = self.train_RUL(
                            self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0

                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)
            #如果 loss 是 nan 或者 inf，跳过这次迭代
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print("loss is too large, skip this iteration!")
                continue
            ## 当update_grad为True时，会更新梯度
            norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))
            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum += loss_display
            loss_sum_display = loss_sum

            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                wandb.log(
                    {'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 'norm_value': norm_value, "loss_sum": loss_sum_display/(i+1)})

            if (i + 1) % 100 == 0:
                if norm_value == None:
                    norm_value = -1
                if is_main_process():
                    print("\titers: {0}, epoch: {1} | norm: {2:.2f} | loss: {3:.7f} | current_loss: {4} |current task: {5}".format(
                        i + 1, epoch + 1, norm_value, loss_sum_display/(i+1), loss_display, task_name, folder=self.path))

        print("Epoch: {} cost time: {}".format(
            epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)
        torch.cuda.synchronize()
        if self.args.ddp:
            dist.barrier()

        return train_loss

    def train_classification(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        try:
             batch_x, label, padding_mask = this_batch
        except:
            batch_x, label = this_batch
            padding_mask = torch.zeros_like(batch_x)

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = padding_mask.float().to(self.device_id)
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.float().squeeze(-1))
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.float().squeeze(-1))
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print("loss is too large, skip this iteration!")
        return loss

    def train_RUL(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        try:
            batch_x, label, padding_mask = this_batch
        except:
            batch_x, label = this_batch
            padding_mask = torch.zeros_like(batch_x)

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = padding_mask.float().to(self.device_id)
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.float().squeeze(-1))
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.long().squeeze(-1))

        return loss

    def test(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test')
        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu')['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        avg_classification_acc = []
        avg_rul_f_mse = []
        avg_rul_f_mae = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            if 'classification' in task_name:
                acc = self.test_classification(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'acc': acc}
                if is_main_process():
                    wandb.log({'eval_CLS-acc_'+data_task_name: acc})
                avg_classification_acc.append(acc)

            elif 'RUL' in task_name:
                mse,mae = self.test_RUL(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'mse': mse,'mae':mae}
                if is_main_process():
                    wandb.log({'eval_RUL-mse_'+data_task_name: mse})
                    wandb.log({'eval_RUL-mae_'+data_task_name: mae})
                avg_rul_f_mse.append(mse)
                avg_rul_f_mae.append(mae)

        avg_classification_acc = np.average(avg_classification_acc)
        avg_rul_mse = np.average(avg_rul_f_mse) if avg_rul_f_mse else 0
        avg_rul_mae = np.average(avg_rul_f_mae) if avg_rul_f_mae else 0

        if is_main_process():
            wandb.log({'avg_eval_CLS-acc': avg_classification_acc,
                       'avg_eval_RUL-mse': avg_rul_mse, 'avg_eval_RUL-mae': avg_rul_mae})
            print("Avg score: CLS-acc {}, RUL-mse: {}, RUL-mae: {}".format(avg_classification_acc,
                                                                           avg_rul_mse, avg_rul_mae), folder=self.path)
            print(total_dict, folder=self.path)
        return avg_classification_acc, avg_rul_mse, avg_rul_mae

    def test_classification(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                padding_mask = torch.zeros_like(batch_x)
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = padding_mask.float().to(self.device_id)
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='classification')
                # outputs = torch.nn.functional.softmax(outputs)
                predictions = torch.argmax(outputs, dim=1)
                true = torch.argmax(label, dim=1)
                preds.append(predictions.detach())
                trues.append(true)
        if self.args.ddp:
            preds = gather_tensors_from_all_gpus(
                preds, self.device_id, to_numpy=False)
            trues = gather_tensors_from_all_gpus(
                trues, self.device_id, to_numpy=False)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # if len(trues.shape) >= 2:
        #     trues = torch.argmax(trues, dim=1)
        predictions = preds.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        del predictions
        del trues
        torch.cuda.empty_cache()

        print('data_task_name: {} accuracy:{}'.format(
            data_task_name, accuracy), folder=self.path)

        return accuracy


    def test_RUL(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        ids_pres ={}
        ids_trues = {}
        with torch.no_grad():
            for i, (batch_x, label,name_and_id) in enumerate(test_loader):
                name = name_and_id[0]
                padding_mask = torch.zeros_like(batch_x)
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = padding_mask.float().to(self.device_id)
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='PHM_RUL')
                #判断是否是新的id
                if name in ids_pres.keys():
                    ids_pres[name].append(outputs)
                    ids_trues[name].append(label)
                else:
                    ids_pres[name] = [outputs]
                    ids_trues[name] = [label]
        for key in ids_pres.keys():
            if self.args.ddp:
                ids_pres[key] = gather_tensors_from_all_gpus(
                    ids_pres[key], self.device_id, to_numpy=False)
                ids_trues[key] = gather_tensors_from_all_gpus(
                    ids_trues[key], self.device_id, to_numpy=False)
            
            ids_pres[key] = torch.cat(ids_pres[key], 0)
            ids_trues[key] = torch.cat(ids_trues[key], 0)
            ids_pres[key] = ids_pres[key].cpu().numpy()
            ids_trues[key] = ids_trues[key].cpu().numpy()
            if len(ids_pres[key].shape)>=2:
                ids_pres[key] = ids_pres[key].squeeze(-1)
            mae, mse, rmse, mape, mspe = metric(ids_pres[key] , ids_trues[key])
            print('data_task_name: {} id:{} mse:{}, mae:{}'.format(
                data_task_name,key, mse, mae), folder=self.path)
        torch.cuda.empty_cache()
        del ids_pres, ids_trues
        return mse, mae


    def split_batch(self, batch, small_batch_size, task_name):
        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]
        # Works for any tuple arity used by the dataset loaders.
        split_inputs = [split_tensor(t, small_batch_size) for t in batch]
        return list(zip(*split_inputs))

    def memory_check(self, data_loader_cycle, criterion_list, holdout_memory=3):
        """
        Checks the memory usage of the model by gradually increasing the batch size until it reaches the maximum batch size that can be supported without running out of memory.

        Args:
            data_loader_cycle (DataLoaderCycle): The data loader cycle object.
            holdout_memory (int): The amount of memory (in GB) to hold out for other operations.

        Returns:
            None
        """
        num_elements = holdout_memory * 1024 * 1024 * 1024 // 4
        extra_mem = torch.empty(
            num_elements, dtype=torch.float32, device=self.device_id)

        model_tmp = self._build_model(ddp=False)
        model_tmp.train()
        model_tmp.zero_grad(set_to_none=True)

        for data_loader_id in range(data_loader_cycle.num_dataloaders):
            batch_size = 1  # Initial batch size
            max_batch_size = 0  # Record the maximum batch size before OOM
            torch.cuda.synchronize()
            model_tmp.zero_grad(set_to_none=True)
            while True:
                try:
                    sample, task_id = data_loader_cycle.generate_fake_samples_for_batch(
                        data_loader_id, batch_size)  # 2 makes the memory larger
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    # Try running the function with the current batch size
                    print(task_id, task_name,
                          sample[0].shape, "max batch size", max_batch_size)
                    if 'classification' in task_name:
                        loss = self.train_classification(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif  'RUL' in task_name:
                        loss = self.train_RUL(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss = loss * 0.0
                    loss.backward()
                    max_batch_size = batch_size  # Update the maximum batch size
                    batch_size *= 2  # Increase the batch size

                    if max_batch_size >= self.args.batch_size:
                        print("Can support default batch size:",
                              self.args.batch_size, max_batch_size)
                        self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                        self.task_data_config_list[task_id][1]['checkpointing'] = False
                        break

                except Exception as e:
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    print(task_id,  "max batch size:", max_batch_size)
                    # If any exception occurs, break the loop
                    self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                    del model_tmp
                    model_tmp = self._build_model(ddp=False)
                    print(f"An exception occurred: {e}")
                    break
        print(self.task_data_config_list)
        del model_tmp
        del extra_mem
        torch.cuda.empty_cache()
        return

class Remote_test(Exp_All_Task):
    def __init__(self, args):
        super().__init__(args)
        self.label_code = {
            'TYPE0': 0,
            'TYPE1': 1,
            'TYPE2': 2,
            'TYPE3': 3,
            'TYPE4': 4,
            'TYPE5': 5,
            'TYPE6': 6,
            'TYPE7': 7,
            'TYPE8': 8,
            'TYPE9': 9,
            'TYPE10': 10,
            'TYPE11': 11,
            'TYPE12': 12,
            'TYPE13': 13,
            'TYPE14': 14,
            'TYPE15': 15,
            'TYPE16': 16,
        }
        self.label_code_reversed = {v: k for k, v in self.label_code.items()}

    def predict(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        test_data_list, test_loader_list = self._get_data(
            flag='test_remote')
        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu')['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        save_path = 'PHM2024_challenge_result'
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]

            if 'classification' in task_name:
                all_result = self.test_classification(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = all_result
                from collections import Counter
                import pickle
                for file, results in all_result.items():
                    counter = Counter(results)
                    total = len(results)
                    print(f"For file {file}:")
                    for number, count in counter.items():
                        percentage = 100 * count / total
                        print(f"Number {number} appears {count} times, which is {percentage:.2f}% of the total.")
                    #保存all_result字典
                    with open(os.path.join(save_path, 'classification_result.pkl'), 'wb') as f:
                        pickle.dump(all_result, f)

                


    def test_classification(self, setting, test_data, test_loader, data_task_name, task_id):
        all_result ={}
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, file_name) in enumerate(test_loader):
                padding_mask = torch.zeros_like(batch_x)
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = padding_mask.float().to(self.device_id)
                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='classification')
                outputs = torch.nn.functional.softmax(outputs)
                # 获取预测分类
                predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                # 将预测分类和文件名配对起来
                for file, prediction in zip(file_name, predictions):
                    if file in all_result.keys():
                        all_result[file].append(prediction)
                    else:
                        all_result[file] = [prediction]
        if self.args.ddp:
            preds = gather_tensors_from_all_gpus( 
                preds, self.device_id, to_numpy=False)
            trues = gather_tensors_from_all_gpus(
                trues, self.device_id, to_numpy=False)
        return all_result
