from data_provider.data_loader import Dataset_CWRU, Dataset_QPZZ, Dataset_SLIET, \
    Dataset_SMU, Dataset_Challenge2024, Dataset_XJTU
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'CWRU': Dataset_CWRU,
    'QPZZ': Dataset_QPZZ,
    'SLIET': Dataset_SLIET,
    'SMU': Dataset_SMU,
    'PHM_Challenge2024': Dataset_Challenge2024,
    'RUL_XJTU': Dataset_XJTU,
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root_path(root_path):
    # Relative dataset paths in YAML configs resolve against the repo root.
    path = Path(root_path)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def random_subset(pct, seed):
    # Draw a random training window whose length is pct of the full range.
    pct = pct * 0.8
    max_start_pct = 0.8 - pct
    generator = torch.Generator()
    if seed is None:
        seed = 2024
    generator.manual_seed(seed)
    start_pct = torch.rand(1, generator=generator) * max_start_pct
    end_pct = start_pct + pct
    return start_pct, end_pct


def data_provider(args, config, flag, ddp=False):
    Data = data_dict[config['data']]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    elif flag == 'test_remote':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
    elif flag == 'pretrain':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False if 'RUL' in config['task_name'] else True
        drop_last = True
        batch_size = args.batch_size

    if 'RUL' in config['task_name']:
        data_set = Data(
            root_path=_resolve_root_path(config['root_path']),
            seq_len=config['seq_len'],
            stride_len=config['stride_len'],
            down_sampling_scale=config['down_sample'],
            flag=flag,
            test_idx=config['test_idx'],
        )
    else:
        if args.subsample_pct is not None and flag == "train":
            if isinstance(args.subsample_pct, float):
                start_percentage, end_percentage = random_subset(
                    args.subsample_pct, args.fix_seed)
            else:
                start_percentage = 0
                end_percentage = args.subsample_pct
        else:
            start_percentage = None
            end_percentage = None
        data_set = Data(
            root_path=_resolve_root_path(config['root_path']),
            seq_len=config['seq_len'],
            stride_len=config['stride_len'],
            down_sampling_scale=config['down_sample'],
            label_type=config['label_type'] if 'label_type' in config else None,
            flag=flag,
            start_percentage=start_percentage,
            end_percentage=end_percentage,
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False if ddp else shuffle_flag,
        num_workers=args.num_workers,
        sampler=DistributedSampler(data_set) if ddp else None,
        drop_last=drop_last)
    return data_set, data_loader
