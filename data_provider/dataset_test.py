from data_loader import *
from torch.utils.data import DataLoader, Subset
if __name__ == '__main__':

    root_path = './datasets/PHM_dataset/PHM_Challenge2024'

    print('*'*10, 'PHM_Challenge2024', '*'*10)
    dataset = Dataset_Challenge2024(root_path, seq_len=1024, stride_len=128, flag='pretrain',
                    down_sampling_scale=10, start_percentage=0.0, end_percentage=1.0)
    dataset.show_file_info()