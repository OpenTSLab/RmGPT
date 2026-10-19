from pathlib import Path
import numpy as np
import torch
import h5py
import bisect
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

PHM_LABEL_TO_INDEX = {
    'B007': 0,
    'B014': 1,
    'B021': 2,
    'B028': 3,
    'IR007': 4,
    'IR014': 5,
    'IR021': 6,
    'IR028': 7,
    'OR007_3': 8,
    'OR007_6': 9,
    'OR007_12': 10,
    'OR014_6': 11,
    'OR021_3': 12,
    'OR021_6': 13,
    'OR021_12': 14,
    'normal': 15,
    'broken': 16,
    'wear': 17,
    'pitting': 18,
    'pitting_wear': 19,
    'chip': 20,
    'N': 21,
    'IR_1': 22,
    'IR_2': 23,
    'IR_3': 24,
    'IR_4': 25,
    'OR_1': 26,
    'OR_2': 27,
    'OR_3': 28,
    'OR_4': 29,
    'RO_1': 30,
    'RO_2': 31,
    'RO_3': 32,
    'RO_4': 33,
}

class Dataset_PHM(Dataset):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, flag=None, cross_condition=None,
                 down_sampling_scale=1, data_path=None, start_percentage=None, end_percentage=None, padd_channel_num =None,
                ):
        
        self.condition_code = {}
        self.channel_code = {}

        self.__root_path = root_path
        self.__data_path = data_path
        self.__file_path = Path(self.__root_path) / self.__data_path
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag
        self.cross_condition = cross_condition
        self.feature_size = None
        self.length = None

        self.file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_channel_names = []


        self.__read_data__()
        if padd_channel_num is not None:
            self.feature_size[1] = padd_channel_num
            self.__pad_channel__()
        self.__standardize__()
        self.global_label_index = PHM_LABEL_TO_INDEX

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None
    def __pad_channel__(self):
        for i, (data, channel_names) in enumerate(zip(self.file_data, self.file_channel_names)):
            if data.shape[1] == 1:
                new_data = np.concatenate([data, data, data], axis=1)
                new_channel_names = channel_names * 3
                self.file_data[i] = new_data
                self.file_channel_names[i] = new_channel_names
            if data.shape[1] == 2:
                new_data = np.concatenate([data, data[:, 0:1]], axis=1)
                new_channel_names = channel_names + [channel_names[0]]
                self.file_data[i] = new_data
                self.file_channel_names[i] = new_channel_names
        return None



    def __read_data__(self):
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.file_list = [i for i in self.__file]

        self.feature_size = [self.seq_len, self.__file[self.file_list[0]][:].shape[1]]

        if self.cross_condition is not None:
            # cross_condition 不为None时，表示按工况划分训练集和测试集
            # cross_condition 应为一个长度为工况数的 [0, 1] 列表，表明在 train 的时候是否使用该工况，1表示使用，0表示不使用
            # 在 test 的时候列表值要取反
            print(f'Cross condition setting when [{self.flag} stage]')
            
            condition_set = set([self.__file[file].attrs['condition'] for file in self.file_list])

            # 检查该数据集是否存在多个工况
            if len(condition_set) > 1:
                # 如果存在，检查cross_condition的长度是否等于工况数
                assert len(condition_set) == len(self.cross_condition), 'The condition set is not equal to the cross_condition'

                # 按工况取出训练集或测试集
                self.file_list = self.__condition_masker__(condition_set)
                # self.start_percentage 和 self.end_percentage 不需要进行修改

            else:
                # 如果不存在多个工况，则无法按工况划分训练集和测试集，退出程序
                raise Exception('The dataset has only one condition, can not split the dataset by condition')

        else:
            # cross_condition为None时，表示不按工况划分训练集和测试集
            # 直接使用所有的数据按8：2切分
            # 非跨工况的数据集切分方式
            print(f'Non-cross condition setting when [{self.flag} stage]')
            
            if self.flag == 'pretrain':
                pass

            # 需要根据 train 和 test 修改 start_percentage 和 end_percentage
            elif self.flag == 'train':
                if self.start_percentage is None:
                    self.start_percentage = 0.0
                    self.end_percentage = 0.8
            elif self.flag == 'test':
                self.start_percentage = 0.8
                self.end_percentage = 1.0
            print(f'[{Path(self.__root_path).name}] start_percentage: {self.start_percentage}, end_percentage: {self.end_percentage}')

        global_index = 0

        if isinstance(self.end_percentage, str):
            print("Begin few-shot evaluation,evluation situation:{} per class".format(self.end_percentage))
            shot_num = int(self.end_percentage[0])
            #选出满足self.__file中attrs['channel_name']=3的file
            # self.file_list = [file for file in self.file_list if len(self.__file[file].attrs['channel_name']) == 3]
            self.file_label = [self.__file[file].attrs['label'] for file in self.file_list]
            if shot_num ==1:
                #选出每个label的第一个file
                # 创建一个空字典来存储每个唯一label的第一个文件
                unique_label_files = {}
                # 遍历原始的file_list
                for file in self.file_list:
                    label = self.__file[file].attrs['label']
                    # 如果该label还没有在字典中，就添加当前文件
                    if label not in unique_label_files:
                        unique_label_files[label] = file

                # 从字典中提取所有的文件名，构成新的file_list
                self.file_list = list(unique_label_files.values())
                shot_per_sample = 1
            else:
                shot_per_sample = int(shot_num/4)
            for file in self.file_list:
                data = self.__file[file][:]
                self.global_idxes.append(global_index)
                data = data[::self.down_sampling_scale]
                file_length = data.shape[0]
                self.file_data.append(data)
                self.file_label.append(self.__file[file].attrs['label'])
                self.file_condition.append(self.__file[file].attrs['condition'])
                self.file_channel_names.append(self.__file[file].attrs['channel_name'])
                total_sample_num =shot_per_sample
                start_idx =0
                end_idx = total_sample_num * self.stride_len
                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.stride_len
        else:
            for file in self.file_list:
                data = self.__file[file][:]
                for i in range(self.down_sampling_scale):
                    self.global_idxes.append(global_index)
                    data_i = data[i::self.down_sampling_scale]
                    file_length = data_i.shape[0]
                    self.file_data.append(data_i)
                    self.file_label.append(self.__file[file].attrs['label'])
                    self.file_condition.append(self.__file[file].attrs['condition'])
                    self.file_channel_names.append(self.__file[file].attrs['channel_name'])

                    total_sample_num = (file_length - self.seq_len) // self.stride_len + 1

                    start_idx = int(total_sample_num * self.start_percentage) * self.stride_len
                    end_idx = int(total_sample_num * self.end_percentage-1) * self.stride_len
                    self.local_idxes.append(start_idx)
                    global_index += (end_idx - start_idx) // self.stride_len + 1
        self.length = global_index

        # self.__file.close()

        return None
    
    def __condition_masker__(self, condition_set):

        if self.flag == 'pretrain':
            print('Use all condition')
            return self.file_list
        elif self.flag == 'train':
            pass
        elif self.flag == 'test':
            self.cross_condition = [1 - x for x in self.cross_condition]
        
        used_condition_set = [condition for condition, mask in zip(condition_set, self.cross_condition) if mask]
        
        print(f'[{Path(self.__root_path).name}] All condition: {condition_set}')
        print(f'[{Path(self.__root_path).name}] Used condition: {used_condition_set}')

        new_file_list = []
        
        for file in self.file_list:
            if self.__file[file].attrs['condition'] in used_condition_set:
                new_file_list.append(file)
        
        return new_file_list


    
    def show_file_info(self):
        print(f"Channel_names_list: {self.__file[self.file_list[0]].attrs['channel_name']}")
        point_number = 0
        for i, file in enumerate(self.file_list):
            data = self.__file[file][:]
            point_number += data.shape[0]
            attrs = self.__file[file].attrs
            attrs_str = ' \t| '.join([f"No. {key}: {len(val)}" if key in ['channel_names', 'channel_name'] else f"{key}: {val}" for key, val in attrs.items()])
            print(f"File {(i+1):02d}: {file} \t| Data shape: {data.shape} \t| {attrs_str}")
        from collections import Counter
        print('Total points:', point_number)
        print('Total labels:', len(set(self.file_label)), ' :', Counter(self.file_label))
        print('Total conditions:', len(set(self.file_condition)), ' :', set(self.file_condition))
        return None
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]

        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.file_label[file_idx]
        condition = self.file_condition[file_idx]
        channel_names = self.file_channel_names[file_idx]

        return {'data': data, 'label': label, 'condition': condition, 'channel_names': channel_names}
    
    def __len__(self):
        return self.length
    
    @property
    def file_num(self):
        return len(self.file_list)
    
class Dataset_CWRU(Dataset_PHM):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='CWRU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_CWRU, self).__init__(root_path, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            self.label_code = {
                'B007': 0,
                'B014': 1,
                'B021': 2,
                'B028': 3,
                'IR007': 4,
                'IR014': 5,
                'IR021': 6,
                'IR028': 7,
                'OR007_3': 8,
                'OR007_6': 9,
                'OR007_12': 10,
                'OR014_6': 11,
                'OR021_3': 12,
                'OR021_6': 13,
                'OR021_12': 14,
                'N': 15,
            }
        if label_type == 'global':
            self.label_code = self.global_label_index
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        num_classes = len(self.label_code)  # 假设标签的数量等于label_code的长度
        label = F.one_hot(torch.tensor(label), num_classes=num_classes)

        return data, label

class Dataset_QPZZ(Dataset_PHM):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='QPZZ.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_QPZZ, self).__init__(root_path, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage)
        
        if label_type == 'local':
            self.label_code = {
                'normal': 0,
                'broken': 1,
                'wear': 2,
                'pitting': 3,
                'pitting_wear': 4,
            }
        if label_type == 'global':
            self.label_code = self.global_label_index

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        num_classes = len(self.label_code)  # 假设标签的数量等于label_code的长度
        label = F.one_hot(torch.tensor(label), num_classes=num_classes)
        return data, label
    
    
class Dataset_SLIET(Dataset_PHM):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='SLIET.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_SLIET, self).__init__(root_path, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage)

        if label_type == 'local':
            self.label_code = {
                'N': 0,
                'IR_1': 1,
                'IR_2': 2,
                'IR_3': 3,
                'IR_4': 4,
                'OR_1': 5,
                'OR_2': 6,
                'OR_3': 7,
                'OR_4': 8,
                'RO_1': 9,
                'RO_2': 10,
                'RO_3': 11,
                'RO_4': 12,
            }
        if label_type == 'global':
            self.label_code = self.global_label_index

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        #
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        num_classes = len(self.label_code)  # 假设标签的数量等于label_code的长度
        label = F.one_hot(torch.tensor(label), num_classes=num_classes)
        return data, label
    
class Dataset_SMU(Dataset_PHM):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='SMU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_SMU, self).__init__(root_path, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage)

        if label_type == 'local':
            self.label_code = {
                'normal': 0,
                'wear': 1,
                'chip': 2,
            }
        if label_type == 'global':
            self.label_code = self.global_label_index

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]


        # 将离散的标签转换为one-hot向量
        num_classes = len(self.label_code)  # 假设标签的数量等于label_code的长度
        label = F.one_hot(torch.tensor(label), num_classes=num_classes)
        return data, label

class Dataset_XJTU(Dataset):
    def __init__(self, root_path, seq_len=1024, stride_len=1024, num_look_back=8, flag=None, test_idx=None,
                 down_sampling_scale=1, data_path='XJTU_2.hdf5', start_percentage=0.0, end_percentage=1.0):

        '''
        ********** XJTU_RUL Bearing List **********
        # 打#的轴承剔除掉，不参与训练和测试
        # [0] 2100rpm_12kN_2h2min_bearing1_4      112
        [0] 2100rpm_12kN_2h38min_bearing1_3     56
        [1] 2100rpm_12kN_2h3min_bearing1_1      75
        [2] 2100rpm_12kN_2h41min_bearing1_2     45
        [3] 2100rpm_12kN_52min_bearing1_5       31
        [4] 2250rpm_11kN_2h41min_bearing2_2     47
        [5] 2250rpm_11kN_42min_bearing2_4       24
        [6] 2250rpm_11kN_5h39min_bearing2_5     145
        [7] 2250rpm_11kN_8h11min_bearing2_1     450
        [8] 2250rpm_11kN_8h53min_bearing2_3     290
        [9] 2400rpm_10kN_1h54min_bearing3_5    8
        [10] 2400rpm_10kN_25h15min_bearing3_4   1428
        # [12] 2400rpm_10kN_41h36min_bearing3_2   2069
        [11] 2400rpm_10kN_42h18min_bearing3_1   2358
        [12] 2400rpm_10kN_6h11min_bearing3_3    338
        '''
        #如果seq_len不被num_look_back整除，报错
        assert seq_len % num_look_back == 0, 'seq_len should be divisible by num_look_back'
        self.__root_path = root_path
        self.__data_path = data_path
        self.__file_path = Path(self.__root_path) / self.__data_path
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.num_look_back = num_look_back
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag
        self.test_idx = test_idx

        self.feature_size = None
        self.channel_names = None
        self.length = None

        self.condition_code = {}
        self.channel_code = {}
        self.fault_code = {}

        # 以下定义的列表用于存放后处理的数据切片，索引，标签，工况，故障元素等信息
        # 所有列表的长度均一致，且一一对应
        self.file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_name = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_fault_element = []

        # 以下定义的字典用于存放文件名和其对应的寿命、自适应步长、初始预测时间FPT
        # 均是以字典的形式访问，key为文件名，仅在__read_data__中使用
        self.file_life = {}
        self.adaptive_stride = {}
        self.subject_fpt = {
            '2100rpm_12kN_2h2min_bearing1_4': 112,
            '2100rpm_12kN_2h38min_bearing1_3': 56,
            '2100rpm_12kN_2h3min_bearing1_1': 75,
            '2100rpm_12kN_2h41min_bearing1_2': 45,
            '2100rpm_12kN_52min_bearing1_5': 31,
            '2250rpm_11kN_2h41min_bearing2_2': 47,
            '2250rpm_11kN_42min_bearing2_4': 24,
            '2250rpm_11kN_5h39min_bearing2_5': 145,
            '2250rpm_11kN_8h11min_bearing2_1': 450,
            '2250rpm_11kN_8h53min_bearing2_3': 290,
            '2400rpm_10kN_1h54min_bearing3_5': 8,
            '2400rpm_10kN_25h15min_bearing3_4': 1428,
            '2400rpm_10kN_41h36min_bearing3_2': 2069,
            '2400rpm_10kN_42h18min_bearing3_1': 2358,
            '2400rpm_10kN_6h11min_bearing3_3': 338,
        }

        self.__read_data__()
        self.__standardize__()

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None


    def __read_data__(self):
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.file_list = [i for i in self.__file]
        self.file_list.remove('2100rpm_12kN_2h2min_bearing1_4')
        self.file_list.remove('2400rpm_10kN_41h36min_bearing3_2')
        self.file_life = {file: self.format_life_time(self.__file[file].attrs['life_time']) for file in self.file_list}
        self.degrade_len = {file: self.file_life[file] - self.subject_fpt[file] + 1 for file in self.file_list}

        self.feature_size = [self.seq_len, self.__file[self.file_list[0]][:].shape[1]]
        self.one_sample_length = int(25600 * 1.28)
        self.down_sampled_one_sample_length = self.one_sample_length // self.down_sampling_scale
        self.adaptive_stride = self.get_adaptive_stride()
        self.cut_length_at_one_life_time = self.seq_len // self.num_look_back

        if self.flag == 'pretrain':
            current_file_list = self.file_list
        elif self.flag == 'train':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i not in self.test_idx]
        elif self.flag == 'test':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i in self.test_idx]

        global_index = 0
        total_point_num = 0
        for file in current_file_list:

            life_label = np.concatenate([np.ones(self.subject_fpt[file] - 1), np.linspace(1, 0, self.file_life[file] - self.subject_fpt[file] + 1)])
            life_label = torch.tensor(life_label.repeat(self.one_sample_length), dtype=torch.float32)
            data = self.__file[file][:]
            total_point_num += data.shape[0]
            if self.flag in ['train', 'test']:
                data = data[(self.subject_fpt[file] - self.num_look_back) * self.one_sample_length:]
                life_label = life_label[(self.subject_fpt[file] - self.num_look_back) * self.one_sample_length:]
            elif self.flag == 'pretrain':
                pass

            # 去除每个sample文件降采样后剩余的部分, 保证每部分降采样后的序列取出的样本数相同
            data = data.reshape(-1, self.one_sample_length, data.shape[-1])
            life_label = life_label.reshape(-1, self.one_sample_length)
            data = data[:, :self.down_sampling_scale * self.down_sampled_one_sample_length, :]
            life_label = life_label[:, :self.down_sampling_scale * self.down_sampled_one_sample_length]
            data = data.reshape(-1, data.shape[-1])
            life_label = life_label.reshape(-1)

            for i in range(self.down_sampling_scale):
                # 降采样后的数据
                self.global_idxes.append(global_index)
                data_i = data[i::self.down_sampling_scale]
                label_i = life_label[i::self.down_sampling_scale]
                file_length = data_i.shape[0] - (self.num_look_back - 1) * self.down_sampled_one_sample_length
                self.file_name.append(file)
                self.file_data.append(data_i)
                self.file_label.append(label_i)
                self.file_condition.append(self.__file[file].attrs['condition'])

                total_sample_num = (file_length - self.seq_len) // self.adaptive_stride[file] + 1
                start_idx = int(total_sample_num * self.start_percentage) * self.adaptive_stride[file]
                end_idx = int(total_sample_num * self.end_percentage - 1) * self.adaptive_stride[file]

                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.adaptive_stride[file] + 1
        self.length = global_index

        # 生成一组初始的mask索引，即如果self.num_look_back=8, 则生成8组256长度的的索引，
        # 每组之间间隔为self.down_sampled_one_sample_length = 6553, 总长度为self.seq_len=2048
        self.init_mask_indexes = torch.tensor(np.concatenate([np.arange(256) + i * self.down_sampled_one_sample_length for i in range(self.num_look_back)]), dtype=torch.long)

        # self.__file.close()
        print('Total points:', total_point_num)
        return None
    
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.adaptive_stride[self.file_name[file_idx]] + self.local_idxes[file_idx]

        # 检查item_start_idx是否针对长度为self.down_sampled_one_sample_length的同寿命样本存在溢出
        overflow = (item_start_idx + self.cut_length_at_one_life_time) % self.down_sampled_one_sample_length

        # 如果溢出，向后移直到不溢出，保证label的一致性
        item_start_idx = item_start_idx if overflow >= self.cut_length_at_one_life_time else item_start_idx + self.cut_length_at_one_life_time - overflow

        # 根据item_start_idx重新定位mask索引
        mask_indexes = self.init_mask_indexes + item_start_idx

        # 根据mask从原序列中取数据和标签，标签取最后一个时刻的寿命
        data = torch.index_select(self.file_data[file_idx], 0, mask_indexes)
        label = torch.index_select(self.file_label[file_idx], 0, mask_indexes[-self.cut_length_at_one_life_time:])

        if self.flag in ['pretrain', 'train']:
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element}
            return data, label[0]
        elif self.flag == 'test':
            name_and_id = (self.file_name[file_idx], idx - self.global_idxes[file_idx])
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element, 'name_and_id': name_and_id}
            return data, label[0], name_and_id
        
        
    def __len__(self):
        return self.length
    


    def get_adaptive_stride(self):

        min_degrade_len = min(self.degrade_len.values())
        total_num_sample_of_one_down_sampled_sample = (self.down_sampled_one_sample_length * min_degrade_len - self.seq_len) // self.stride_len + 1
        adaptive_stride = {}

        for file, degrade_len in self.degrade_len.items():
            if degrade_len == min_degrade_len:
                adaptive_stride[file] = self.stride_len
            else:
                adaptive_stride[file] = (degrade_len * self.down_sampled_one_sample_length - self.seq_len) // (total_num_sample_of_one_down_sampled_sample + 1)
        return adaptive_stride


    def format_life_time(self, life_time_str):

        if 'h' not in life_time_str:
            life_time_str = '0h' + life_time_str
        hour = int(life_time_str.split('h')[0])
        minute = int(life_time_str.split('h')[1][:-3])
        life = hour * 60 + minute
        return life



class Dataset_Challenge2024(Dataset):

    def __init__(self, root_path, seq_len=1024, stride_len=1024, flag=None,label_type='local',
                 down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0):

        self.__root_path = root_path
        self.__data_path = ['PHM2024_train_1.hdf5', 'PHM2024_train_2.hdf5', 'PHM2024_test.hdf5']
        self.__file_path = [(Path(self.__root_path) / path) for path in self.__data_path]
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag

        self.feature_size = None
        self.length = None

        self.all_file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_name = []
        self.file_data = []
        self.file_label = []
        self.channel_names = None
        self.__read_data__()
        self.__standardize__()
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

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None


    def __read_data__(self):

        if self.flag == 'pretrain':
            self.file_path = self.__file_path
        elif self.flag in ['train', 'test']:
            self.file_path = self.__file_path[:2]
        elif self.flag == 'test_remote':
            self.file_path = self.__file_path[2:]

        self.__file = []

        self.all_file_flag = []
        for i, file_path in enumerate(self.file_path):
            self.__current_file = h5py.File(str(file_path), 'r')
            self.__file.append(self.__current_file)
            file_list = [i for i in self.__current_file]
            file_flag = [i,] * len(file_list)
            self.all_file_list.extend(file_list)
            self.all_file_flag.extend(file_flag)

        if self.channel_names is None:
            self.channel_names = (self.__file[self.all_file_flag[0]][self.all_file_list[0]].attrs['channel_name'])
        if self.feature_size is None:
            self.feature_size = [self.seq_len, self.__file[self.all_file_flag[0]][self.all_file_list[0]][:].shape[1]]

        if self.flag == 'train':
            self.start_percentage = 0.0
            self.end_percentage = 0.8
        elif self.flag == 'test':
            self.start_percentage = 0.8
            self.end_percentage = 1.0

        global_index = 0
        for file, flag in zip(self.all_file_list, self.all_file_flag):
            data = self.__file[flag][file][:]
            for i in range(self.down_sampling_scale):
                self.global_idxes.append(global_index)
                data_i = data[i::self.down_sampling_scale]
                file_length = data_i.shape[0]
                self.file_name.append(file)
                self.file_data.append(data_i)
                self.file_label.append(self.__file[flag][file].attrs['label'])

                total_sample_num = (file_length - self.seq_len) // self.stride_len + 1
                start_idx = int(total_sample_num * self.start_percentage) * self.stride_len
                end_idx = int(total_sample_num * self.end_percentage - 1) * self.stride_len

                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.stride_len + 1

        self.length = global_index

        # for f in self.__file:
        #     f.close()

        return None

    
    def show_file_info(self):
        print(f"Channel_names_list: {self.channel_names}")
        total_point_num = 0
        for i, (file, flag) in enumerate(zip(self.all_file_list, self.all_file_flag)):
            data = self.__file[flag][file][:]
            total_point_num += data.shape[0]
            attrs = self.__file[flag][file].attrs
            attrs_str = ' \t| '.join([f"No. {key}: {len(val)}" if key == 'channel_name' else f"{key}: {val}" for key, val in attrs.items()])
            print(f"File {(i+1):02d}: {file} \t| Data shape: {data.shape} \t| {attrs_str}")
        from collections import Counter
        print('Total labels:', len(set(self.file_label)), ' :', Counter(self.file_label))
        print('Total points:', total_point_num)
        return None
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]

        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]

        if self.flag == 'pretrain':
            return data, 0
        elif self.flag in ['train', 'test']:
            label = self.label_code[self.file_label[file_idx]]    
            # 将离散的标签转换为one-hot向量
            num_classes = len(self.label_code)  # 假设标签的数量等于label_code的长度
            label = F.one_hot(torch.tensor(label), num_classes=num_classes)
            return data, label
        elif self.flag == 'test_remote':
            return data, self.file_name[file_idx]

    def __len__(self):
        return self.length
    
    @property
    def file_num(self):
        return len(self.all_file_list)


