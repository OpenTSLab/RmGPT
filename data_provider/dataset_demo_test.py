from data_loader import *
from torch.utils.data import DataLoader, Subset
if __name__ == '__main__':

    # root_path = './datasets/PHM_dataset/PHM_Challenge2024'

    # print('*'*10, 'PHM_Challenge2024', '*'*10)
    # dataset = Dataset_Challenge2024(root_path, seq_len=2048, stride_len=128, flag='test_remote',
    #                 down_sampling_scale=10, start_percentage=0.0, end_percentage=1.0)
    # # dataset.show_file_info()

    # root_path = './datasets/PHM_dataset/XJTU_RUL_NEW'
    # print('*'*10, 'XJTU_RUL', '*'*10)
    # dataset_train = Dataset_XJTU(root_path, seq_len=2048, stride_len=128, num_look_back=8, flag='train', test_idx=[12],
    #              down_sampling_scale=5, start_percentage=0.0, end_percentage=1.0)
    # # for i in range(len(dataset_train)):
    # #     data, label = dataset_train[i]
    # #     if i % 1000 == 0:
    # #         print(i)

    # # print('Train:', len(dataset_train), len(dataset_train.file_data))

    # # dataset_test = Dataset_XJTU(root_path, seq_len=2048, stride_len=128, flag='test', test_idx=[12],
    # #              down_sampling_scale=5, start_percentage=0.0, end_percentage=1.0)
    # # print('Test:', len(dataset_test), len(dataset_test.file_data))

    # # sample = dataset_train[0]
    
    # # root_path = './datasets/PHM_dataset/PHM2010_RUL'
    # # print('*'*10, 'XJTU_RUL', '*'*10)
    # # dataset = Dataset_PHM2010(root_path, seq_len=1024, stride_len=1024, flag='pretrain',
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)


    # root_path = './datasets/PHM_dataset/QPZZ'

    # # # 只要缺省 cross_condition 参数，即可进入非跨工况模式， 四个数据集都可以使用
    # # # pretrain 使用全部数据
    # # # train 使用前 80% 数据
    # # # test 使用后 20% 数据
    # # print('*'*10, 'QPZZ', '*'*10)
    # dataset = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='pretrain',
    #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    # dataset.show_file_info()
    # # dataset = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='train',
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    
    # # dataset = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='test',
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    
    # # # 跨工况模式，只有 CWRU 和 QPZZ 数据集可以使用
    # # # 需要提供 cross_condition 参数，长度为 4 （工况数）的列表，每个元素为 0 或 1，表示在 train 时是否使用对应的工况数据
    # # # cross_condition 参数为 [1, 1, 1, 0]，表示使用前三个工况的数据（在 train 时）

    # # # 特殊说明：
    # # # 如果 flag == 'pretrain'，则使用全部数据，不受 cross_condition 影响，
    # # # 但是 cross_condition 的长度依然要接受检查，以保证程序的健壮性，最好保持 cross_condition = [1, 1, 1, 1]

    # # # Example 1: 使用前三种工况的数据进行训练，最后一种进行测试
    # # print('*'*10, 'QPZZ', '*'*10)
    # # dataset_train = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='train', cross_condition=[1, 1, 1, 0],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    # # dataset_test = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='test', cross_condition=[1, 1, 1, 0],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    

    # # # Example 2: 使用前两种工况的数据进行训练，最后两种进行测试
    # # print('*'*10, 'QPZZ', '*'*10)
    # # dataset_train = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='train', cross_condition=[1, 1, 0, 0],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    # # dataset_test = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='test', cross_condition=[1, 1, 0, 0],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    
    # # # Example 3: 使用所有数据进行 pretrain, cross_condition 参数可以随意设置，但长度必须是 4
    # # print('*'*10, 'QPZZ', '*'*10)
    # # dataset = Dataset_QPZZ(root_path, seq_len=1024, stride_len=1024, flag='pretrain', cross_condition=[1, 1, 1, 1],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    
    # # # Example 4: 对 SMU 和 SLIET 数据集加入 cross_condition 参数，程序会报错退出
    # # root_path = './datasets/PHM_dataset/SMU'
    # # print('*'*10, 'SMU', '*'*10)
    # # dataset = Dataset_SMU(root_path, seq_len=1024, stride_len=1024, flag='pretrain', cross_condition=[1, ],
    # #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)


    # root_path = './datasets/PHM_dataset/CWRU'
    # print('*'*10, 'CWRU', '*'*10)
    # dataset = Dataset_CWRU(root_path, seq_len=2048, stride_len=128, flag='pretrain',
    #              down_sampling_scale=2,end_percentage=0.1)
    # # dataset_train = Dataset_CWRU(root_path, seq_len=1024, stride_len=128, flag='train',
    # #              down_sampling_scale=2)
    # # dataset_test = Dataset_CWRU(root_path, seq_len=1024, stride_len=128, flag='test',
    # #              down_sampling_scale=2)
    # dataset.show_file_info()
    # # data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    # # for i, (X, y) in enumerate(data_loader):
    # #     print(X.mean(dim=(0,1)), y.shape)
    # #     break
    # root_path = './datasets/PHM_dataset/SMU'
    # print('*'*10, 'SMU', '*'*10)
    # dataset = Dataset_SMU(root_path, seq_len=1024, stride_len=1024,
    #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    # dataset.show_file_info()

    # root_path = './datasets/PHM_dataset/SLIET'
    # print('*'*10, 'SLIET', '*'*10)
    # dataset = Dataset_SLIET(root_path, seq_len=1024, stride_len=1024,
    #              down_sampling_scale=1, start_percentage=0.0, end_percentage=1.0)
    # dataset.show_file_info()


    root_path = './datasets/PHM_dataset/PHM_Challenge2024'
    print('*'*10, 'PHM_Challenge2024', '*'*10)
    dataset = Dataset_Challenge2024(root_path, seq_len=2048, stride_len=128, flag='test_remote',
                    down_sampling_scale=10, start_percentage=0.0, end_percentage=1.0)
    dataset.show_file_info()



    # pass