from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.tools import DataLoader2
from data_provider.data_loader import Dataset_h5, Dataset_PEMS04, Dataset_PEMS08, \
    Dataset_Pretrain_Forecast, Dataset_NYCTaxi, Dataset_Pretrain_PEMS_AlignTask

data_dict = {
    'METRLA': Dataset_PEMS08,
    'PEMSBAY': Dataset_PEMS08,
    'PEMS04': Dataset_PEMS08,
    'PEMS08': Dataset_PEMS08,
    'PEMS03': Dataset_PEMS08,
    'PEMS07': Dataset_PEMS08,
    'Pretrain_Forecast': Dataset_Pretrain_Forecast,
    'Pretrain_Class': Dataset_Pretrain_PEMS_AlignTask,
    'NYCTaxi': Dataset_NYCTaxi,
    'CHIBike': Dataset_NYCTaxi,
    'TDrive': Dataset_NYCTaxi
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    time_to_feature = args.time_to_feature
    if args.use_multi_gpu or flag != 'train':
        shuffle_flag = False
        # drop_last 是 DataLoader 类的一个参数，用于指定在数据集大小
        # 不能被批次大小整除时是否**丢弃最后一个小于批次大小的 batch**。
        drop_last = True
        batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        time_to_feature=time_to_feature
    )
    if args.use_multi_gpu:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,  # 设置了新的 sampler，参数 shuffle 要设置为 False
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=DistributedSampler(data_set)  # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last
        )
    return data_set, data_loader
