import argparse
from datetime import datetime
import os
import torch
from torch.distributed import init_process_group, destroy_process_group

from exp.exp_DSFormer import Exp_DSFormer
# from utils.print_args import print_args
import random
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 设置环境变量以获取更多调试信息
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

if __name__ == '__main__':
    seed = torch.randint(10000, (1,))  # set random seed here
    seed = 2024
    print('seed:', seed)
    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Taformer')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='DSFormer',
                        help='task name, options:[DSFormer]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='train or test')
    parser.add_argument('--model', type=str, required=False, default='DSFormer',
                        help='model name, options: [DSFormer]')

    # path to modify
    # 1. data and adj
    parser.add_argument('--adj_path', type=str, default=r'datasets/PEMS08/pems08adj.pkl', help='path of the adjmx')
    parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='PEMS08/data.npz', help='data file')
    parser.add_argument('--data', type=str, required=False, default='PEMS08', help='dataset type, [Pretrain_Forecast, Pretrain_Class]')
    parser.add_argument('--num_nodes', type=int, required=False, default=170, help='the nodes of dataset')
    parser.add_argument('--steps_per_day', type=int, default=288, help='')
    parser.add_argument('--mask_threshold', type=int, default=0, help='')
    parser.add_argument('--input_dim', type=int, default=3, help='')
    parser.add_argument('--output_dim', type=int, default=1, help='')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--patch_size', type=int, default=1, help='The size of one patch')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--clip', type=int, default=None, help='clip grad')


    # 2. model path
    parser.add_argument('--best_model_path', type=str, default='checkpoints/stTrans_seqlen6_NYtaxi/checkpoint.pth', help='the path of pretrain model')
    # finetune task
    parser.add_argument('--is_finetune', type=bool, default=False, help='if use pretrain model to finetune')

    # forecasting task
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

    # pretrain task
    parser.add_argument('--mode', type=str, default='predict', help='choose the mode of model. options: [pretrain, predict]')
    parser.add_argument('--gcn_bool', type=bool, default=True, help='if use GCN in model or not')
    parser.add_argument('--addaptadj', type=bool, default=True, help='if use adaptive adjacency matrices in GCN or not')

    # model define
    parser.add_argument('--feed_forward_dim', type=int, default=256, help='')
    parser.add_argument('--pos_dim', type=int, default=32, help='')
    parser.add_argument('--spatial_embedding_dim', type=int, default=48, help='')
    parser.add_argument('--dow_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--tod_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--input_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--init_d', type=int, default=8, help='')
    parser.add_argument('--num_layers', type=int, default=3, help='')
    parser.add_argument('--use_mixed_proj', type=bool, default=True, help='')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--label_patch_size', type=int, default=1, help='The size of one  decoder input patch')
    parser.add_argument('--pretrain_layers', type=int, default=1, help='num of pretrain decoder layers')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio of pretrain')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0003, help='optimizer weight_decay')
    parser.add_argument('--lamata', type=float, default=1, help='')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='masked_huberLoss',
                        help='loss function, options:[masked_mae, masked_huberLoss]')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', type=bool, help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,2,3', help='device ids of multile gpus')

    # data loader
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--log_path', type=str, default='./myTrain_logs/', help='location of train logs')
    parser.add_argument('--time_to_feature', type=int, default=0,
                        help='Adding time features to the data. options: [0, 1], 0 stands for 2 features. 1 stands for 4 features')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.main_gpu = args.gpu

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print_args(args)

    if args.task_name == 'DSFormer':
        Exp = Exp_DSFormer
    else:
        Exp = Exp_DSFormer

    # 获取当前日期和时间
    current_datetime = datetime.now()

    # 格式化为字符串
    formatted_string = current_datetime.strftime("%Y_%m_%d_%H_%M")

    if args.is_training:
        for ii in range(args.itr):
            if args.use_multi_gpu:
                init_process_group(backend="nccl")
                torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            # setting record of experiments
            exp = Exp(args)
            setting = '{}_{}_sl{}_pl{}_l{}_h{}_{}_{}'.format(
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.num_layers,
                args.n_heads,
                formatted_string,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if args.task_name != 'Pretrain':
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            torch.cuda.empty_cache()
            if args.use_multi_gpu:
                # 销毁进程池
                destroy_process_group()
    else:
        ii = 0
        setting = '{}_{}_sl{}_pl{}_l{}_h{}_{}_{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.num_layers,
            args.n_heads,
            formatted_string,
            ii)

        # setting = 'stTrans_seqlen6_NYtaxi'

        exp = Exp(args)
        if args.task_name != 'Pretrain':
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
