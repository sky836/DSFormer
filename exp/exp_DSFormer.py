import math
import pickle

import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from datetime import datetime

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, masked_mae, masked_huber_loss
from utils.print_args import print_args
from utils.tools import EarlyStopping, save_trainlog, print_log, WarmupMultiStepLR, WarmupLastMultiStepLR
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_DSFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_DSFormer, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = masked_huber_loss
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, maes, rmses, mapes = [], [], [], []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, batch_y)
                # outputs= self.model(batch_x, batch_y)

                y = batch_y[..., :self.args.output_dim]
                if vali_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes, n_feats = outputs.shape
                    outputs = vali_data.inverse_transform(outputs.reshape(-1, n_nodes, n_feats)).reshape(batch_size,
                                                                                                pred_len, n_nodes, n_feats)

                loss = criterion(outputs, y, 0.0)
                total_loss.append(loss.item())

                mae, rmse, mape = metric(outputs, y, 0.0, self.args.mask_threshold)

                preds.append(outputs)
                trues.append(y)
                maes.append(mae.item())
                rmses.append(rmse.item())
                mapes.append(mape.item())

        maes, rmses, mapes = np.average(maes), np.average(rmses), np.average(mapes)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        self.model.train()
        return total_loss, maes, rmses, mapes, preds, trues

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        log_path = self.args.log_path + setting + '/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # log_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log = os.path.join(log_path, f"{self.args.model}-{self.args.data}-{now}.log")
        log = open(log, "a")
        log.seek(0)
        log.truncate()
        self.log = log

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # path = '/mnt/workspace/'  # 使用阿里天池跑代码的路径
        # path = '/kaggle/working/'  # 使用kaggle跑实验时的路径

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     model_optim,
        #     milestones=[20, 30, 40],
        #     gamma=0.1
        # )
        # 设置warm up的轮次为100次
        scheduler = WarmupMultiStepLR(model_optim, self.args.warmup_epochs, milestones=[20], gamma=0.1)
        # scheduler = WarmupLastMultiStepLR(model_optim, warmup_epochs=[65, 75, 85, 95], milestones=[5, 20, 40,70,90,100,130,150], gamma=0.1)

        criterion = self._select_criterion()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.device == self.args.main_gpu:
            print_args(self.args, log)
            print_log(
                log,
                summary(
                    self.model,
                    [
                        (self.args.batch_size, self.args.seq_len, self.args.num_nodes, self.args.input_dim),
                        (self.args.batch_size, self.args.pred_len, self.args.num_nodes, self.args.input_dim)
                    ],
                    verbose=0,  # avoid print twice
                )
            )
            print_log(log, 'number of params (M): %.2f' % (n_parameters / 1.e6))

            # 获取当前时间
            current_time = datetime.now()
            # 格式化时间输出
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            print_log(log, "The begining of training:"+formatted_time)

        tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        # tensorboard_path = '/mnt/workspace/'  # 使用阿里天池跑实验时的路径
        # tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        if self.device == self.args.main_gpu:
            writer = SummaryWriter(log_dir=tensorboard_path)

        step, best_epoch = 0, -1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs, time_pred = self.model(batch_x, batch_y)
                # outputs = self.model(batch_x, batch_y)

                y = batch_y[..., :self.args.output_dim]

                batch_size, pred_len, n_nodes, n_feats = outputs.shape
                if train_data.scale and self.args.inverse:
                    outputs = train_data.inverse_transform(outputs.reshape(-1, n_nodes, n_feats)).reshape(batch_size,
                                                                                                 pred_len, n_nodes, n_feats)

                loss = criterion(outputs, y, 0.0) + self.args.lamata*criterion(time_pred, batch_y[:, :, 0, self.args.output_dim:])
                # loss = criterion(outputs, y, 0.0)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    if self.device == self.args.main_gpu:
                        print_log(log,
                                  "\tepoch: {1} | iters: {0} | loss: {2:.7f} | speed: {3:.4f}s/iter | left time: {4:.4f}s".
                                  format(i + 1, epoch + 1, loss.item(), speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                model_optim.step()

            if self.device == self.args.main_gpu:
                current_lr = model_optim.param_groups[0]['lr']
                print_log(log, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                print_log(log, "Epoch: {} current lr: {}".format(epoch + 1, current_lr))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_rmse, vali_mape, _, _ = self.vali(vali_data, vali_loader, criterion)
            scheduler.step()  # 学习率调整
            test_loss, test_mae, test_rmse, test_mape, test_preds, test_trues = self.vali(test_data, test_loader, criterion)
            if self.device == self.args.main_gpu:
                print_log(
                    log,
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".
                    format(epoch + 1, train_steps, train_loss, vali_mae)
                )
                print_log(
                    log,
                    "Epoch: {0}, Steps: {1} | Test mae:{2} Test rmse:{3} Test mape: {4}".
                    format(epoch + 1, train_steps, test_mae, test_rmse, test_mape)
                )
                _, pred_len, _, _ = test_preds.shape

                writer.add_scalar(scalar_value=train_loss, global_step=epoch, tag='Loss/train')
                writer.add_scalar(scalar_value=vali_loss, global_step=epoch, tag='Loss/valid')
            
                early_stopping(vali_mae, self.model, path, epoch)
                if early_stopping.early_stop:
                    print_log(log, "Early stopping")
                    print_log(log, "best epoch: {0}".format(early_stopping.best_epoch))
                    break

        # 获取当前时间
        current_time = datetime.now()
        # 格式化时间输出
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print_log(log, "The end of training:"+formatted_time)

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            if self.log == None:
                log_path = self.args.log_path + setting + '/'
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                # log_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
                now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                log = os.path.join(log_path, f"{self.args.model}-{self.args.data}-{now}.log")
                log = open(log, "a")
                log.seek(0)
                log.truncate()
                self.log = log
            print_log(self.log, '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            print_log(self.log, 'loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')), strict=False)
            print_log(self.log, "Model = %s" % str(self.model))
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print_log(self.log, 'number of params (M): %.2f' % (n_parameters / 1.e6))
        preds = []
        trues = []
        x_trues = []
        x_marks = []
        y_marks = []

        maes, rmses, mapes = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs, _ = self.model(batch_x, batch_y)
                # outputs = self.model(batch_x, batch_y)

                y = batch_y[..., :self.args.output_dim]

                if test_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes, n_feats = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, n_nodes, n_feats)).reshape(batch_size,
                                                                                                pred_len, n_nodes, n_feats)

                Mae, Rmse, Mape = [], [], []
                for i in range(n_feats):
                    mae, rmse, mape = metric(outputs[..., i], y[..., i], 0.0, self.args.mask_threshold)
                    Mae.append(mae.item())
                    Rmse.append(rmse.item())
                    Mape.append(mape.item())
                if n_feats > 1:
                    mae, rmse, mape = metric(outputs, y, 0.0, self.args.mask_threshold)
                    Mae.append(mae.item())
                    Rmse.append(rmse.item())
                    Mape.append(mape.item())

                maes.append(Mae)
                rmses.append(Rmse)
                mapes.append(Mape)

                inputs = batch_x[:, :, :, :self.args.output_dim].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    batch_size, seq_len, n_nodes, n_feats = inputs.shape
                    inputs = test_data.inverse_transform(inputs.reshape(-1, n_nodes, n_feats)).reshape(batch_size, seq_len,
                                                                                              n_nodes, n_feats)
                x_trues.append(inputs.astype(np.float32))
                batch_x_mark = batch_x_mark.detach().cpu().numpy()
                batch_y_mark = batch_y_mark.detach().cpu().numpy()
                x_marks.append(batch_x_mark)
                y_marks.append(batch_y_mark)
                preds.append(outputs)
                trues.append(y)

        maes, rmses, mapes = np.array(maes), np.array(rmses), np.array(mapes)
        mae, rmse, mape = np.average(maes, axis=0), np.average(rmses, axis=0), np.average(mapes, axis=0)
        for i in range(self.args.output_dim):
            print_log(self.log, 'rmse:{:.7f}, mae:{:.7f}, mape:{:.7f}'.format(rmse[i], mae[i], mape[i]))

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        x_trues = np.array(x_trues).astype(np.float32)
        x_marks = np.array(x_marks)
        y_marks = np.array(y_marks)
        maes = np.array(maes)
        rmses = np.array(rmses)
        mapes = np.array(mapes)
        print_log(self.log, 'test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, rmses.shape, mapes.shape)
        x_trues = x_trues.reshape((-1, x_trues.shape[-2], x_trues.shape[-1]))
        x_marks = x_marks.reshape((-1, x_marks.shape[-2], x_marks.shape[-1]))
        y_marks = y_marks.reshape((-1, y_marks.shape[-2], y_marks.shape[-1]))
        print_log(self.log, 'test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, rmses.shape, mapes.shape)

        np.save(folder_path + 'metrics.npy', np.array([mae, rmse, mape]))
        np.save(folder_path + 'pred.npy', preds.detach().cpu().numpy())
        np.save(folder_path + 'true.npy', trues.detach().cpu().numpy())
        np.save(folder_path + 'x_trues.npy', x_trues)
        np.save(folder_path + 'x_marks.npy', x_marks)
        np.save(folder_path + 'y_marks.npy', y_marks)
        np.save(folder_path + 'maes.npy', maes)
        np.save(folder_path + 'rmses.npy', rmses)
        np.save(folder_path + 'mapes.npy', mapes)
