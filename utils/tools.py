import os
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# 将[2019, 8, 1, 1]类型转换为datetime类型2019-08-01-01
def to_datetime(time_mark):
    batch_size = time_mark.shape[0]
    datetime_array = []
    for i in range(batch_size):
        # 对每一行的数字列表构造 datetime 对象
        Tmp = time_mark[i].to(int)
        datetime_list = [datetime(*timearray).strftime('%Y-%m-%d-%H') for timearray in Tmp]
        # 将构造的 datetime 对象添加到 datetime_array 中
        datetime_array.append(datetime_list)

    return np.array(datetime_array)


def save_trainlog(path, log):
    print(log)
    path = path + 'train_log.txt'
    with open(path, "a") as f:
        f.write(log + '\n')


def print_log(log=None, *values, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = -1

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, loss, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.text(0.01, 20.05, 'Loss:{0}'.format(loss))
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 在warmup阶段，学习率线性增长
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 在warmup阶段结束后，使用MultiStepLR策略
            return [base_lr * (self.gamma ** self._get_step_count()) for base_lr in self.base_lrs]

    def _get_step_count(self):
        count = 0
        for milestone in self.milestones:
            if self.last_epoch >= milestone:
                count += 1
        return count


class WarmupLastMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, beta=2, last_epoch=-1):
        self.warmup_epochs = warmup_epochs  # 列表类型
        self.milestones = milestones
        self.gamma = gamma
        self.beta = beta
        super(WarmupLastMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs[0]:
            return [base_lr * (self.gamma ** self._get_step_count()) for base_lr in self.base_lrs]
        else:
            # 在warmup last阶段，学习率线性增长
            lrs = []
            for base_lr in self.base_lrs:
                lr = base_lr * (self.gamma ** self._get_step_count())
                lr = lr * (self.beta * self._get_step_count_last())
                lrs.append(lr)
            return lrs

    def _get_step_count(self):
        count = 0
        for milestone in self.milestones:
            if self.last_epoch >= milestone:
                count += 1
        return count

    def _get_step_count_last(self):
        count = 0
        for warmup_epoch in self.warmup_epochs:
            if warmup_epoch <= self.last_epoch <= self.warmup_epochs[-1]:
                count += 1
        return count

class DataLoader2(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()
