# Part of the file is from https://github.com/thuml/Time-Series-Library/blob/main/utils/losses.py
"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn as nn


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class UnifiedMaskRecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_mim_loss(self, target, pred, pad_mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per step,mean on dimension D

        combined_mask = pad_mask.bool()

        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss
    
    def foward_full_loss(self, target, pred):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum()
        return loss

    def forward(self, outputs, target, pad_mask):
        student_cls, student_fore, _ = outputs

        mask_loss = self.forward_mim_loss(target, student_fore, pad_mask)

        if student_cls is not None:
            cls_loss = self.forward_mim_loss(target, student_cls, pad_mask)
        else:
            cls_loss = 0.0 * mask_loss

        total_loss = dict(cls_loss=cls_loss,
                          mask_loss=mask_loss, loss=mask_loss+cls_loss)
        return total_loss


class FreqTimeMaskRecLoss(nn.Module):
    def __init__(self,alpha =1.0,lamda=0.5):
        ##alpha: the weight of the time loss and frequency loss
        ##lamda: the weight of the frequency loss of amplitude and phase
        super().__init__()
        self.lamda = lamda
        self.alpha = alpha

    def forward_mim_loss(self, target, pred, pad_mask):
        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per step,mean on dimension D
        loss_feq = (t.fft.rfft(pred, dim=1) - t.fft.rfft(target, dim=1)).abs().mean()

        combined_mask = pad_mask.bool()

        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss
    
    # def batch_std(self, x):
    #     mean = x.mean(dim=(0,1), keepdim=True) #在batch和时间维度上求均值，得到channel的均值
    #     std = x.std(dim=(0,1), keepdim=True)
    #     x = (x-mean)/std
    #     return x

    # # def freq_loss(self,pred,target):
    # #     fft_target = t.fft.fft(target,dim=1)
    # #     amplitude_pred = self.batch_std(pred[1])
    # #     amplitude_target = self.batch_std(t.abs(fft_target))
    # #     phase_pred =  self.batch_std(pred[0])
    # #     phase_target = self.batch_std(t.angle(fft_target))
    # #     #检查一下为什么会出错
    # #     _,len,_ = target.shape
    # #     loss_amp = (amplitude_pred[:,:len,:] - amplitude_target) ** 2
    # #     loss_pha = (phase_pred[:,:len,:] - phase_target) ** 2
    # #     loss = self.lamda*loss_amp +(1-self.lamda)*loss_pha
    # #     loss = loss.mean()
    # #     return loss

    def forward(self, outputs, target):
        mask_out, _, pad_mask = outputs

        mask_loss = self.forward_mim_loss(target, mask_out, pad_mask)

        # # if cls_out is not None:
        # #     cls_out = self.forward_mim_loss(target, cls_out, pad_mask)
        # # else:
        # #     cls_out = t.tensor(0.0)
        # freq_loss = self.freq_loss(cls_out,target)
        # loss = self.alpha*mask_loss + (1-self.alpha)*freq_loss
        total_loss = dict(cls_loss=mask_loss,
                          freq_loss=mask_loss, loss=mask_loss)
        return total_loss



#测试函数
if  __name__ == '__main__':
    #测试loss函数
    loss = FreqTimeMaskRecLoss()
    target = t.rand(32,224, 111)
    pred = t.rand(32,224, 111)
    pad_mask = t.ones(32, 224)
    # print(loss.freq_loss([pred,pred], target))
    print(loss.forward([pred,pred,pred], target))