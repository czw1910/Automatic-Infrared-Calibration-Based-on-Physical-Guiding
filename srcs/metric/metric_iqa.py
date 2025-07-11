# -----------------------------------------
# 🎯 Image quality assessment metrics used in image/video reconstruction and generation tasks
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyiqa  # pip install pyiqa
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse


# 💡 the inputs and outputs are in 'torch tensor' format

class IQA_Metric(nn.Module):
    """image quality assessment metric calculation using [pyiqa package](https://github.com/chaofengc/IQA-PyTorch)
    Note: use `print(pyiqa.list_models())` to list all available metrics
    """

    def __init__(self, metric_name: str, calc_mean: bool = True):
        super(IQA_Metric, self).__init__()
        self.__name__ = metric_name
        if metric_name == "similarity":
            self.metric = similarity
        elif metric_name == "mse":
            self.metric = calc_mse
        else:
            self.metric = pyiqa.create_metric(metric_name=metric_name)
        self.calc_mean = calc_mean

    def forward(self, output, target):
        with torch.no_grad():
            metric_score = self.metric(output, target)
        if self.calc_mean:
            return torch.mean(metric_score)
        else:
            return metric_score


# 💡 the inputs and outputs are in 'numpy ndarray' format

def calc_psnr(output, target):
    '''
    calculate psnr
    '''
    assert output.shape[0] == target.shape[0]
    total_psnr = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_psnr[k] = compare_psnr(pred, gt, data_range=1)
    return np.mean(total_psnr)


def calc_ssim(output, target):
    '''
    calculate ssim
    '''
    assert output.shape[0] == target.shape[0]
    total_ssim = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_ssim[k] = compare_ssim(
            pred, gt, data_range=1, multichannel=True)
    return np.mean(total_ssim)


def calc_mse(output, target):
    '''
    calculate mse
    '''
    assert output.shape[0] == target.shape[0]
    # print(output.shape,target.shape)
    # output = output.squeeze(0)
    # target = target.squeeze(0)
    # 计算 MSE

    mse = torch.mean(((output*10 - target*10) ** 2)/target/10*100, dim=[1, 2, 3])  # 沿着空间维度计算 MSE
    # mse = torch.mean(((output - target) ** 2), dim=[1, 2, 3])  # 沿着空间维度计算 MSE


    return mse
    output = output.cpu().numpy()
    target = target.cpu().numpy()

    total_psnr = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_psnr[k] = compare_mse(pred, gt)
    print(total_psnr)
    return np.mean(total_psnr)

def similarity(output, target):
    '''
    计算相似度，基本算法为 (abs(output - target) - abs(target)) / target
    '''
    assert output.shape == target.shape, "output 和 target 的形状必须一致"
    
    # 计算相似度

    similarity = torch.abs((output - target)) / target * 100

    #     # 提取单个值
    # output_value = output[:,:,10:11,10:11].item()  # 使用 .item() 提取标量值
    # target_value = target[:,:,10:11,10:11].item()
    # output_value = torch.tensor(output_value)
    # target_value = torch.tensor(target_value)



    # # 打印结果
    # print(f"Extracted output value: {output_value:.6f}")
    # print(f"Extracted target value: {target_value:.6f}")
    # print("similarity Error:", similarity)
    # print(ee)

    
    # 返回相似度的均值
    return torch.mean(similarity)