import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM  # pip install pytorch-msssim

# ===========================
# global loss info extract
# ===========================
LOSSES = {}

def add2loss(cls):
    if cls.__name__ in LOSSES:
        raise ValueError(f'{cls.__name__} is already in the LOSSES list')
    else:
        LOSSES[cls.__name__] = cls
    return cls

# ===========================
# weighted_loss
# ===========================


class WeightedLoss(nn.Module):
    """
    weighted multi-loss
    loss_conf_dict: {loss_type1: weight|[weight,{kwargs_dict_for_init}], ...}
        eg: loss_conf_dict = {'CharbonnierLoss':0.5, 'EdgeLoss':0.5}
        eg: loss_conf_dict = {'CharbonnierLoss':[0.5, {'eps':1e-3}], 'EdgeLoss':0.5}
    """

    def __init__(self, loss_conf_dict):
        super(WeightedLoss, self).__init__()
        self.loss_conf_dict = loss_conf_dict

        # instantiate classes
        self.losses = []
        for k, v in loss_conf_dict.items():
            if isinstance(v, (float, int)):
                assert v >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](), 'weight': v})
            elif isinstance(v, list) and len(v) == 2:
                assert v[0] >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](**v[1]), 'weight': v[0]})
            else:
                raise ValueError(
                    f"the Key({k})'s Value {v} in Dict(loss_conf_dict) should be scalar(weight) | list[weight, args] ")

    def forward(self, output, target):
        loss_v = 0
        for loss in self.losses:
            loss_v += loss['cls'](output, target)*loss['weight']


        return loss_v

# ===========================
# basic_loss
# ===========================


@add2loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):

        diff = output.to('cuda:0') - target.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


@add2loss
class L1Loss(nn.Module):
    """Mean Square Error Loss (L2)"""

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, target):
        return F.l1_loss(output, target)


@add2loss
class L2Loss(nn.Module):
    """Mean Square Error Loss (L2)"""

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)

@add2loss
class MSELoss(nn.Module):
    """Mean Square Error Loss (L2)"""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        # print(target)
        # print(target[0,0:1,0:1,0:1])
        # print(target[1,0:1,0:1,0:1])
        # print(F.mse_loss(output[0,:,:,:], target[0,:,:,:]))
        # print(F.mse_loss(output[1,:,:,:], target[1,:,:,:]))
        # print(F.mse_loss(output, target))
        # target_mean = torch.mean(target)
        loss1 = F.mse_loss(output[0,:,:,:], target[0,:,:,:]) / (target[0,0:1,0:1,0:1])
        loss2 = F.mse_loss(output[1,:,:,:], target[1,:,:,:]) / (target[1,0:1,0:1,0:1])
        loss = (loss1+loss2)/2

        return loss.mean()


@add2loss
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@add2loss
class SSIMLoss(SSIM):
    """Structural Similarity Index Measure Loss
    Directly use the SSIM class provided by pytorch_msssim
    """
    pass


@add2loss
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)

        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')

        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, output, target):

        loss = self.loss(self.laplacian_kernel(output.to('cuda:0')),
                         self.laplacian_kernel(target.to('cuda:0')))
        return loss


@add2loss
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, output, target):
        diff = torch.fft.fft2(output.to('cuda:0')) - \
            torch.fft.fft2(target.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


@add2loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, output, *args):

        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        count_h = self._tensor_size(output[:, :, 1:, :])
        count_w = self._tensor_size(output[:, :, :, 1:])
        h_tv = torch.pow((output[:, :, 1:, :]-output[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((output[:, :, :, 1:]-output[:, :, :, :w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size


@add2loss
class PixelFocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, threshold=0.7, epsilon=1e-6):
        super().__init__()
        self.gamma = gamma        # 聚焦强度（建议2-5）
        self.threshold = threshold # 低像素阈值（需根据数据分布调整）
        self.epsilon = epsilon    # 防止数值不稳定

    def forward(self, pred, target):
        # 基础L1损失
        l1_loss = torch.abs(pred - target)
        
        # 计算动态权重（聚焦低像素区域）
        pixel_weights = 1.0 - torch.exp(-target * self.gamma)
        
        # 增强阈值以下区域的权重
        mask = (target < self.threshold).float()
        enhanced_weights = 1.0 + mask * pixel_weights
        
        # 组合加权损失
        focal_loss = enhanced_weights * l1_loss
        return focal_loss.mean()


@add2loss
class AdaptiveFocalTversky(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=2):
        super().__init__()
        self.alpha = alpha  # FP惩罚系数
        self.beta = beta    # FN惩罚系数
        self.gamma = gamma  # 高像素聚焦强度

    def forward(self, pred, target):
        # 计算基础指标
        tp = (pred * target).sum(dim=(1,2,3))      # True Positive
        fp = (pred * (1-target)).sum(dim=(1,2,3))  # False Positive
        fn = ((1-pred) * target).sum(dim=(1,2,3))  # False Negative
        
        # Tversky指数计算
        tversky = tp / (tp + self.alpha*fp + self.beta*fn + 1e-6)
        
        # 动态权重（高像素区域增强）
        high_pixel_weight = torch.exp(target.mean(dim=(1,2,3)) * self.gamma)
        
        # 组合损失
        loss = (1 - tversky) * high_pixel_weight
        return loss.mean()


@add2loss
class HighPixelPenaltyLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=2.0, threshold=0.35):
        super().__init__()
        self.gamma = gamma    # 高亮区域聚焦强度（建议2-5）
        self.alpha = alpha    # 预测不足惩罚系数（建议1-3）
        self.threshold = threshold

    def forward(self, pred, target):
        # 基础L1损失（参考网页3的回归基础）
        l1_loss = torch.abs(pred - target)
        
        # 双条件惩罚掩码（融合网页1的检测思想）
        high_mask = (target > self.threshold).float()  # 目标高亮区域
        under_mask = (pred < target).float()           # 预测不足区域
        penalty_mask = high_mask * under_mask          # 交叉惩罚区域
        
        # 指数型空间权重（替代动态卷积，参考网页5的分布建模）
        spatial_weights = torch.exp(target * self.gamma)  # 亮度越高权重越大
        
        # 复合损失计算（结合网页2的Focal机制）
        enhanced_loss = (
            (1 + self.alpha * penalty_mask) *  # 线性惩罚项
            spatial_weights * l1_loss          # 空间权重调整
        )
        return enhanced_loss.mean()



if __name__ == "__main__":
    import PerceptualLoss
    output = torch.randn(4, 3, 10, 10)
    target = torch.randn(4, 3, 10, 10)
    loss_conf_dict = {'CharbonnierLoss': 0.5, 'fftLoss': 0.5}

    Weighted_Loss = WeightedLoss(loss_conf_dict)
    loss_v = Weighted_Loss(output, target)
    # print('loss: ', loss_v)
