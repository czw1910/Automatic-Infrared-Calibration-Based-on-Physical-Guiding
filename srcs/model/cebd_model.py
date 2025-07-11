import torch.nn as nn
from srcs.model.bd_model import BDNeRV_RC
from srcs.model.ce_model import CEBlurNet

class CEBDNet(nn.Module):
    '''
    coded exposure blur decomposition network
    '''
    def __init__(self, sigma_range=0, test_sigma_range=0, ce_code_n=8, frame_n=8, ce_code_init=None, opt_cecode=False, ce_net=None, binary_fc=None, bd_net=None):
        super(CEBDNet, self).__init__()
        self.ce_code_n = ce_code_n
        self.frame_n = frame_n
        self.bd_net = bd_net
        # coded exposure blur net
        if ce_net == 'CEBlurNet':
            self.BlurNet = CEBlurNet(
                sigma_range=sigma_range, test_sigma_range=test_sigma_range, ce_code_n=ce_code_n, frame_n=frame_n, ce_code_init=ce_code_init, opt_cecode=opt_cecode, binary_fc=binary_fc)
        else:
            raise NotImplementedError(f'No model named {ce_net}')

        # blur decomposition net
        if bd_net=='BDNeRV_RC':
            self.DeBlurNet = BDNeRV_RC()
        else:
            raise NotImplementedError(f'No model named {bd_net}')
        

    def forward(self, frames,time_idx,temp_huanjing,components):
    #         # 获取第一个样本的索引
    # first_sample_idx = frames.dataset.indices[0]

    # # 通过原始 dataset 访问 temperature_idx 和 time_idx
    # print("Temperature idx of first sample:", frames.dataset.temperature_idx[first_sample_idx])
    # print("Time idx of first sample:", frames.dataset.time_idx[first_sample_idx])
    # print(eee)
        time_idx, ce_code_up, ce_blur_img = self.BlurNet(
            frames,time_idx)
        
        output = self.DeBlurNet(ce_blur=ce_blur_img, time_idx=time_idx, ce_code=ce_code_up,temp_huanjing = temp_huanjing,components=components)
        return output, ce_blur_img

    # def forward(self, frames,temperature_indices, time_indices):
    #     #         # 获取第一个样本的索引
    #     # first_sample_idx = frames.dataset.indices[0]

    #     # # 通过原始 dataset 访问 temperature_idx 和 time_idx
    #     # print("Temperature idx of first sample:", frames.dataset.temperature_idx[first_sample_idx])
    #     # print("Time idx of first sample:", frames.dataset.time_idx[first_sample_idx])
    #     # print(eee)
    #     ce_blur_img_noisy, temperature_indices, time_indices, ce_code_up, ce_blur_img = self.BlurNet(
    #         frames,temperature_indices, time_indices)


    #     output = self.DeBlurNet(ce_blur=ce_blur_img_noisy, temperature_indices=temperature_indices,time_indices=time_indices, ce_code=ce_code_up)
    #     return output, ce_blur_img, ce_blur_img_noisy
