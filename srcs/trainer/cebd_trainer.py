import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import platform
from omegaconf import OmegaConf
from .base import BaseTrainer
from srcs.utils.util import collect, instantiate, get_logger
from srcs.logger import BatchMetrics
import os
import torch
import cv2
import numpy as np
import torchvision.utils as vutils
#======================================
# Trainer: modify '_train_epoch'
#======================================



def load_temperature_docs():
    """加载四组环境温度对应的温度表"""
    docs = [
        {   # 文档1对应环境温度索引0
            '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 40, 60, 80],
            '主镜温度': [5.0, 4.8, 4.5, 4.3, 4.1, 4.0, 3.8, 3.6, 3.5, 3.4],
            '分色镜温度': [6.0, 5.8, 5.6, 5.5, 5.2, 5.1, 5.0, 4.8, 4.7, 4.6],
            '系统温度': [9.0, 8.8, 8.6, 8.4, 8.2, 8.1, 8.0, 7.9, 7.8, 7.7],
            '相机温度': [7.3, 7.1, 7.0, 6.7, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0],
            '芯轴温度': [4.6, 4.3, 4.1, 3.8, 3.6, 3.5, 3.3, 3.3, 3.1, 3.0]
        },
        {   # 文档2对应环境温度索引10
            '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 30, 40, 60, 80],
            '主镜温度': [13.8, 13.7, 13.6, 13.5, 13.3, 13.3, 13.2, 13.2, 13.1, 13.0, 13.0],
            '分色镜温度': [13.9, 14.0, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.2, 14.3, 14.2],
            '系统温度': [16.4, 16.8, 17.1, 17.2, 17.3, 17.3, 17.3, 17.4, 17.4, 17.4, 17.4],
            '相机温度': [14.8, 15.1, 15.4, 15.5, 15.6, 15.6, 15.6, 15.6, 15.4, 15.7, 15.7],
            '芯轴温度': [13.2, 13.2, 13.1, 13.0, 12.8, 12.8, 12.8, 12.9, 12.8, 12.7, 12.7]
        },
        {   # 文档3对应环境温度索引20
            '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 40, 60],
            '主镜温度': [15.5, 16.0, 16.5, 16.9, 17.3, 17.6, 17.8, 18.1, 18.1],
            '分色镜温度': [15.9, 16.8, 17.7, 18.3, 19.0, 19.3, 19.6, 20.0, 20.0],
            '系统温度': [19.1, 20.5, 21.5, 22.1, 22.7, 23.1, 23.3, 23.7, 23.6],
            '相机温度': [17.6, 18.9, 19.8, 20.4, 21.0, 21.3, 21.6, 22.0, 22.0],
            '芯轴温度': [17.6, 18.0, 18.3, 18.6, 18.8, 19.1, 19.1, 19.4, 19.6]
        },
        {   # 文档4对应环境温度索引35
            '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 30, 40, 60, 80],
            '主镜温度': [31.0, 31.4, 31.8, 32.1, 32.5, 32.8, 33.0, 33.2, 33.5, 33.8, 34.0],
            '分色镜温度': [31.9, 32.8, 33.5, 34.0, 34.4, 34.8, 35.1, 35.4, 35.6, 36.0, 36.3],
            '系统温度': [35.1, 36.5, 37.2, 37.8, 38.1, 38.5, 38.8, 39.0, 39.3, 39.6, 39.8],
            '相机温度': [33.5, 34.7, 35.5, 36.0, 36.5, 36.8, 37.0, 37.3, 37.5, 37.8, 38.1],
            '芯轴温度': [32.8, 33.1, 33.4, 33.6, 33.8, 34.0, 34.1, 34.3, 34.5, 34.6, 34.8]
        }
    ]
    return docs


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.limit_train_iters = config['trainer'].get(
            'limit_train_iters', len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config['trainer'].get(
            'limit_valid_iters', len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)
        self.log_weight = config['trainer'].get('log_weight', False)
        args = ['loss', *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(
            *args, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics(
            *args, postfix='/valid', writer=self.writer)
        self.losses = self.config['loss']
        self.temperature_docs = load_temperature_docs()
        self.env_temp_to_index = {
            0: 0,   # 文档1对应环境温度0
            10: 1,  # 文档2对应环境温度10
            20: 2,  # 文档3对应环境温度20
            35: 3   # 文档4对应环境温度35
        }

    def get_temperatures(self, env_idx, bb_temp):
        """
        根据环境温度索引和黑体温度获取各部件温度
        :param env_idx: 环境温度索引（0-3）
        :param bb_temp: 黑体温度值
        :return: 字典包含各部件温度
        """
        # 索引有效性检查
        if env_idx < 0 or env_idx >= len(self.temperature_docs):
            raise ValueError(f"无效的环境温度索引: {env_idx}，有效值为0-3")
        
        doc = self.temperature_docs[env_idx]
        bb_list = doc['黑体温度']
        
        # 找到最接近的黑体温度索引
        closest_idx = min(
            range(len(bb_list)),
            key=lambda i: abs(bb_list[i] - bb_temp)
        )
        
        return {
            '主镜温度': doc['主镜温度'][closest_idx],
            '分色镜温度': doc['分色镜温度'][closest_idx],
            '系统温度': doc['系统温度'][closest_idx],
            '相机温度': doc['相机温度'][closest_idx],
            '芯轴温度': doc['芯轴温度'][closest_idx]
        }

    def _ce_reblur(self, output):
        # frame_n should equal to ce_code_n cases
        ce_weight = self.model.BlurNet.ce_weight.detach().squeeze()
        ce_code = ((torch.sign(ce_weight)+1)/2)
        ce_code_ = torch.tensor(ce_code).view(1, -1, 1, 1, 1)
        ce_output = torch.sum(torch.mul(output, ce_code_), dim=1)/len(ce_code)
        return ce_output
    
    def _after_iter(self, epoch, batch_idx, phase, loss, metrics, image_tensors: dict):
        # hook after iter
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}')

        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f'{phase}_metrics').update('loss', loss_v)

        for k, v in metrics.items():
            getattr(self, f'{phase}_metrics').update(k, v.item()) # `v` is a torch tensor

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, vid in enumerate(self.data_loader):
            # 数据解析
            temperature_indices = vid[1]
            time_indices = vid[2]
            temp_huanjing_indices = vid[3].long()  # 确保为整型
            vid = vid[0].to(self.device).float() / 16383

            # 验证环境温度值有效性
            # valid_env_values = [0, 10, 20, 35]
            # invalid_mask = ~torch.isin(temp_huanjing_indices, torch.tensor(valid_env_values, device=self.device))
            # if invalid_mask.any():
            #     invalid_values = temp_huanjing_indices[invalid_mask].cpu().numpy()
            #     raise ValueError(f"无效环境温度值: {np.unique(invalid_values)}，应为0/10/20/35")

            # 转换到CPU处理
            env_indices = temp_huanjing_indices.cpu().numpy()
            bb_temps = temperature_indices.cpu().numpy()

            # 批量获取温度参数
            component_dict = {
                'zhujing': [], 'fensejing': [], 'xitong': [],
                'xiangji': [], 'xinzhou': []
            }
            
            # 批量处理环境温度和黑体温度
            for env_val, bb_temp in zip(env_indices, bb_temps):
                # 将环境温度值转换为文档索引
                env_val_python = env_val.item()  # 提取为 Python int
                env_idx = self.env_temp_to_index.get(env_val_python)
                if env_idx is None:
                    raise ValueError(f"无效的环境温度值: {env_val}，应为0/10/20/35")
                # 获取部件温度
                temps = self.get_temperatures(env_idx, bb_temp)
                component_dict['zhujing'].append(temps['主镜温度'])
                component_dict['fensejing'].append(temps['分色镜温度'])
                component_dict['xitong'].append(temps['系统温度'])
                component_dict['xiangji'].append(temps['相机温度'])
                component_dict['xinzhou'].append(temps['芯轴温度'])
            
            # 转换为张量并调整维度
            components = [
                torch.tensor(component_dict[k], device=self.device).float().unsqueeze(1)
                for k in ['zhujing', 'fensejing', 'xitong', 'xiangji', 'xinzhou']
        ]


            # # 调试打印
            # print("输入维度检查:")
            # print(f"vid: {vid.shape}")
            # print(f"time_indices: {time_indices.unsqueeze(1).shape}")
            # print(f"env_indices: {temp_huanjing_indices.unsqueeze(1).shape}")
            # for name, comp in zip(['主镜', '分色镜', '系统', '相机', '芯轴'], components):
            #     print(f"{name}温度: {comp.shape}")
            # print("输入检查:")
            # print(f"vid: {vid.shape}")
            # print(f"time_indices: {time_indices.unsqueeze(1)}")
            # print(f"env_indices: {temp_huanjing_indices.unsqueeze(1)}")
            # print(f"temperature_indices:{temperature_indices.unsqueeze(1)}")
            # for name, comp in zip(['主镜', '分色镜', '系统', '相机', '芯轴'], components):
            #     print(f"{name}温度: {comp}")
            
            
            components_tensor = torch.cat(components, dim=1)  # 合并5个温度参数为[B,5]

             # 假设 txt 文件路径
            txt_file_path = "/opt/data/private/czw/207/BDINR-master/3.70-4.80radian.txt"

            # 读取 txt 文件中的数据
            data = np.loadtxt(txt_file_path)  # 假设文件格式为两列：温度（K氏温度）和辐射亮度值
            temperature_k = data[:, 0]  # 第一列是温度（K氏温度）
            radiance_values = data[:, 1]  # 第二列是辐射亮度值
             # 将 temperature_k 转换为 Tensor
            temperature_k = torch.from_numpy(temperature_k).to(self.device)
            # temperature_k = torch.from_numpy(temperature_k)

            # 将 temperature_indices 从摄氏温度转换为 K 氏温度
            temperature_indices_k = temperature_indices + 273.15

            # 根据 temperature_indices_k 找到对应的辐射亮度值
            target_radiance = []
            for idx in temperature_indices_k:


                # 找到最接近的温度值
                temperature_k = temperature_k.to(idx.device)

                # 使用 Tensor 操作
                closest_idx = torch.argmin(torch.abs(temperature_k - idx))
                target_radiance.append(radiance_values[closest_idx]/10)

            # 将辐射亮度值转换为张量
            target_radiance = torch.tensor(target_radiance, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # 扩展张量到目标形状 [1, 1, 3, 512, 640]
            # 步骤 1：调整为 [8,1,1,1]
            target_radiance = target_radiance.permute(3, 0, 1, 2)

            # 步骤 2：调整为 [8,1,3,1,1]
            target_radiance = target_radiance.unsqueeze(2).repeat(1, 1, 3, 1, 1)

            # 步骤 3：调整为 [8,1,3,511,640]
            # target_radiance = target_radiance.expand(8, 1, 3, 511, 640)
            target = target_radiance.expand(-1, 1, 3, 80, 80).to(self.device)
            # 调用模型
            output, data = self.model(
                vid,
                time_indices,
                temp_huanjing_indices,
                components_tensor
            )

            # print(vid.shape,output.shape,time_indices.shape,temp_huanjing_indices.shape)
            # print(vid)
            # print(data)
            # print(ee)
            # print("vid:",      vid[:,:,0:1,20:25,20:25])
            # print("output:",output[:,:,0:1,20:25,20:25])
            # print("target:",target[:,:,0:1,20:30,25:35])
            # print(time_indices,temperature_indices)
            # print(output.shape)
            # print(target.shape)
            output = output.float()
            target = target.float()

            output = torch.mean(output, dim=[3,4], keepdim=True)

            output_ = torch.flatten(output, end_dim=1)
            target_ = torch.flatten(target, end_dim=1)
            # print(target_)
            # 对 output_ 和 target_ 取均值，使其形状变为 1x1 的张量
            # output_mean = torch.mean(output_, dim=[1,2,3], keepdim=True)
            # print(output_mean)
            # target_mean = torch.mean(target_)
            # print(target_.shape)

            # 恢复形状
            output_ = torch.broadcast_to(output_, (-1, 3, 80, 80))
            
            # target_ = torch.full(target_.shape, target_mean.item(), device=target.device)
            # print(target_,target_mean.item())

            # if(epoch==2):
            #     for i in range(vid.shape[0]):
            #         # 提取第 i 个张量
            #         vid_i = vid[i,:,:,:,:]  # 形状为 [1, 1, 511, 640]

            #         # 调整张量的形状
            #         vid_ = vid_i.squeeze()  # 去掉多余的维度，形状变为 [511, 640]

            #         # # 确保张量的值在 [0, 255] 范围内
            #         # vid_ = (vid_ * 255).to(torch.uint8)

            #         # 使用 torchvision.utils.save_image 保存图片
            #         if len(vid_.shape) == 2:
            #             vid_ = vid_.unsqueeze(0).unsqueeze(0)  # 调整为 [1, 1, 511, 640]

            #         vutils.save_image(vid_i, f'/opt/data/private/czw/207/BDINR-master/vid{epoch}_{i+1}.png')
            #         print("vid:",time_indices,temp_huanjing_indices,temperature_indices_k)
            #         print("output_:",output_[:,0:1,10:11,10:11])
            #         print("target_",target_[:,0:1,10:11,10:11])
            #     print(e)
            


            # print(temperature_indices,output_,target_,output_.shape,target_.shape)
            # print(temperature_indices)
            # main loss
            loss = self.losses['main_loss'] * \
                self.criterion['main_loss'](output_, target_)
            
            # reblur loss: frame_n should equal to ce_code_n cases
            if 'reblur_loss' in self.losses:
                ce_output = self._ce_reblur(output)
                loss = loss + self.losses['reblur_loss'] * \
                    self.criterion['reblur_loss'](ce_output, data)
            # print(loss)
            # print(ee)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_metrics.update('loss', loss.item())

            # iter record
            if batch_idx % self.logging_step == 0 or (batch_idx+1) == self.limit_train_iters:
                # loss
                loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
                self.writer.set_step(
                    (epoch - 1) * self.limit_train_iters + batch_idx, speed_chk='train')
                
                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output_.shape,target_.shape)
                        # print("output_",output_[:,0:1,10:11,10:11],"target_",target_[:,0:1,10:11,10:11])
                        # print(met)
                        metric_v = met(output_, target_)
                        # print(metric_v)

                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': vid[0, ...], 'output': output[0,0::frame_num//1, ...], 'target': target[0,0::frame_num//1, ...]}
                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'train',
                                 loss, iter_metrics, {})
                # iter log
                self.logger.info(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]["lr"]:.3e}')
            

            if (batch_idx+1) == self.limit_train_iters:
                # save demo images to tensorboard after trainig epoch
                self.writer.set_step(epoch)
                for k, v in image_tensors.items():
                    self.writer.add_image(
                        f'train/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                break
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        interp_scale = self.model.frame_n//self.model.ce_code_n
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, vid in enumerate(self.valid_data_loader):
                # vid = vid.to(self.device).float()/255

                # # 获取当前批次的样本索引
                # start_idx = batch_idx * self.data_loader.batch_size
                # end_idx = start_idx + vid.size(0)  # vid.size(0) 是当前批次的实际大小

                # # 提取当前批次的样本索引
                # batch_indices = self.data_loader.dataset.indices[start_idx:end_idx]

                #  # 通过索引获取 Temperature idx 和 Time idx
                # temp_huanjing_indices = torch.tensor([self.data_loader.dataset.dataset.temp_huanjing_idx[idx] for idx in batch_indices], device=self.device).to(dtype=torch.float32)
                # time_indices = torch.tensor([self.data_loader.dataset.dataset.time_idx[idx] for idx in batch_indices], device=self.device)
                # temperature_indices = torch.tensor([self.data_loader.dataset.dataset.temperature_idx[idx] for idx in batch_indices], device=self.device).to(dtype=torch.float32)

                # print(f"Epoch {epoch}, Batch {batch_idx}: Indices range from {start_idx} to {end_idx}")


                temperature_indices = vid[1]
                time_indices = vid[2]
                temp_huanjing_indices = vid[3]
                vid = vid[0]


                vid = vid.to(self.device).float() / 16383
                temperature_indices = temperature_indices.to(self.device).float()
                time_indices = time_indices.to(self.device).float() 
                temp_huanjing_indices = temp_huanjing_indices.to(self.device).float() 

              # 转换到CPU处理
                env_indices = temp_huanjing_indices.cpu().numpy()
                bb_temps = temperature_indices.cpu().numpy()

                # 批量获取温度参数
                component_dict = {
                    'zhujing': [], 'fensejing': [], 'xitong': [],
                    'xiangji': [], 'xinzhou': []
                }
                
                # 批量处理环境温度和黑体温度
                for env_val, bb_temp in zip(env_indices, bb_temps):
                    # 将环境温度值转换为文档索引
                    env_val_python = env_val.item()  # 提取为 Python int
                    env_idx = self.env_temp_to_index.get(env_val_python)
                    if env_idx is None:
                        raise ValueError(f"无效的环境温度值: {env_val}，应为0/10/20/35")
                    # 获取部件温度
                    temps = self.get_temperatures(env_idx, bb_temp)
                    component_dict['zhujing'].append(temps['主镜温度'])
                    component_dict['fensejing'].append(temps['分色镜温度'])
                    component_dict['xitong'].append(temps['系统温度'])
                    component_dict['xiangji'].append(temps['相机温度'])
                    component_dict['xinzhou'].append(temps['芯轴温度'])
                
                # 转换为张量并调整维度
                components = [
                    torch.tensor(component_dict[k], device=self.device).float().unsqueeze(1)
                    for k in ['zhujing', 'fensejing', 'xitong', 'xiangji', 'xinzhou']
            ]


                # # 调试打印
                # print("输入维度检查:")
                # print(f"vid: {vid.shape}")
                # print(f"time_indices: {time_indices.unsqueeze(1).shape}")
                # print(f"env_indices: {temp_huanjing_indices.unsqueeze(1).shape}")
                # for name, comp in zip(['主镜', '分色镜', '系统', '相机', '芯轴'], components):
                #     print(f"{name}温度: {comp.shape}")
                # print("输入检查:")
                # print(f"vid: {vid.shape}")
                # print(f"time_indices: {time_indices.unsqueeze(1)}")
                # print(f"env_indices: {temp_huanjing_indices.unsqueeze(1)}")
                # print(f"temperature_indices:{temperature_indices.unsqueeze(1)}")
                # for name, comp in zip(['主镜', '分色镜', '系统', '相机', '芯轴'], components):
                #     print(f"{name}温度: {comp}")
                
                
                components_tensor = torch.cat(components, dim=1)  # 合并5个温度参数为[B,5]
                    # 假设 txt 文件路径
                txt_file_path = "/opt/data/private/czw/207/BDINR-master/3.70-4.80radian.txt"

                # 读取 txt 文件中的数据
                data = np.loadtxt(txt_file_path)  # 假设文件格式为两列：温度（K氏温度）和辐射亮度值
                temperature_k = data[:, 0]  # 第一列是温度（K氏温度）
                radiance_values = data[:, 1]  # 第二列是辐射亮度值

                # 将 temperature_indices 从摄氏温度转换为 K 氏温度
                temperature_indices_k = temperature_indices + 273.15
                temperature_k = torch.from_numpy(temperature_k)

                # 根据 temperature_indices_k 找到对应的辐射亮度值
                target_radiance = []
                for idx in temperature_indices_k:
                    # 找到最接近的温度值
                    # 将 temperature_k 转换为 Tensor
                    temperature_k = temperature_k.to(idx.device)

                    # 使用 Tensor 操作
                    closest_idx = torch.argmin(torch.abs(temperature_k - idx))
                    target_radiance.append(radiance_values[closest_idx]/10)

                # 将辐射亮度值转换为张量
                target_radiance = torch.tensor(target_radiance, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

                 # 步骤 1：调整为 [8,1,1,1]
                target_radiance = target_radiance.permute(3, 0, 1, 2)

                # 步骤 2：调整为 [8,1,3,1,1]
                target_radiance = target_radiance.unsqueeze(2).repeat(1, 1, 3, 1, 1)

                # 步骤 3：调整为 [8,1,3,511,640]
                # target_radiance = target_radiance.expand(8, 1, 3, 511, 640)
                target = target_radiance.expand(-1, 1, 3, 80, 80).to(self.device)



                # 调用模型，传递 vid、Temperature idx 和 Time idx

                output, data = self.model(vid, time_indices,temp_huanjing_indices,components_tensor)


                output = output.float()
                target = target.float()
                output = torch.mean(output, dim=[3,4], keepdim=True)
                # print(temp_huanjing_indices.shape)

                # print(temp_huanjing_indices[1,:],time_indices[1,:],temperature_indices[1,:])
                # print("output:",output[1,0,0,:,:],"target:",target_radiance[1,0,0,:,:])


                # forward
                # output, data, data_noisy = self.model(vid)
                output_ = torch.flatten(output, end_dim=1)
                target_ = torch.flatten(target, end_dim=1)
                # output_value = output_[1,:,:,:]
                # target_value = target_[1,:,:,:]

                # print(torch.mean(((output_value - target_value) ** 2)/target_value) *100)
  
                # target_mean = torch.mean(target_)

                 # 恢复形状
                output_ = torch.broadcast_to(output_, (-1, 3, 80, 80))
                # target_ = torch.full(target_.shape, target_mean.item(), device=target.device)


                # # 遍历 vid 的第一个维度（2）
                # if(epoch==1 or epoch==1):

                #     for i in range(vid.shape[0]):
                #         # 提取第 i 个张量
                #         vid_i = vid[i,:,:,:,:]  # 形状为 [1, 1, 511, 640]

                #         # 调整张量的形状
                #         vid_ = vid_i.squeeze()  # 去掉多余的维度，形状变为 [511, 640]

                #         # # 确保张量的值在 [0, 255] 范围内
                #         # vid_ = (vid_ * 255).to(torch.uint8)

                #         # 使用 torchvision.utils.save_image 保存图片
                #         if len(vid_.shape) == 2:
                #             vid_ = vid_.unsqueeze(0).unsqueeze(0)  # 调整为 [1, 1, 511, 640]

                #         vutils.save_image(vid_i, f'/opt/data/private/czw/207/BDINR-master/vid{epoch}_{i+1}.png')
                #         print("vid:",time_indices,temp_huanjing_indices,temperature_indices)
                #         print("output_:",output_[:,0:1,10:11,10:11])
                #         print("target_",target_[:,0:1,10:11,10:11])
            
                # main loss

                loss = self.losses['main_loss'] * \
                    self.criterion['main_loss'](output_, target_)
                # reblur loss: frame_n should equal to ce_code_n cases
                if 'reblur_loss' in self.losses:
                    ce_output = self._ce_reblur(output)
                    loss = loss + self.losses['reblur_loss'] *self.criterion['reblur_loss'](ce_output, data)
                
                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output.shape, target.shape)
                        # print(output_.shape,target_.shape)
                        # print("output_",output_[:,0:1,10:11,10:11],"target_",target_[:,0:1,10:11,10:11])
                        # print(time_indices,temp_huanjing_indices,temperature_indices_k)
                        metric_v = met(output_, target_)
                        # print(metric_v)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': vid[0, ...], 'output': output[0, 0::frame_num//1, ...], 'target': target[0, 0::frame_num//1, ...]}

                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'valid',
                                 loss, iter_metrics, {})

                if (batch_idx+1) == self.limit_valid_iters:
                    # save demo images to tensorboard after valid epoch
                    self.writer.set_step(epoch)
                    for k, v in image_tensors.items():
                        self.writer.add_image(
                            f'valid/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                    break

        # add histogram of model parameters to the tensorboard
        if self.log_weight:
            for name, p in self.model.BlurNet.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            total = self.data_loader.batch_size * self.limit_train_iters
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.limit_train_iters
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)


#======================================
# Trainning: run Trainer for trainning
#======================================


def trainning(gpus, config):
    # enable access to non-existing keys
    OmegaConf.set_struct(config, False)
    n_gpu = len(gpus)
    config.n_gpu = n_gpu
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # if n_gpu > 1:
    #     torch.multiprocessing.spawn(
    #         multi_gpu_train_worker, nprocs=n_gpu, args=(gpus, config))
    # else:
    train_worker(config)


def train_worker(config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    logger = get_logger('train')
    # setup data_loader instances


    data_loader, valid_data_loader = instantiate(config.data_loader)
    # print(data_loader.dataset[0].shape)
    # print(e)

    # build model. print it's structure

    model = instantiate(config.arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = {}
    if 'main_loss' in config.loss:
        criterion['main_loss'] = instantiate(config.main_loss)
    if 'reblur_loss' in config.loss:
        criterion['reblur_loss'] = instantiate(
            config.reblur_loss)
    metrics = [instantiate(met) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


# def multi_gpu_train_worker(rank, gpus, config):
#     """
#     Training with multiple GPUs

#     Args:
#         rank ([type]): [description]
#         gpus ([type]): [description]
#         config ([type]): [description]

#     Raises:
#         RuntimeError: [description]
#     """
#     # initialize training config
#     config.local_rank = rank
#     if(platform.system() == 'Windows'):
#         backend = 'gloo'
#     elif(platform.system() == 'Linux'):
#         backend = 'nccl'
#     else:
#         raise RuntimeError('Unknown Platform (Windows and Linux are supported')
#     dist.init_process_group(
#         backend=backend,
#         init_method='tcp://127.0.0.1:34567',
#         world_size=len(gpus),
#         rank=rank)
#     torch.cuda.set_device(gpus[rank])

#     # start training processes
#     train_worker(config)
