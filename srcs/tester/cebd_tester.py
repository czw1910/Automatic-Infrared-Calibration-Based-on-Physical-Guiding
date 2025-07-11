# import logging
# import os
# import torch
# import time
# from omegaconf import OmegaConf
# from tqdm import tqdm
# from srcs.utils.util import instantiate
# from srcs.utils.utils_image_kair import tensor2uint, imsave
# import numpy as np
# # 强制标准输出使用UTF-8编码
# import sys
# import io
# # 强制标准输出使用UTF-8编码
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, 
#                              encoding='utf-8', 
#                              errors='replace', 
#                              line_buffering=True)

# def testing(gpus, config):
#     test_worker(gpus, config)


# def test_worker(gpus, config):
#     # prevent access to non-existing keys
#     OmegaConf.set_struct(config, True)

#     # logger & dir setting
#     logger = logging.getLogger('test')
#     os.makedirs(config.outputs_dir, exist_ok=True)

#     # 计算输入图片均值并动态选择模型
#     data_loader = instantiate(config.test_data_loader)
#     first_batch = next(iter(data_loader))  # 取第一个批次用于计算均值
#     vid = first_batch[0]  # 假设vid是第一个元素
    
#     # 计算均值（考虑数据归一化情况）
#     vid_mean = torch.mean(vid).item()
#     if vid_mean > 6000:
#         a = 1
#         checkpoint = config.checkpoint1  # 使用第一个预训练模型
#         logger.info(f"choose checkpoint1,{vid_mean}")
#     else:
#         a = 2
#         checkpoint = config.checkpoint2  # 使用第二个预训练模型
#         logger.info(f"choose checkpoint2, {vid_mean}")

#     # prepare model & checkpoint for testing
#     logger.info(f" Loading checkpoint: {checkpoint} ...")
#     checkpoint = torch.load(checkpoint)
#     logger.info("Checkpoint loaded!")

#     # select config file
#     if 'config' in checkpoint:
#         loaded_config = OmegaConf.create(checkpoint['config'])
#     else:
#         loaded_config = config

#     # instantiate model
#     model = instantiate(loaded_config.arch)
#     logger.info(model)
#     if len(gpus) > 1:
#         model = torch.nn.DataParallel(model, device_ids=gpus)

#     # load weight
#     state_dict = checkpoint['state_dict']
#     # 处理可能的权重不匹配问题（沿用之前的解决方案）
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)

#     # instantiate loss and metrics
#     criterion = None  # don't calc loss in test
#     metrics = [instantiate(met) for met in config.metrics]

#     # 重新初始化data_loader（因为之前取了第一个批次）
#     data_loader = instantiate(config.test_data_loader)
    
#     # test
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     log = test(data_loader, model, device, criterion, metrics, config, logger)
#     logger.info(log)

# def load_temperature_docs():
#     """加载四组环境温度对应的温度表"""
#     docs = [
#         {   # 文档1对应环境温度索引0
#             '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 40, 60, 80],
#             '主镜温度': [5.0, 4.8, 4.5, 4.3, 4.1, 4.0, 3.8, 3.6, 3.5, 3.4],
#             '分色镜温度': [6.0, 5.8, 5.6, 5.5, 5.2, 5.1, 5.0, 4.8, 4.7, 4.6],
#             '系统温度': [9.0, 8.8, 8.6, 8.4, 8.2, 8.1, 8.0, 7.9, 7.8, 7.7],
#             '相机温度': [7.3, 7.1, 7.0, 6.7, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0],
#             '芯轴温度': [4.6, 4.3, 4.1, 3.8, 3.6, 3.5, 3.3, 3.3, 3.1, 3.0]
#         },
#         {   # 文档2对应环境温度索引10
#             '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 30, 40, 60, 80],
#             '主镜温度': [13.8, 13.7, 13.6, 13.5, 13.3, 13.3, 13.2, 13.2, 13.1, 13.0, 13.0],
#             '分色镜温度': [13.9, 14.0, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.2, 14.3, 14.2],
#             '系统温度': [16.4, 16.8, 17.1, 17.2, 17.3, 17.3, 17.3, 17.4, 17.4, 17.4, 17.4],
#             '相机温度': [14.8, 15.1, 15.4, 15.5, 15.6, 15.6, 15.6, 15.6, 15.4, 15.7, 15.7],
#             '芯轴温度': [13.2, 13.2, 13.1, 13.0, 12.8, 12.8, 12.8, 12.9, 12.8, 12.7, 12.7]
#         },
#         {   # 文档3对应环境温度索引20
#             '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 40, 60],
#             '主镜温度': [15.5, 16.0, 16.5, 16.9, 17.3, 17.6, 17.8, 18.1, 18.1],
#             '分色镜温度': [15.9, 16.8, 17.7, 18.3, 19.0, 19.3, 19.6, 20.0, 20.0],
#             '系统温度': [19.1, 20.5, 21.5, 22.1, 22.7, 23.1, 23.3, 23.7, 23.6],
#             '相机温度': [17.6, 18.9, 19.8, 20.4, 21.0, 21.3, 21.6, 22.0, 22.0],
#             '芯轴温度': [17.6, 18.0, 18.3, 18.6, 18.8, 19.1, 19.1, 19.4, 19.6]
#         },
#         {   # 文档4对应环境温度索引35
#             '黑体温度': [-50, -35, -20, -10, 0, 10, 20, 30, 40, 60, 80],
#             '主镜温度': [31.0, 31.4, 31.8, 32.1, 32.5, 32.8, 33.0, 33.2, 33.5, 33.8, 34.0],
#             '分色镜温度': [31.9, 32.8, 33.5, 34.0, 34.4, 34.8, 35.1, 35.4, 35.6, 36.0, 36.3],
#             '系统温度': [35.1, 36.5, 37.2, 37.8, 38.1, 38.5, 38.8, 39.0, 39.3, 39.6, 39.8],
#             '相机温度': [33.5, 34.7, 35.5, 36.0, 36.5, 36.8, 37.0, 37.3, 37.5, 37.8, 38.1],
#             '芯轴温度': [32.8, 33.1, 33.4, 33.6, 33.8, 34.0, 34.1, 34.3, 34.5, 34.6, 34.8]
#         }
#     ]
#     return docs



# def get_temperatures( env_idx, bb_temp):
#     """
#     根据环境温度索引和黑体温度获取各部件温度
#     :param env_idx: 环境温度索引（0-3）
#     :param bb_temp: 黑体温度值
#     :return: 字典包含各部件温度
#     """
#     temperature_docs = load_temperature_docs()

#     # 索引有效性检查
#     if env_idx < 0 or env_idx >= len(temperature_docs):
#         raise ValueError(f"无效的环境温度索引: {env_idx}，有效值为0-3")
    
#     doc = temperature_docs[env_idx]
#     bb_list = doc['黑体温度']
    
#     # 找到最接近的黑体温度索引
#     closest_idx = min(
#         range(len(bb_list)),
#         key=lambda i: abs(bb_list[i] - bb_temp)
#     )
    
#     return {
#         '主镜温度': doc['主镜温度'][closest_idx],
#         '分色镜温度': doc['分色镜温度'][closest_idx],
#         '系统温度': doc['系统温度'][closest_idx],
#         '相机温度': doc['相机温度'][closest_idx],
#         '芯轴温度': doc['芯轴温度'][closest_idx]
#     }




# def test(data_loader, model,  device, criterion, metrics, config, logger=None):
#     '''
#     test step
#     '''

#     # 初始化温度文档数据（只需加载一次）
#     temperature_docs = load_temperature_docs()
#     env_temp_to_index = {
#         0: 0,   # 文档1对应环境温度0
#         10: 1,  # 文档2对应环境温度10
#         20: 2,  # 文档3对应环境温度20
#         35: 3   # 文档4对应环境温度35
#     }
#     # init
#     model = model.to(device)

#     interp_scale = getattr(model, 'frame_n', 8)//getattr(model, 'ce_code_n', 8)
#     if config.get('save_img', False):
#         os.makedirs(config.outputs_dir+'/output')
#         os.makedirs(config.outputs_dir+'/target')
#         os.makedirs(config.outputs_dir+'/input')

#     # run
#     ce_weight = model.BlurNet.ce_weight.detach().squeeze()

#     ce_code = ((torch.sign(ce_weight)+1)/2).int()


#     model.eval()
#     total_metrics = torch.zeros(len(metrics), device=device)
#     time_start = time.time()
#     total_error = 0  # 用于累加所有图像的误差
#     num_images = len(data_loader)  # 图像数量
#     num = 0
#     with torch.no_grad():
#         for i, vid in enumerate(tqdm(data_loader, desc='Testing')):
#             # # move vid to gpu, convert to 0-1 float
#             # # vid = vid.to(device).float()/255 
#             # # N, F, C, Hx, Wx = vid.shape
#             # # 获取当前批次的样本索引
#             # start_idx = i * data_loader.batch_size
#             # end_idx = start_idx + vid.size(0)  # vid.size(0) 是当前批次的实际大小

#             # # 提取当前批次的样本索引
#             # # 提取当前批次的样本索引
#             # batch_indices = list(range(start_idx, end_idx))  # 确保返回的是一个整数列表

#             # # 通过索引获取 Temperature idx 和 Time idx
#             # temp_huanjing_indices = torch.tensor([data_loader.dataset.temp_huanjing_idx[idx] for idx in batch_indices], device=device).to(dtype=torch.float32)
#             # time_indices = torch.tensor([data_loader.dataset.time_idx[idx] for idx in batch_indices], device=device)
#             # temperature_indices = torch.tensor([data_loader.dataset.temperature_idx[idx] for idx in batch_indices], device=device).to(dtype=torch.float32)
#             temperature_indices = vid[1]
#             time_indices = vid[2]
#             temp_huanjing_indices = vid[3]
#             vid = vid[0]
#             # print(vid,temperature_indices,time_indices,temp_huanjing_indices)

#             vid = vid.to(device).float() / 16383
#             temperature_indices = temperature_indices.to(device).float()
#             time_indices = time_indices.to(device).float()
#             temp_huanjing_indices = temp_huanjing_indices.to(device).float()

#             N, F, C, Hx, Wx = vid.shape

#                         # 转换到CPU处理
#             env_indices = temp_huanjing_indices.cpu().numpy()
#             bb_temps = temperature_indices.cpu().numpy()

#             # 批量获取温度参数
#             component_dict = {
#                 'zhujing': [], 'fensejing': [], 'xitong': [],
#                 'xiangji': [], 'xinzhou': []
#             }
            

#             # 批量处理环境温度和黑体温度
#             for env_val, bb_temp in zip(env_indices, bb_temps):
#                 # 将环境温度值转换为文档索引
#                 env_val_python = env_val.item()  # 提取为 Python int
#                 env_idx = env_temp_to_index.get(env_val_python)
#                 if env_idx is None:
#                     raise ValueError(f"无效的环境温度值: {env_val}，应为0/10/20/35")
#                 # 获取部件温度
#                 temps = get_temperatures(env_idx, bb_temp)
#                 component_dict['zhujing'].append(temps['主镜温度'])
#                 component_dict['fensejing'].append(temps['分色镜温度'])
#                 component_dict['xitong'].append(temps['系统温度'])
#                 component_dict['xiangji'].append(temps['相机温度'])
#                 component_dict['xinzhou'].append(temps['芯轴温度'])
            
#             # 转换为张量并调整维度
#             components = [
#                 torch.tensor(component_dict[k], device=device).float().unsqueeze(1)
#                 for k in ['zhujing', 'fensejing', 'xitong', 'xiangji', 'xinzhou']
#             ]
#             components_tensor = torch.cat(components, dim=1)  # 合并5个温度参数为[B,5]

#             # direct

#             output, data = model(vid,time_indices,temp_huanjing_indices, components_tensor)

#             # clamp to 0-1
#             # output = torch.clamp(output, 0, 1)
#             output_ = torch.mean(output, dim=[3,4], keepdim=True)
#             vid_mean = torch.mean(vid, dim=[3,4], keepdim=True)


#             output_ = torch.clamp(output_, 0, 1)
#             # 恢复形状
#             output_ = output_.repeat(1, 1, 1, 80, 80)  # 显式复制代替广播
#              # 假设 txt 文件路径
#             txt_file_path = "/opt/data/private/czw/207/BDINR-master/3.70-4.80radian.txt"

#             # 读取 txt 文件中的数据
#             data1 = np.loadtxt(txt_file_path)  # 假设文件格式为两列：温度（K氏温度）和辐射亮度值
#             temperature_k = data1[:, 0]  # 第一列是温度（K氏温度）
#             radiance_values = data1[:, 1]  # 第二列是辐射亮度值
#              # 将 temperature_k 转换为 Tensor
#             temperature_k = torch.from_numpy(temperature_k)

#             # 将 temperature_indices 从摄氏温度转换为 K 氏温度
#             temperature_indices_k = temperature_indices + 273.15

#             # 根据 temperature_indices_k 找到对应的辐射亮度值
#             target_radiance = []
#             for idx in temperature_indices_k:

#                 # 找到最接近的温度值
#                 temperature_k = temperature_k.to(idx.device)

#                 # 使用 Tensor 操作
#                 closest_idx = torch.argmin(torch.abs(temperature_k - idx))
#                 target_radiance.append(radiance_values[closest_idx]/10)
#                 # print(temperature_indices)
#                 # print(target_radiance)


#             # 将辐射亮度值转换为张量
#             target_radiance = torch.tensor(target_radiance, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
#             target_radiance = target_radiance.expand(1, 1, 1, 80, 80).to(device)

#             # print(temp_huanjing_indices[0,:],time_indices[0,:],temperature_indices[0,:])
#             # print("output:",output_[0,0,0,:,:],"target:",target_radiance[0,0,0,:,:])

#             # 计算单张图像的百分比误差

#             # 提取单个值
#             output_value = output_[:,:,:,10:11,10:11].item()  # 使用 .item() 提取标量值
#             target_value = target_radiance[:,:,:,10:11,10:11].item()
#             output_value = torch.tensor(output_value)
#             target_value = torch.tensor(target_value)

#             # 计算百分比误差
#             error_percentage1 = torch.abs((output_value-target_value)) / (torch.abs(target_value)) * 100
#             error_percentage2 = torch.abs((output_value*10-target_value*10)**2) / (torch.abs(target_value*10)) * 100
#             error_percentage3 = torch.mean(((output_value*10 - target_value*10) ** 2)/target_value/10) *100 # 沿着空间维度计算 MSE

#             # 打印结果
#             # print(f"环境温度：{temp_huanjing_indices},积分时间：{time_indices},黑体温度：{temperature_indices}")
#             # print(f"初始输入: {vid_mean.item():.6f}")
#             # print(f"预测黑体辐射亮度: {output_value:.6f}")
#             # print(f"真实辐射亮度: {target_value:.6f}")
#             # print("Similarity:", error_percentage1)
#             # print("Normalized MSE:", error_percentage2)


#             if(error_percentage3.item()>8):
#                 num=num+1
#                             # 打印结果
#                 print(f"环境温度：{temp_huanjing_indices},积分时间：{time_indices},黑体温度：{temperature_indices}")
#                 print(f"初始输入: {vid_mean.item():.6f}")
#                 print(f"预测黑体辐射亮度: {output_value:.6f}")
#                 print(f"真实辐射亮度: {target_value:.6f}")
#                 print("Similarity_error:", error_percentage1)
#                 print("Normalized MSE_error:", error_percentage2)
#                 print(num)    
#             # 累加误差
#             total_error += error_percentage3.mean().item()  # 取均值后累加

#             # print(vid.shape)
#             # print(target_radiance.shape)
#             # print(eee)

#             # print(output_.shape)
#             # save some sample images
#             if config.get('save_img', False):
#                 scale_fc =1
#                 for k, (in_img, out_img, gt_img) in enumerate(zip(data, output_, target_radiance)):
#                     # print(in_img.shape,out_img.shape,gt_img.shape)
#                     in_img = tensor2uint(in_img*scale_fc)
#                     imsave(
#                         in_img, f'{config.outputs_dir}input/ce-blur#{i*N+k+1:04d}.jpg')
#                     for j in range(1):
#                         # print(len(ce_code))
#                         # print(temp_huanjing_indices,time_indices,temperature_indices)

#                         # print("out_img",out_img[:,:,10:11,10:11])
#                         # print("gt_img",gt_img[:,:,10:11,10:11])
#                         out_img_j = tensor2uint(out_img[j])
#                         gt_img_j = tensor2uint(gt_img[j])

#                         imsave(
#                             out_img_j, f'{config.outputs_dir}output/out-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')
#                         imsave(
#                             gt_img_j, f'{config.outputs_dir}target/gt-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')

#             # metrics on test set
#             output_all = torch.flatten(output_, end_dim=1)
#             target_all = torch.flatten(target_radiance, end_dim=1)


#             batch_size = data.shape[0]
#             for i, metric in enumerate(metrics):
#                 total_metrics[i] += metric(output_all, target_all) * batch_size
#     # 计算所有图像的平均误差
#     average_error = total_error / num_images
#     print(total_error,num_images)
#     print("Average Percentage Error:", average_error)

#     time_end = time.time()
#     time_cost = time_end-time_start
#     n_samples = len(data_loader.sampler)
#     log = {'time/sample': time_cost/n_samples,
#            'ce_code': ce_code}
#     log.update({
#         met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
#     })
#     return log
import logging
import os
import torch
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave
import numpy as np
import sys
import io

# 强制标准输出使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, 
                             encoding='utf-8', 
                             errors='replace', 
                             line_buffering=True)

def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    os.makedirs(config.outputs_dir, exist_ok=True)

    # 加载两个模型的配置和权重
    logger.info("准备加载两个模型...")
    model1, config1 = load_model(config.checkpoint1, gpus, logger)
    model2, config2 = load_model(config.checkpoint2, gpus, logger)
    logger.info("两个模型加载完成")

    # 初始化data_loader
    data_loader = instantiate(config.test_data_loader)
    
    # instantiate loss and metrics
    criterion = None  # don't calc loss in test
    metrics = [instantiate(met) for met in config.metrics]
    
    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, model1, model2, device, criterion, metrics, config, logger)
    logger.info(log)

def load_model(checkpoint_path, gpus, logger):
    """加载模型及其配置"""
    logger.info(f" Loading checkpoint: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path)
    logger.info("Checkpoint loaded!")

    # select config file
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = None  # 假设config中已包含模型配置

    # instantiate model
    model = instantiate(loaded_config.arch) if loaded_config else instantiate(config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model, loaded_config if loaded_config else config

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



def get_temperatures( env_idx, bb_temp):
    """
    根据环境温度索引和黑体温度获取各部件温度
    :param env_idx: 环境温度索引（0-3）
    :param bb_temp: 黑体温度值
    :return: 字典包含各部件温度
    """
    temperature_docs = load_temperature_docs()

    # 索引有效性检查
    if env_idx < 0 or env_idx >= len(temperature_docs):
        raise ValueError(f"无效的环境温度索引: {env_idx}，有效值为0-3")
    
    doc = temperature_docs[env_idx]
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




def test(data_loader, model1, model2, device, criterion, metrics, config, logger=None):
    '''
    test step，支持根据每个批次动态选择模型
    '''

    # 初始化温度文档数据（只需加载一次）
    temperature_docs = load_temperature_docs()
    env_temp_to_index = {
        0: 0,   # 文档1对应环境温度0
        10: 1,  # 文档2对应环境温度10
        20: 2,  # 文档3对应环境温度20
        35: 3   # 文档4对应环境温度35
    }
    # 将两个模型移至设备
    model1 = model1.to(device)
    model2 = model2.to(device)

    interp_scale = getattr(model1, 'frame_n', 8)//getattr(model1, 'ce_code_n', 8)
    if config.get('save_img', False):
        os.makedirs(config.outputs_dir+'/output')
        os.makedirs(config.outputs_dir+'/target')
        os.makedirs(config.outputs_dir+'/input')

    # run
    ce_weight = model1.BlurNet.ce_weight.detach().squeeze() if hasattr(model1, 'BlurNet') else None
    ce_code = ((torch.sign(ce_weight)+1)/2).int() if ce_weight is not None else None

    model1.eval()
    model2.eval()
    total_metrics = torch.zeros(len(metrics), device=device)
    time_start = time.time()
    total_error = 0  # 用于累加所有图像的误差
    num_images = len(data_loader)  # 图像数量
    num = 0
    with torch.no_grad():
        for i, vid in enumerate(tqdm(data_loader, desc='Testing')):
            # 解析数据
            temperature_indices = vid[1]
            time_indices = vid[2]
            temp_huanjing_indices = vid[3]
            vid_input = vid[0]  # 重命名以便清晰识别
            
            # 数据预处理
            vid_input = vid_input.to(device).float() / 16383
            temperature_indices = temperature_indices.to(device).float()
            time_indices = time_indices.to(device).float()
            temp_huanjing_indices = temp_huanjing_indices.to(device).float()

            N, F, C, Hx, Wx = vid_input.shape

            # 计算当前批次vid的均值
            vid_mean = torch.mean(vid_input).item()
            
            # 根据均值选择模型
            if vid_mean > 0.3662333:
                current_model = model1
                model_choice = "checkpoint1"
            else:
                current_model = model2
                model_choice = "checkpoint2"
            
            logger.info(f"Batch {i}, choose {model_choice}, input mean: {vid_mean}")

            # 转换到CPU处理温度参数
            env_indices = temp_huanjing_indices.cpu().numpy()
            bb_temps = temperature_indices.cpu().numpy()

            # 批量获取温度参数
            component_dict = {
                'zhujing': [], 'fensejing': [], 'xitong': [],
                'xiangji': [], 'xinzhou': []
            }

            # 批量处理环境温度和黑体温度
            for env_val, bb_temp in zip(env_indices, bb_temps):
                env_val_python = env_val.item()  # 提取为 Python int
                env_idx = env_temp_to_index.get(env_val_python)
                if env_idx is None:
                    raise ValueError(f"无效的环境温度值: {env_val}，应为0/10/20/35")
                # 获取部件温度
                temps = get_temperatures(env_idx, bb_temp)
                component_dict['zhujing'].append(temps['主镜温度'])
                component_dict['fensejing'].append(temps['分色镜温度'])
                component_dict['xitong'].append(temps['系统温度'])
                component_dict['xiangji'].append(temps['相机温度'])
                component_dict['xinzhou'].append(temps['芯轴温度'])
            
            # 转换为张量并调整维度
            components = [
                torch.tensor(component_dict[k], device=device).float().unsqueeze(1)
                for k in ['zhujing', 'fensejing', 'xitong', 'xiangji', 'xinzhou']
            ]
            components_tensor = torch.cat(components, dim=1)  # 合并5个温度参数为[B,5]

            # 模型推理
            output, data = current_model(vid_input, time_indices, temp_huanjing_indices, components_tensor)

            # 处理输出
            output_ = torch.mean(output, dim=[3,4], keepdim=True)
            vid_mean = torch.mean(vid_input, dim=[3,4], keepdim=True)
            output_ = torch.clamp(output_, 0, 1)
            output_ = output_.repeat(1, 1, 1, 80, 80)  # 恢复形状

            # 读取辐射亮度数据
            txt_file_path = "/opt/data/private/czw/207/BDINR-master/3.70-4.80radian.txt"
            data1 = np.loadtxt(txt_file_path)
            temperature_k = data1[:, 0]  # K氏温度
            radiance_values = data1[:, 1]  # 辐射亮度值
            temperature_k = torch.from_numpy(temperature_k).to(device)

            # 计算目标辐射亮度
            temperature_indices_k = temperature_indices + 273.15
            target_radiance = []
            for idx in temperature_indices_k:
                closest_idx = torch.argmin(torch.abs(temperature_k - idx))
                target_radiance.append(radiance_values[closest_idx]/10)
            target_radiance = torch.tensor(target_radiance, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            target_radiance = target_radiance.expand(1, 1, 1, 80, 80).to(device)

            # 计算误差
            output_value = output_[:,:,:,10:11,10:11]
            target_value = target_radiance[:,:,:,10:11,10:11]
            error_percentage1 = torch.abs((output_value-target_value)) / (torch.abs(target_value)) * 100
            error_percentage2 = torch.abs((output_value*10-target_value*10)**2) / (torch.abs(target_value*10)) * 100
            error_percentage3 = torch.mean(((output_value*10 - target_value*10) ** 2)/target_value/10) * 100
            

            # 记录高误差样本（仅在需要打印时转换为标量）
            if error_percentage3.item() > 8:
                num += 1
                print(f"环境温度：{temp_huanjing_indices}, 积分时间：{time_indices}, 黑体温度：{temperature_indices}")
                print(f"初始输入: {vid_mean.item():.6f}")
                print(f"预测黑体辐射亮度: {output_value.item():.6f}")
                print(f"真实辐射亮度: {target_value.item():.6f}")
                print("Similarity_error:", error_percentage1.item())
                print("Normalized MSE_error:", error_percentage2.item())
                print(f"高误差样本数: {num}")    
            
            # 累加误差
            total_error += error_percentage3.mean().item()

            # 保存图像样本
            if config.get('save_img', False):
                scale_fc = 1
                for k, (in_img, out_img, gt_img) in enumerate(zip(data, output_, target_radiance)):
                    in_img = tensor2uint(in_img * scale_fc)
                    imsave(in_img, f'{config.outputs_dir}input/ce-blur#{i*N+k+1:04d}.jpg')
                    for j in range(1):
                        out_img_j = tensor2uint(out_img[j])
                        gt_img_j = tensor2uint(gt_img[j])
                        imsave(out_img_j, f'{config.outputs_dir}output/out-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')
                        imsave(gt_img_j, f'{config.outputs_dir}target/gt-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')

            # 计算评估指标
            output_all = torch.flatten(output_, end_dim=1)
            target_all = torch.flatten(target_radiance, end_dim=1)
            batch_size = data.shape[0]
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output_all, target_all) * batch_size
    
    # 计算平均误差
    average_error = total_error / num_images
    print(f"总误差: {total_error}, 图像总数: {num_images}")
    print(f"平均百分比误差: {average_error}")

    time_end = time.time()
    time_cost = time_end - time_start
    n_samples = len(data_loader.sampler)
    log = {'time/sample': time_cost / n_samples, 'ce_code': ce_code}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)})
    return log