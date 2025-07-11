import yaml
import io
import sys
import hydra
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from importlib import import_module
from itertools import repeat
from functools import partial, update_wrapper


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


# def get_logger(name=None):
#     if is_master():
#         # 加载 Hydra 配置
#         hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
#         logging_config = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)
        
#         # 配置日志系统
#         logging.config.dictConfig(logging_config)
        
#         # 替换 sys.stdout 为支持 UTF-8 编码的 TextIOWrapper
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#         sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
#     return logging.getLogger(name)

def get_logger(name=None):
    if is_master():
        hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
        job_logging = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)
        
        # 动态添加文件编码
        for handler in job_logging["handlers"].values():
            if handler.get("class") == "logging.FileHandler":
                handler["encoding"] = "utf-8"  # 强制指定 UTF-8 编码[3,6](@ref)
        
        logging.config.dictConfig(job_logging)
    
    return logging.getLogger(name)


def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in config, f'Config should have \'_target_\' for class instantiation.'
    target = config['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)

def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)
