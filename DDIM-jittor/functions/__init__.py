import jittor as jt
from jittor import optim

def get_optimizer(config, parameters): # 根据 config.optim 中的配置，返回对应的优化器实例
    # 选择 Adam 优化器
    if config.optim.optimizer == 'Adam':
        return optim.Adam(
            params=parameters,
            lr=config.optim.lr, # 学习率
            weight_decay=config.optim.weight_decay, # L2 权重衰减
            eps=config.optim.eps, # 避免除零的小常数
        )

    # 选择 RMSProp 优化器
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(
            params=parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay
        )

    # 选择 SGD 优化器
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(
            params=parameters,
            lr=config.optim.lr,
            momentum=0.9 # 标准动量设置
        )

    # 不支持的优化器类型
    else:
        raise NotImplementedError(
            f"未知优化器类型：{config.optim.optimizer}"
        )
