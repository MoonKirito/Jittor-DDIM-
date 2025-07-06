class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu              # 衰减系数
        self.shadow = {}          # 存储滑动平均后的参数副本

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                # clone当前权重（会被 EMA 更新）
                self.shadow[name] = param.clone().detach()

    def update(self, module): # 每训练一步后调用，用当前模型参数对 shadow 参数做滑动平均更新
        for name, param in module.named_parameters():
            if param.requires_grad:
                new_data = (1. - self.mu) * param + self.mu * self.shadow[name] # EMA 更新：shadow = (1-mu)*new + mu*old
                self.shadow[name] = new_data.detach()

    def ema(self, module): # 用 shadow 参数替换当前模型参数
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.assign(self.shadow[name]) # 将 shadow 值写入模型参数

    def ema_copy(self, module): # 创建一个当前模型的副本，并将 EMA 参数应用于副本
        model_copy = type(module)(module.config).to(module.config.device) # 新建模型实例
        model_copy.load_parameters(module.state_dict())
        self.ema(model_copy)
        return model_copy

    def state_dict(self): # 用于保存或恢复 EMA 状态
        return self.shadow

    def load_state_dict(self, state_dict): # 用于训练恢复
        self.shadow = state_dict
