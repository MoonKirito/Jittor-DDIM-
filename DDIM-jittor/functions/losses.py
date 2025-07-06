import jittor as jt

def noise_estimation_loss(model, # MSE Loss
                          x0: jt.Var,
                          t: jt.Var,
                          e: jt.Var,
                          b: jt.Var,
                          keepdim=False):
    # 计算 alpha_bar_t = ∏(1 - beta_i)
    alpha_bar = jt.cumprod(1 - b, dim=0)
    a = alpha_bar[t].reshape((-1, 1, 1, 1))  # 替代 index_select

    # 构造 x_t
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # 前向推理，预测 ε_hat
    output = model(x, t.float32())

    # 计算 loss
    mse = (e - output) ** 2
    if keepdim:
        # 每个样本的总误差
        return mse.reshape([e.shape[0], -1]).sum(dim=1)  # [B]
    else:
        # 所有样本平均误差
        return mse.mean()  # 等价于 sum 后除以总元素数
    
loss_registry = {
'simple': noise_estimation_loss,
}