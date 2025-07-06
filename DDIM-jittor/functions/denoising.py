import jittor as jt

def compute_alpha(beta, t): # ᾱ_t = ∏_{i=1}^{t}(1 - β_i)

    beta = jt.concat([jt.zeros((1,), dtype=beta.dtype), beta], dim=0) # 手动对齐 beta 索引
    alpha_bar = jt.cumprod(1 - beta, dim=0) # 累乘得到 ᾱ
    t = (t + 1).int32()
    alpha_bar_t = jt.index_select(alpha_bar, 0, t) # 取出每个样本对应的 ᾱ_t
    return alpha_bar_t.reshape((-1, 1, 1, 1)) # 扩展维度与图像匹配

def generalized_steps(x, seq, model, b, **kwargs): # DDIM推理过程
    with jt.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])  # 对应每一步的“下一步”时间
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = jt.full((n,), i, dtype=jt.int32)
            next_t = jt.full((n,), j, dtype=jt.int32)
            at = compute_alpha(b, t)
            at_next = compute_alpha(b, next_t)
            xt = xs[-1].stop_grad()  # 当前 x_t
            et = model(xt, t)        # 模型预测 ε_t
            # 重建 x0
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.stop_grad())
            # 采样项
            eta = kwargs.get("eta", 0)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            noise = jt.randn_like(x)
            xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et
            xs.append(xt_next.stop_grad())
    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs): # DDPM推理过程
    with jt.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = jt.full((n,), i, dtype=jt.int32)
            next_t = jt.full((n,), j, dtype=jt.int32)
            at = compute_alpha(b, t)
            atm1 = compute_alpha(b, next_t)
            beta_t = 1 - at / atm1
            xt = xs[-1].stop_grad()
            et = model(xt, t.float32())
            x0_from_e = (1.0 / at).sqrt() * xt - ((1.0 / at - 1).sqrt()) * et # 从 ε 反推出 x0
            x0_from_e = jt.clamp(x0_from_e, -1.0, 1.0)
            x0_preds.append(x0_from_e.stop_grad())
            mean = ( # 计算均值项
                atm1.sqrt() * beta_t * x0_from_e +
                ((1 - beta_t).sqrt() * (1 - atm1)) * xt
            ) / (1.0 - at)
            noise = jt.randn_like(xt) # 加噪
            mask = 1 - (t == 0).float32().reshape((-1, 1, 1, 1))
            logvar = beta_t.log()
            xt_prev = mean + mask * jt.exp(0.5 * logvar) * noise
            xs.append(xt_prev.stop_grad())
    return xs, x0_preds
