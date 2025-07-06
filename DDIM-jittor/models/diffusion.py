import math
import jittor as jt
from jittor import nn

def get_timestep_embedding(timesteps, embedding_dim): # 构造时间步t的正弦嵌入向量
    assert len(timesteps.shape) == 1  # 确保输入是一维张量
    half_dim = embedding_dim // 2  # 嵌入维度的一半，用于 sin 和 cos
    emb_factor = math.log(10000) / (half_dim - 1) # 计算频率的缩放因子
    freq = jt.exp(jt.arange(half_dim).float32() * -emb_factor) # 构造频率序列
    args = timesteps.float32().unsqueeze(1) * freq.unsqueeze(0) # 扩展时间步维度
    emb = jt.concat([jt.sin(args), jt.cos(args)], dim=1)# 拼接 sin 和 cos 两部分
    if embedding_dim % 2 == 1: # 如果 embedding_dim 是奇数，就补 0 保持维度一致
        emb = nn.pad(emb, [0, 1], mode='constant', value=0.0)
    return emb

def nonlinearity(x): # Swish 激活函数
    return x * jt.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv: # 可以选择是否上采样后加一个卷积
            self.conv = nn.Conv(
                in_channels, in_channels,
                kernel_size=3, stride=1, padding=1
            )
    def execute(self, x):
        x = nn.interpolate(x, scale_factor=2.0, mode='nearest') # 最近邻插值上采样，特征图扩大两倍
        if self.with_conv: # 卷积处理
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv( # 使用 stride=2 的卷积实现下采样
                in_channels, in_channels,
                kernel_size=3, stride=2, padding=0 # 不设置 padding，因为我们手动 pad
            )
    def execute(self, x):
        if self.with_conv:
            x = nn.pad(x, [0, 1, 0, 1], mode="constant", value=0.0) # 手动 padding：右边 +1、下边 +1
            x = self.conv(x)
        else:
            x = nn.pool(x, kernel_size=2, stride=2, op="mean") # 平均池化

        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.1, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_conv_shortcut = conv_shortcut # 使用 1×1 卷积进行简单投影
        # 第一层归一化 + 卷积
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        # 时间步嵌入的线性变换层
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        # 第二层归一化 + dropout + 卷积
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入输出通道不一致，需转换：1x1或3x3卷积
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
    def execute(self, x, temb): # x: 输入图像特征 (B, C, H, W)，temb: 时间步嵌入 (B, temb_channels)
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1) # 注入时间步嵌入：先激活，再投影，然后广播加到 feature map
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels: # 残差连接处理：如果 in/out channel 不一致则先转换
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h  # 残差连接
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def execute(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1) # [B, HW, C]
        k = k.reshape(b, c, h*w) # [B, C, HW]
        w_ = jt.bmm(q, k) # [B, HW, HW]
        w_ = w_ * (c ** -0.5) # 缩放
        w_ = nn.softmax(w_, dim=2) # softmax 注意力
        v = v.reshape(b, c, h*w) # [B, C, HW]
        w_ = w_.permute(0, 2, 1) # [B, HW, HW]
        h_ = jt.bmm(v, w_) # [B, C, HW]
        h_ = h_.reshape(b, c, h, w) # 恢复空间结构
        h_ = self.proj_out(h_)
        return x + h_

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = jt.nn.Parameter(jt.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # 时间步编码
        self.temb_dense = nn.Sequential(
            nn.Linear(self.ch, self.temb_ch),
            nonlinearity,
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        # 下采样
        self.conv_in = nn.Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,out_channels=block_out,dropout=dropout,temb_channels=self.temb_ch))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        # 中间模块
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, temb_channels=self.temb_ch)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, temb_channels=self.temb_ch)

        # 上采样
        self.up = nn.ModuleList()
        ups = []
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, dropout=dropout, temb_channels=self.temb_ch))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            ups.insert(0, up)
        self.up = nn.ModuleList(ups)

        # 尾部
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def execute(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # 时间步编码
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense(temb)

        # 下采样
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间模块
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](jt.concat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 尾部
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h