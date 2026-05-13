#batch_norm.py:nvidia-smi
import torch
import torch.nn as nn

class DimensionAgnosticBatchNorm(nn.Module):
    """专为3D数据设计的批归一化层 -"""

    def __init__(self,
                 num_channels: int,  # 改为必需参数
                 eps: float = 1e-5,
                 affine: bool = True,  # 默认设为True
                 momentum: float = 0.1,
                 sigma: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.nc = num_channels
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.sigma = sigma

        if affine:
            self.weight = nn.Parameter(torch.ones(self.nc))
            self.bias = nn.Parameter(torch.zeros(self.nc))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.nc))
        self.register_buffer('running_var', torch.ones(self.nc))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, D, H, W) 对于3D数据
        if x.dim() != 5:
            raise ValueError(f"预期5D输入 (N, C, D, H, W)，得到 {x.dim()}D")

        if self.training:
            # 计算当前批次的统计量
            dims = [0, 2, 3, 4]  # 沿批次和空间维度求平均，保留通道
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)

            # 更新运行统计量
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # 重塑以进行广播
        shape = [1, self.nc, 1, 1, 1]
        mean = mean.view(shape)
        var = var.view(shape)

        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)

        # 可学习的仿射变换
        if self.affine:
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x = x * weight + bias

        # 额外的缩放（扩散模型常用）
        x = x * self.sigma

        return x

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化，用于扩散模型采样"""
        shape = [1, self.nc, 1, 1, 1]

        # 反转额外缩放
        x = x / self.sigma

        # 反转仿射变换
        if self.affine:
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x = (x - bias) / weight

        # 反转归一化
        mean = self.running_mean.view(shape)
        var = self.running_var.view(shape)
        x = x * torch.sqrt(var + self.eps) + mean

        return x


class IdentityBatchNorm(nn.Module):
    """恒等批归一化，用于跳过归一化"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        return x