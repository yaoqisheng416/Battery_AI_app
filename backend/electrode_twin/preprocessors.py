#preprocessors.py:
"""
边缘检测预处理器模块
功能：从距离变换（Distance Transform）或图像中提取多尺度边缘特征
特点：支持2D和3D数据，可微分，适用于深度学习模型
注意：本模块主要设计用于处理距离变换数据，但也可用于普通图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Union


def smoothstep_window(length: int, border: int, device=None) -> torch.Tensor:
    """
    创建平滑过渡的边界窗口函数，用于减少边缘伪影。
    内部区域为1，边界区域通过三次平滑步函数平滑过渡到0。

    参数：
        length: 窗口长度
        border: 边界宽度（每侧）
        device: 计算设备

    返回：
        形状为 (length,) 的张量
    """
    if border == 0:
        return torch.ones(length, device=device)

    # 创建索引
    idx = torch.arange(length, device=device, dtype=torch.float32)
    win = torch.ones(length, device=device, dtype=torch.float32)

    # 左侧边界：三次平滑步函数 3x^2 - 2x^3
    left = idx < border
    x = idx[left] / border
    win[left] = 3 * x ** 2 - 2 * x ** 3

    # 右侧边界
    right = idx >= (length - border)
    x = (length - idx[right] - 1) / border
    win[right] = 3 * x ** 2 - 2 * x ** 3

    return win


class EdgeDetectionPreprocessor(nn.Module):
    """
    可微分的边缘检测预处理器，专为距离变换数据设计。
    支持2D和3D数据，可同时应用多种边缘检测算法并加权融合结果。

    在电极生成项目中的应用：
        1. 输入电极体素数据（二值或距离变换）
        2. 提取孔隙-固相界面的边缘特征
        3. 将边缘特征作为条件输入扩散模型，引导生成结构清晰的电极

    初始化参数：
        dim: 数据维度，2表示2D图像，3表示3D体素（对您适用）
        processors: 边缘检测器列表，可选 'original', 'sobel', 'laplacian', 'gradient', 'morph'
        feature_weights: 各检测器的权重字典
        border_width: 边界平滑窗口的宽度，用于减少边缘伪影
    """

    def __init__(
            self,
            dim: int = 2,
            processors: Union[str, List[str]] = 'all',
            feature_weights: Optional[Dict[str, float]] = None,
            border_width: int = 8,
    ):
        super().__init__()

        # 数据维度：2D或3D
        self.dim = dim
        # 边界平滑宽度
        self.border_width = border_width

        # 定义所有可用的边缘检测器
        valid_processors = ['original', 'sobel', 'laplacian', 'gradient', 'morph']

        # 处理processors参数
        if processors == 'all':
            self.processors = valid_processors
        elif isinstance(processors, str):
            if processors not in valid_processors:
                raise ValueError(f"未知的处理器: {processors}")
            self.processors = [processors]
        else:
            for p in processors:
                if p not in valid_processors:
                    raise ValueError(f"未知的处理器: {p}")
            self.processors = list(processors)

        # 处理特征权重
        if feature_weights is None:
            feature_weights = {k: 1.0 for k in valid_processors}

        # 计算归一化权重
        selected_weights = [float(feature_weights.get(p, 1.0)) for p in self.processors]
        total_weight = sum(selected_weights)

        if total_weight == 0:
            self.normalized_weights = {p: 0.0 for p in self.processors}
        else:
            self.normalized_weights = {
                p: w / total_weight for p, w in zip(self.processors, selected_weights)
            }

        # 注册边缘检测卷积核（非可训练参数）
        self._register_kernels()

    def _register_kernels(self):
        """注册边缘检测所需的卷积核（2D和3D版本）"""

        if self.dim == 2:
            # ========== 2D卷积核 ==========

            # Sobel算子：检测边缘和梯度方向
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=torch.float32)

            # Laplacian算子：检测图像的二阶导数，对边缘更敏感
            laplacian = torch.tensor([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=torch.float32)

            # 结构张量核：用于计算梯度
            struct_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32)
            struct_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32)

        else:  # 3D
            # ========== 3D卷积核 ==========
            # 注意：这些是3D Sobel算子的近似实现

            # 3D Sobel X方向
            sobel_x = torch.tensor([[[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]],
                                    [[-2, 0, 2],
                                     [-4, 0, 4],
                                     [-2, 0, 2]],
                                    [[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]], dtype=torch.float32)

            # 3D Sobel Y方向
            sobel_y = torch.tensor([[[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]],
                                    [[-2, -4, -2],
                                     [0, 0, 0],
                                     [2, 4, 2]],
                                    [[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]], dtype=torch.float32)

            # 3D Sobel Z方向
            sobel_z = torch.tensor([[[-1, -2, -1],
                                     [-2, -4, -2],
                                     [-1, -2, -1]],
                                    [[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]],
                                    [[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]], dtype=torch.float32)

            # 3D Laplacian：中心为-6，6个相邻体素为1
            laplacian = torch.zeros(3, 3, 3, dtype=torch.float32)
            laplacian[1, 1, 1] = -6
            laplacian[0, 1, 1] = laplacian[2, 1, 1] = 1
            laplacian[1, 0, 1] = laplacian[1, 2, 1] = 1
            laplacian[1, 1, 0] = laplacian[1, 1, 2] = 1

            # 3D梯度核
            struct_x = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32)
            struct_y = torch.tensor([[[-1], [0], [1]]], dtype=torch.float32)
            struct_z = torch.tensor([[[-1]], [[0]], [[1]]], dtype=torch.float32)

        # 将卷积核注册为buffer（不参与训练，但会随模型保存/加载）
        # 添加批次和通道维度：[输出通道, 输入通道, *核尺寸]
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))
        self.register_buffer('struct_x', struct_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('struct_y', struct_y.unsqueeze(0).unsqueeze(0))

        if self.dim == 3:
            self.register_buffer('sobel_z', sobel_z.unsqueeze(0).unsqueeze(0))
            self.register_buffer('struct_z', struct_z.unsqueeze(0).unsqueeze(0))

    def _conv_nd(self, x: torch.Tensor, kernel: torch.Tensor, padding: str = 'same') -> torch.Tensor:
        """N维卷积封装，自动处理2D/3D"""
        if self.dim == 2:
            return F.conv2d(x, kernel, padding=padding)
        else:
            return F.conv3d(x, kernel, padding=padding)

    def sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sobel边缘检测：计算梯度幅度

        在电极数据中的应用：
            检测孔隙-固相界面的强度变化，突出相边界

        返回：
            梯度幅度图 [B, 1, D, H, W] 或 [B, 1, H, W]
        """
        # 应用Sobel卷积
        grad_x = self._conv_nd(x, self.sobel_x)
        grad_y = self._conv_nd(x, self.sobel_y)

        if self.dim == 3:
            grad_z = self._conv_nd(x, self.sobel_z)
            # 计算3D梯度幅度
            magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        else:
            # 计算2D梯度幅度
            magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 注：原始代码中可选择返回梯度方向信息，当前只返回幅度
        return torch.cat([magnitude], dim=1)

    def laplacian_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Laplacian边缘检测：计算二阶导数

        在电极数据中的应用：
            对边缘更敏感，可检测细小的孔隙结构

        返回：
            Laplacian响应 [B, 1, D, H, W] 或 [B, 1, H, W]
        """
        laplacian_response = self._conv_nd(x, self.laplacian)
        return torch.cat([laplacian_response], dim=1)

    def gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        直接梯度幅度检测：使用简单的梯度算子

        在电极数据中的应用：
            计算距离场的变化率，对距离变换数据特别有效

        返回：
            梯度幅度 [B, 1, D, H, W] 或 [B, 1, H, W]
        """
        grad_x = self._conv_nd(x, self.struct_x)
        grad_y = self._conv_nd(x, self.struct_y)

        if self.dim == 3:
            grad_z = self._conv_nd(x, self.struct_z)
            magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        else:
            magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return torch.cat([magnitude], dim=1)

    def morphological_gradient(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        形态学梯度：膨胀与腐蚀的差值

        在电极数据中的应用：
            检测二值图像的边界，对电极的孔隙结构边界敏感

        参数：
            kernel_size: 形态学操作的核大小

        返回：
            形态学梯度 [B, 1, D, H, W] 或 [B, 1, H, W]
        """
        if self.dim == 2:
            # 2D形态学操作
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
            dilated = F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
            eroded = -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)
        else:
            # 3D形态学操作
            kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=x.device)
            dilated = F.max_pool3d(x, kernel_size, stride=1, padding=kernel_size // 2)
            eroded = -F.max_pool3d(-x, kernel_size, stride=1, padding=kernel_size // 2)

        # 形态学梯度 = 膨胀 - 腐蚀
        morph_grad = dilated - eroded
        return morph_grad

    def _gradient_coherence_2d(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
        """计算2D梯度一致性（局部梯度方向的一致性）"""
        # 结构张量分量
        Jxx = grad_x * grad_x
        Jxy = grad_x * grad_y
        Jyy = grad_y * grad_y

        # 高斯平滑结构张量
        sigma = 1.0
        kernel_size = 5
        gaussian_kernel = self._gaussian_kernel_2d(kernel_size, sigma).to(grad_x.device)

        Jxx_smooth = self._conv_nd(Jxx, gaussian_kernel)
        Jxy_smooth = self._conv_nd(Jxy, gaussian_kernel)
        Jyy_smooth = self._conv_nd(Jyy, gaussian_kernel)

        # 一致性度量
        trace = Jxx_smooth + Jyy_smooth
        det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
        coherence = (trace - 2 * torch.sqrt(det + 1e-8)) / (trace + 1e-8)

        return coherence

    def _gradient_coherence_3d(self, grad_x: torch.Tensor, grad_y: torch.Tensor, grad_z: torch.Tensor) -> torch.Tensor:
        """计算3D梯度一致性（简化版）"""
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)

        # 归一化梯度方向
        gx_norm = grad_x / (magnitude + 1e-8)
        gy_norm = grad_y / (magnitude + 1e-8)
        gz_norm = grad_z / (magnitude + 1e-8)

        # 简单的一致性度量
        coherence = magnitude * (torch.abs(gx_norm) + torch.abs(gy_norm) + torch.abs(gz_norm))
        return coherence

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """创建2D高斯核"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2

        # 1D高斯分布
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # 外积得到2D核
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)

    def _make_window(self, shape: tuple, device) -> torch.Tensor:
        """
        创建N维平滑窗口：内部为1，边界平滑过渡到0

        参数：
            shape: 空间维度元组，如 (H, W) 或 (D, H, W)

        返回：
            窗口张量，形状与输入空间维度相同
        """
        # 为每个维度创建1D窗口
        windows = []
        for size in shape:
            windows.append(smoothstep_window(size, self.border_width, device))

        # 从第一个窗口开始
        window = windows[0]

        # 逐维度相乘，扩展维度以匹配
        for i, win in enumerate(windows[1:], 1):
            shape_dims = [1] * len(shape)
            shape_dims[i] = -1
            win = win.view(*shape_dims)
            window = window.unsqueeze(i) * win

        return window

    def _apply_border_window(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入应用边界窗口，减少边缘伪影

        参数：
            x: 输入张量 [B, C, H, W] 或 [B, C, D, H, W]

        返回：
            窗口化后的张量
        """
        if self.border_width is None or self.border_width <= 0:
            return x

        # 获取空间维度形状
        shape = x.shape[-2:] if self.dim == 2 else x.shape[-3:]

        # 创建窗口
        window = self._make_window(shape, x.device)

        # 扩展维度以匹配输入
        while window.dim() < x.dim():
            window = window.unsqueeze(0)

        return x * window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：应用选择的边缘检测方法并拼接结果

        参数：
            x: 输入距离变换或图像 [B, C, H, W] 或 [B, C, D, H, W]

        返回：
            边缘特征拼接 [B, n_features, H, W] 或 [B, n_features, D, H, W]

        在电极生成项目中的使用示例：
            # 假设有3D电极体素数据
            edge_detector = EdgeDetectionPreprocessor(dim=3, processors=['sobel', 'laplacian'])
            edge_features = edge_detector(electrode_voxels)  # 形状: [B, 2, D, H, W]

            # 可作为条件输入扩散模型
            model_input = torch.cat([latent, edge_features], dim=1)
        """
        features = []

        # 应用边界窗口（减少边缘伪影）
        x_windowed = self._apply_border_window(x)

        # 应用每个选定的处理器，并按权重加权
        for p in self.processors:
            weight = self.normalized_weights[p]

            if p == 'original':
                # 原始输入（可保留原始信息）
                features.append(x * weight)
            elif p == 'sobel':
                features.append(self.sobel_edges(x_windowed) * weight)
            elif p == 'laplacian':
                features.append(self.laplacian_edges(x_windowed) * weight)
            elif p == 'gradient':
                features.append(self.gradient_magnitude(x_windowed) * weight)
            elif p == 'morph':
                features.append(self.morphological_gradient(x_windowed) * weight)

        # 沿通道维度拼接所有特征
        return torch.cat(features, dim=1)