#callbacks.py

"""EMACallback指数移动平均，用于维护在训练中模型参数的平滑版本"""
"""ScheduleFreeCallback用于配合无学习率调度优化器"""
"""NanToZeroGradCallback用于在优化器更新参数之前检查参数的梯度，将NAN换为0 ，保证安全不崩溃"""

import lightning.pytorch.callbacks as pl_callbacks
import torch
from torch import Tensor


class EMACallback(pl_callbacks.StochasticWeightAveraging):
    def __init__(self, decay=0.99):
        super().__init__(decay)
        self.decay = decay

    def avg_fn(self,
               averaged_model_parameter: torch.Tensor,
               model_parameter: torch.Tensor,
               num_averaged: torch.LongTensor) -> Tensor:
        e = averaged_model_parameter
        m = model_parameter
        return self.decay * e + (1. - self.decay) * m


class ScheduleFreeCallback(pl_callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.train()
        if hasattr(pl_module.optimizer, "train"):
            pl_module.optimizer.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.model.eval()
        if hasattr(pl_module.optimizer, "eval"):
            pl_module.optimizer.eval()

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.model.eval()
        if hasattr(pl_module.optimizer, "eval"):
            pl_module.optimizer.eval()


class NanToZeroGradCallback(pl_callbacks.Callback):
    def on_before_optimizer_step(self,
                                 trainer,
                                 pl_module,
                                 optimizer):
        for p in pl_module.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)