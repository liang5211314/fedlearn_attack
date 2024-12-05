import math
from typing import List, Any, Dict
import torch
import logging
import os
from defenses.fedavg import FedAvg
from utils.parameters import Params


import torch
import torch.nn.functional as F

class RLR(FedAvg):
    update_threshold: float = 1e-3
    max_weight_contrib: float = 1.0  # 限制最大权重贡献
    min_weight_contrib: float = -1.0  # 限制最小权重贡献
    max_grad_norm: float = 1.0  # 限制梯度范数

    def __init__(self, params: Params) -> None:
        self.params = params

    def aggr(self, weight_accumulator, global_model):
        # 确保所有参数在相同设备上
        device = self.params.device
        global_params = global_model.state_dict()

        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            loaded_params = self.params.fl_local_updated_models[user_id]

            # 计算本地更新与全局模型更新的方向一致性
            local_update_direction = self._compute_update_direction(loaded_params, global_params)

            # 检测是否是恶意客户端（例如，如果方向不一致，则认为是恶意客户端）
            if self._is_malicious_update(local_update_direction):
                # 如果是恶意客户端，则翻转其学习率，并限制在合理范围内
                weight_contrib_user = -weight_contrib_user

            # 限制权重贡献值在合理范围内，避免出现过大或过小的数值
            weight_contrib_user = self._clamp_weight_contrib(weight_contrib_user)

            # 进行加权聚合
            self.accumulate_weights(weight_accumulator,
                                    {key: (loaded_params[key] * weight_contrib_user).to(device) for key in
                                     loaded_params})

        # 在每轮聚合后对梯度进行裁剪（防止梯度爆炸）
        self._clip_gradients(global_model)

        return weight_accumulator

    def _clamp_weight_contrib(self, weight_contrib_user):
        """处理 weight_contrib_user 的限制"""
        if isinstance(weight_contrib_user, torch.Tensor):
            # 对于 Tensor 类型的输入，直接进行 clamp 操作
            return torch.clamp(weight_contrib_user, min=self.min_weight_contrib, max=self.max_weight_contrib)
        else:
            # 对于标量类型的输入，先转换为 Tensor，再进行 clamp 操作
            return torch.clamp(torch.tensor(weight_contrib_user, dtype=torch.float), min=self.min_weight_contrib, max=self.max_weight_contrib)

    def _clip_gradients(self, model):
        """裁剪梯度防止梯度爆炸"""
        for param in model.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, self.max_grad_norm)

    def _compute_update_direction(self, loaded_params, global_params):
        # 计算本地更新与全局模型之间的方向一致性
        direction = {}
        for key in loaded_params:
            # 确保使用同样的设备进行计算
            direction[key] = (loaded_params[key] - global_params[key]).flatten()
        return direction

    def _is_malicious_update(self, local_update_direction):
        # 简单的检查：例如，如果本地更新方向与全局方向不一致，认为是恶意更新
        for key in local_update_direction:
            if torch.norm(local_update_direction[key]) < self.update_threshold:
                return False  # 如果更新方向非常小，认为是正常
        return True  # 否则认为是恶意
