import math
from typing import List, Any, Dict
import torch
import logging
import os
from defenses.fedavg import FedAvg
from utils.parameters import Params


class DP(FedAvg):
    sigma: float=0.02

    def __init__(self, params: Params) -> None:
        self.params = params

    def aggr(self, weight_accumulator, global_model):

        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            # 获取该用户的本地更新参数
            loaded_params = self.params.fl_local_updated_models[user_id]

            # 为每个权重添加高斯噪声
            noisy_params = {
                key: (loaded_params[key] * weight_contrib_user).to(self.params.device) + \
                     torch.normal(0, self.sigma, size=loaded_params[key].size()).to(self.params.device)
                for key in loaded_params
            }

            # 累加加权后的噪声参数
            self.accumulate_weights(weight_accumulator, noisy_params)

        return weight_accumulator