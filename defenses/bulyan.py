import logging
from defenses.fedavg import FedAvg
import utils.defense as defense
import copy
import torch
import numpy as np
from sklearn.metrics import accuracy_score  # 用于计算精度

logger = logging.getLogger('logger')

class Bulyan(FedAvg):
    current_epoch: int = 0
    f:int = 2

    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch

    def aggr(self, weight_accumulator, global_model):
        # 收集所有客户端的模型更新以及它们的权重贡献
        updated_models = {}
        weights_contrib = {}

        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            loaded_params = self.params.fl_local_updated_models[user_id]
            updated_models[user_id] = loaded_params
            weights_contrib[user_id] = weight_contrib_user

        # 步骤 1: 排除恶意客户端
        # 计算客户端间的欧氏距离，选择出最远的 2f 个客户端进行排除
        distances = {}
        for user_id_1, params_1 in updated_models.items():
            for user_id_2, params_2 in updated_models.items():
                if user_id_1 != user_id_2:
                    # 计算欧氏距离 (L2 distance)
                    distance = sum(torch.norm(params_1[key] - params_2[key]).item() ** 2 for key in params_1)
                    distances[(user_id_1, user_id_2)] = distance

        # 根据欧氏距离排序，排除成对距离最大的 2f 个客户端
        # 这需要你决定 f 的值，并且根据它选择需要排除的客户端对
        f = 2  # 根据实验设置 f = 2
        excluded_users = set()
        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        for (user_id_1, user_id_2), _ in sorted_distances[:2 * f]:
            excluded_users.add(user_id_1)
            excluded_users.add(user_id_2)

        # 步骤 2: 选择最接近中值的客户端
        remaining_models = {user_id: updated_models[user_id] for user_id in updated_models if
                            user_id not in excluded_users}
        remaining_weights = {user_id: weights_contrib[user_id] for user_id in remaining_models}

        # 计算全局中值（按每个参数维度计算中值）
        global_median = {}
        for key in next(iter(remaining_models.values())).keys():
            median_values = []
            for user_id, params in remaining_models.items():
                median_values.append(params[key].cpu().numpy())
            global_median[key] = torch.tensor(np.median(median_values, axis=0), device=self.params.device)

        # 步骤 3: 按照与中值的距离选择最接近中值的 M - 4f 个客户端
        # 计算每个客户端更新与全局中值的距离
        distances_to_median = {}
        for user_id, params in remaining_models.items():
            distance = sum(torch.norm(params[key] - global_median[key]).item() ** 2 for key in params)
            distances_to_median[user_id] = distance

        # 按距离升序排序，选择最接近的 M - 4f 个客户端
        M = len(remaining_models) + len(excluded_users)  # 总客户端数
        selected_users = sorted(distances_to_median, key=distances_to_median.get)[:M - 4 * f]

        # 步骤 4: 聚合选择的客户端
        # 确保 weight_accumulator 是一个字典，并且每个张量都被零初始化
        for key in weight_accumulator:
            weight_accumulator[key].zero_()  # 对每个张量进行零初始化

        for user_id in selected_users:
            loaded_params = remaining_models[user_id]
            weight_contrib_user = remaining_weights[user_id]
            self.accumulate_weights(weight_accumulator,
                                    {key: (loaded_params[key] * weight_contrib_user).to(self.params.device) for key in
                                     loaded_params})

        return weight_accumulator
