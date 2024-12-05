import logging
from defenses.fedavg import FedAvg
import utils.defense as defense
import copy
import torch
import numpy as np
from sklearn.metrics import accuracy_score  # 用于计算精度

logger = logging.getLogger('logger')

class Crfl(FedAvg):
    current_epoch: int = 0
    sigma: float = 0.01
    threshold: float = 1.0

    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch

    def norm_clipping(self, gradients, threshold=1.0):
        norm = torch.norm(gradients)
        if norm > threshold:
            gradients = gradients * (threshold / norm)  # 对梯度进行裁剪
        return gradients.to(self.params.device)

    def add_gaussian_noise(self, gradients, sigma=0.1, dynamic_sigma=False):
        """ 给客户端更新添加高斯噪声 """
        if dynamic_sigma:
            sigma = torch.norm(gradients) * 0.01  # 根据梯度的范数动态调整噪声
        noise = torch.normal(mean=0, std=sigma, size=gradients.shape).to(self.params.device)
        return gradients + noise

    def create_perturbed_models(self, global_model, num_models=5, sigma=0.1):
        perturbed_models = []
        for _ in range(num_models):
            perturbed_model = copy.deepcopy(global_model)
            for param in perturbed_model.parameters():
                param.data += torch.normal(mean=0, std=sigma, size=param.size()).to(self.params.device)
            perturbed_models.append(perturbed_model)
        return perturbed_models

    def majority_voting(self, predictions):
        predictions = [torch.tensor(pred, device=self.params.device) if not isinstance(pred, torch.Tensor) else pred for
                       pred in predictions]

        predictions = torch.stack(predictions, dim=0)

        # 获取众数及其索引
        values, indices = torch.mode(predictions, dim=0)

        # 只返回众数作为最终预测
        logger.info(f"Values (mode): {values}")

        return values

    def evaluate_model(self, model, data_loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for user_data, user_labels in data_loader:
                user_data, user_labels = user_data.to(self.params.device), user_labels.to(self.params.device)
                output = model(user_data)
                pred = output.argmax(dim=1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(user_labels.cpu().numpy())
        # Flatten the lists to compute accuracy
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def aggr(self, weight_accumulator, global_model):
        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            loaded_params = self.params.fl_local_updated_models[user_id]

            # 对每个客户端的模型更新进行Norm Clipping
            for key in loaded_params:
                loaded_params[key] = self.norm_clipping(loaded_params[key], threshold=self.threshold)

            # 添加高斯噪声
            for key in loaded_params:
                loaded_params[key] = self.add_gaussian_noise(loaded_params[key], sigma=self.sigma)

            # 将加权后的客户端更新累加到全局模型
            self.accumulate_weights(weight_accumulator,
                                    {key: (loaded_params[key] * weight_contrib_user).to(self.params.device) for key in
                                     loaded_params})

        # 生成多个扰动模型
        perturbed_models = self.create_perturbed_models(global_model, num_models=5, sigma=self.sigma)

        best_model = None
        best_accuracy = 0

        # 评估每个扰动模型的精度
        for perturbed_model in perturbed_models:
            accuracy = self.evaluate_model(perturbed_model, self.params.fl_data)
            logger.info(f"Perturbed model accuracy: {accuracy}")

            # 选择精度最高的模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = perturbed_model

        # 将精度最高的模型作为最终的全局模型
        logger.info(f"Best model selected with accuracy: {best_accuracy}")

        # 将 best_model 的参数更新累加到全局模型
        for param_name, param_value in best_model.named_parameters():
            if param_name in global_model.state_dict():
                global_model.state_dict()[param_name].data.copy_(param_value.data)

        return weight_accumulator

