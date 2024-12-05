import math
from typing import List, Any, Dict
import torch
import logging
import os
from defenses.fedavg import FedAvg
from utils.parameters import Params


class  SparseFed(FedAvg):
    k: float = 0.30

    def __init__(self, params: Params) -> None:
        self.params = params

    def aggr(self, weight_accumulator, global_model):
        # Iterate over each user's model update
        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            # Load the local updated model for the current user
            loaded_params = self.params.fl_local_updated_models[user_id]

            # Apply the weight contribution and compute the absolute values of the parameters
            weighted_params = {key: (loaded_params[key] * weight_contrib_user).to(self.params.device)
                               for key in loaded_params}

            # Sort the parameters by their absolute value and select the top k
            for key in weighted_params:
                # Compute absolute values of the tensor elements
                param_abs_values = weighted_params[key].abs()

                # Sort the indices based on absolute value
                top_k_indices = torch.topk(param_abs_values.view(-1),
                                           k=int(len(param_abs_values.view(-1)) * self.k)).indices

                # Create a mask for the top k indices
                mask = torch.zeros_like(param_abs_values.view(-1), dtype=torch.bool)
                mask[top_k_indices] = 1

                # Reshape the mask back to the original tensor shape
                mask = mask.view(weighted_params[key].shape)

                # Apply the mask to retain only the top k elements
                weighted_params[key] = weighted_params[key] * mask

            # Accumulate the weighted parameters into the global model
            self.accumulate_weights(weight_accumulator, weighted_params)

        return weight_accumulator