import logging
from defenses.fedavg import FedAvg
import utils.defense as defense
logger = logging.getLogger('logger')
import torch
import numpy as  np
class Clip(FedAvg):
    current_epoch: int = 0
    clip_factor: int=1
    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch
    def aggr(self, weight_accumulator, global_model):
        # print(weight_accumulator)
        # print("=========================")
        self.clip_updates(weight_accumulator)
        # print(weight_accumulator)
        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            # logger.info(f"Aggregating participant: {user_id} with weight: {weight_contrib_user}")
            loaded_params = self.params.fl_local_updated_models[user_id]
            self.accumulate_weights(weight_accumulator, \
                {key:(loaded_params[key] * weight_contrib_user ).to(self.params.device) for \
                    key in loaded_params})
        return weight_accumulator
    def clip_updates(self, agent_updates_dict):
        for key in agent_updates_dict:
            if 'num_batches_tracked' not in key:
                update = agent_updates_dict[key]
                l2_update = torch.norm(update, p=2)
                update.div_(max(1, l2_update/self.clip_factor))

        return

