import logging
from defenses.fedavg import FedAvg
import utils.defense as defense
logger = logging.getLogger('logger')
import torch
import numpy as  np
class MultiKrum(FedAvg):
    current_epoch: int = 0
    log_distance: bool = False
    frac: float = 0.1
    wrong_mal: int = 1
    turn: int = 1
    mal_score: int = 0
    ben_score: int = 0
    krum_distance:[]
    krum_layer_distance:[]
    k: int=2
    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch
    def aggr(self, weight_accumulator, global_model):

        self.params.history_updates = dict()
        selected_client = self.multi_krum(self.params.fl_local_updated_models, self.k, multi_k=True)
        tempcon={}
        tmpadd=0
        # 重新计算权重贡献
        for index,user_id in enumerate(self.params.fl_weight_contribution):
            if index in selected_client:
                tempcon[user_id]=self.params.fl_weight_contribution[user_id]
                tmpadd+=self.params.fl_weight_contribution[user_id]
        for user_id in tempcon:
            tempcon[user_id]=tempcon[user_id]/tmpadd

        # w_glob = w_locals[selected_client[0]]
        #         weight_accumulator[name].add_((wv[i]*data).to(self.params.device))
        for index, user in enumerate(self.params.round_participants):
            if index in selected_client:
                update_params = self.params.fl_local_updated_models[user.user_id]
                self.accumulate_weights(weight_accumulator, \
                                            {key: (update_params[key]*tempcon[user.user_id] ).to(self.params.device) for \
                                             key in update_params})
                    # weight_accumulator=self.params.fl_local_updated_models[user.user_id]

        return weight_accumulator

    def multi_krum(self,gradients, n_attackers, multi_k=False):

        grads = self.flatten_grads(gradients)
        candidates = []
        candidate_indices = []
        remaining_updates = torch.from_numpy(grads)
        all_indices = np.arange(len(grads))

        score_record = None

        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(
                    distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(
                distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)

            if self.log_distance == True and score_record == None:
                print('defense.py line149 (krum distance scores):', scores)
                score_record = scores
                self.krum_distance.append(scores)
                layer_distance_dict = self.log_layer_wise_distance(gradients)
                self.krum_layer_distance.append(layer_distance_dict)
            indices = torch.argsort(scores)[:len(
                remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(
                candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat(
                (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break

        num_clients = max(int(self.frac * self.params.fl_total_participants), 1)
        num_malicious_clients = self.params.fl_number_of_adversaries
        self.turn += 1
        for selected_client in candidate_indices:
            if selected_client < num_malicious_clients:
                self.wrong_mal += 1

        for i in range(len(scores)):
            if i < num_malicious_clients:
                self.mal_score += scores[i]
            else:
                self.ben_score += scores[i]

        return np.array(candidate_indices)

    def flatten_grads(self,gradients):
        _, param_order = next(iter(gradients.items()))
        # param_order = gradients[0].keys()

        flat_epochs = []

        for index, n_user in enumerate(gradients):
            user_arr = []
            grads = gradients[n_user]
            for param in param_order:
                try:
                    user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
                except:
                    user_arr.extend(
                        [grads[param].cpu().numpy().flatten().tolist()])
            flat_epochs.append(user_arr)

        flat_epochs = np.array(flat_epochs)

        return flat_epochs

    def log_layer_wise_distance(self,updates):
        # {layer_name, [layer_distance1, layer_distance12...]}
        layer_distance = {}
        for layer, val in updates[0].items():
            if 'num_batches_tracked' in layer:
                continue
            # for each layer calculate distance among models
            for model in updates:
                temp_layer_dis = 0
                for model2 in updates:
                    temp_norm = torch.norm((model[layer] - model2[layer]))
                    temp_layer_dis += temp_norm
                if layer not in layer_distance.keys():
                    layer_distance[layer] = []
                layer_distance[layer].append(temp_layer_dis.item())
        return layer_distance