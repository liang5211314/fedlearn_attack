import logging
from defenses.fedavg import FedAvg
import utils.defense as defense
logger = logging.getLogger('logger')
import torch
import numpy as  np
class LayerKrum(FedAvg):
    current_epoch: int = 0
    log_distance: bool = False
    frac: float = 0.1
    wrong_mal: int = 1
    turn: int = 1
    mal_score: int = 0
    ben_score: int = 0
    krum_distance: []
    krum_layer_distance: []
    k: int = 2

    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch

    def aggr(self, weight_accumulator, global_model):

        # self.params.history_updates = dict()
        w_glob_update= self.layer_krum(self.params.fl_local_updated_models, self.k, multi_k=True)

        self.accumulate_weights(weight_accumulator, \
                                {key: (w_glob_update[key] ).to(self.params.device) for \
                                 key in w_glob_update})

        return weight_accumulator

    def layer_krum(self,gradients, n_attackers,multi_k=False):
        new_global = {}
        param_name, param_value = next(iter(gradients.items()))
        for layer in gradients[param_name].keys():
            if layer.split('.')[-1] == 'num_batches_tracked' or layer.split('.')[-1] == 'running_mean' or \
                    layer.split('.')[-1] == 'running_var':
                param_name,_ = next(iter(gradients.items()))
                new_global[layer] = gradients[param_name][layer]
            else:
                # Create an empty list to store the layer gradients
                layer_gradients = []

                # Loop over each element in the gradients list
                for value in gradients:
                    # Extract the gradient corresponding to the layer key from each element (x)
                    layer_gradient = gradients[value][layer]

                    # Append the layer gradient to the layer_gradients list
                    layer_gradients.append(layer_gradient)
                # layer_gradients = [x[layer] for x in gradients]
                new_global[layer] = self.layer_multi_krum(layer_gradients, n_attackers,  multi_k)
        return new_global

    def layer_multi_krum(self,layer_gradients, n_attackers, multi_k=False):
        grads = self.layer_flatten_grads(layer_gradients)
        candidates = []
        candidate_indices = []
        remaining_updates = torch.from_numpy(grads)
        all_indices = np.arange(len(grads))
        score_record = None
        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            # scores = None
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
                layer_distance_dict = self.log_layer_wise_distance(layer_gradients)
                self.krum_layer_distance.append(layer_distance_dict)
                # print('defense.py line149 (layer_distance_dict):', layer_distance_dict)
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
        agg_layer = 0
        for selected_layer in candidate_indices:
            agg_layer += layer_gradients[selected_layer]
        agg_layer /= len(candidate_indices)
        return agg_layer

    def layer_flatten_grads(self,gradients):
        flat_epochs = []
        for n_user in range(len(gradients)):
            # user_arr = []
            # grads = gradients[n_user]
            flat_epochs.append(gradients[n_user].cpu().numpy().flatten().tolist())
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
