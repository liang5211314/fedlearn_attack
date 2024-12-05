import argparse
import yaml
from helper import Helper
from datetime import datetime
from tqdm import tqdm
import wandb
from copy import deepcopy
from utils.utils import *
import numpy as np
logger = logging.getLogger('logger')
import math
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=False, global_model=None):
    criterion = hlpr.task.criterion
    model.train()
    # for i, data in tqdm(enumerate(train_loader)):
    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        if hlpr.params.defense=="RLR":
        # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        # import IPython; IPython.embed()
        # exit(0)
        # print(f"Epoch {epoch} batch {i} loss {loss.item()}")


                # norm_scale = min(1, self.conf['C'] / (model_norm))
                # # print(model_norm, norm_scale)
                # for name, layer in self.local_model.named_parameters():
                #     clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
                #     layer.data.copy_(model.state_dict()[name] + clipped_difference)


        if i == hlpr.params.max_batch_id:
            break
    # metric = hlpr.task.report_metrics(epoch,
    #                          prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return

def test(hlpr: Helper, epoch, backdoor=False, model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric

def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    hlpr.params.round_participants=round_participants
    weight_accumulator = hlpr.task.get_empty_accumulator()
    
    logger.info(f"Round epoch {epoch} with participants: {[user.user_id for user in round_participants]} and weight: {hlpr.params.fl_weight_contribution}")
    # log number of sample per user
    logger.info(f"Round epoch {epoch} with participants sample size: {[user.number_of_samples for user in round_participants]} and sum: {sum([user.number_of_samples for user in round_participants])}")
    
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        if user.compromised:
            # if not user.user_id == 0:
            #     continue
            
            logger.warning(f"Compromised user: {user.user_id} in run_fl_round {epoch}")
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)): # fl_poison_epochs)):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, global_model=global_model)
                
        else:
            logger.warning(f"Non-compromised user: {user.user_id} in run_fl_round {epoch}")
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        
        # hlpr.save_update(model=local_update, userID=user.user_id)
        # Do not save model to files, save it as a variable
        hlpr.task.adding_local_updated_model(local_update = local_update, user_id=user.user_id)
        
        if user.compromised:
            # hlpr.attack.perform_attack(global_model, user, epoch)
            hlpr.attack.local_dataset = deepcopy(user.train_loader)
            if hlpr.attack == 'ThrDFed':
                hlpr.attack.perform_attack(global_model,user, epoch)
    if hlpr.attack=="ModelReplace":
        hlpr.attack.perform_attack(global_model, epoch)
    hlpr.defense.aggr(weight_accumulator, global_model, )
    # logger.info(f"Round {epoch} update global model")
    
    hlpr.task.update_global_model(weight_accumulator, global_model)

def run(hlpr: Helper):
    metric = test(hlpr, -1, backdoor=False)
    logger.info(f"Before training main metric: {metric}")

    # 初始化平滑处理的列表
    smoothed_main_acc = []
    smoothed_main_loss = []
    smoothed_backdoor_acc = []
    smoothed_backdoor_loss = []

    # 设置窗口大小
    window_size = 20

    for epoch in range(hlpr.params.start_epoch, hlpr.params.epochs + 1):
        logger.info(f"Communication round {epoch}")
        run_fl_round(hlpr, epoch)

        # 测试主指标
        metric = test(hlpr, epoch, backdoor=False)
        main_metric = hlpr.task.get_metrics()

        # 测试后门指标
        metric_bd = test(hlpr, epoch, backdoor=True)
        backdoor_metric = hlpr.task.get_metrics()

        # 更新平滑处理
        smoothed_main_acc.append(main_metric['accuracy'])
        smoothed_main_loss.append(main_metric['loss'])
        smoothed_backdoor_acc.append(backdoor_metric['accuracy'])
        smoothed_backdoor_loss.append(backdoor_metric['loss'])

        # 计算并记录移动平均
        if len(smoothed_main_acc) > window_size:
            smoothed_main_acc.pop(0)
            smoothed_main_loss.pop(0)
            smoothed_backdoor_acc.pop(0)
            smoothed_backdoor_loss.pop(0)

        avg_main_acc = np.mean(smoothed_main_acc)
        avg_main_loss = np.mean(smoothed_main_loss)
        avg_backdoor_acc = np.mean(smoothed_backdoor_acc)
        avg_backdoor_loss = np.mean(smoothed_backdoor_loss)

        # 记录到 wandb
        wandb.log({
            'main_acc': avg_main_acc,
            'main_loss': avg_main_loss,
            'backdoor_acc': avg_backdoor_acc,
            'backdoor_loss': avg_backdoor_loss
        }, step=epoch)

        logger.info(f"Epoch {epoch} backdoor metric: {metric}")

        hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)
        hlpr.save_model(hlpr.task.model, epoch, metric)
        
def generate_exps_file(root_file='cifar_fed.yaml'):
    # read file as a string
    with open(root_file, 'r') as file :
        filedata = file.read()
    print(len(filedata), type(filedata))    
    pass        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    # python training.py --name mnist --params exps/mnist_fed.yaml
    # python training.py --name tiny-imagenet-200 --params exps/imagenet_fed.yaml
    # python training.py --name cifar10 --params exps/cifar_fed.yaml
    args = parser.parse_args()
    print(args)
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # print(params)
    # import IPython; IPython.embed()
    pretrained_str = 'pretrained' if params['resume_model'] else 'no_pretrained'
    
    params['name'] = f'vishc_{args.name}.{params["synthesizer"]}.{params["fl_total_participants"]}_{params["fl_no_models"]}_{params["fl_number_of_adversaries"]}_{params["fl_dirichlet_alpha"]}_{params["lr"]}_{pretrained_str}'
    
    params['current_time'] = datetime.now().strftime('%Y.%b.%d_%H.%M.%S')
    print(params)
    # exit(0)
    helper = Helper(params)
    
    # logger = create_logger()
    
    # logger.info(create_table(params))
    
    wandb.init(project="second-benchmark-backdoor-fl", entity="", name=f"{params['name']}-{params['current_time']}")
    try:
        run(helper)
    except Exception as e:
        print(e)
    
    