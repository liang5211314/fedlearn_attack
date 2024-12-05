import os
from datetime import datetime

str_write_to_file = ""

def generate_exps_file(root_file='./cifar_fed.yaml', name_exp = 'cifar10', EXPS_DIR="./exps/extras"):
    # read file as a string
    global str_write_to_file
    print(f'reading from root_file: {root_file}')
    str_write_to_file += f'# Dataset: {name_exp} \n**Reading default config from root_file: {root_file}**\n\n'
    str_write_to_file += '------------------------\n\n'

    fl_total_participants_choices = [100, 200]
    fl_no_models_choices = [10, 20]
    fl_dirichlet_alpha_choices = [0.5]
    fl_number_of_adversaries_choices = [4]
    fl_lr_choices = [0.05]
    resume_model_choices = [False, True]
    lis_resume_model = [
        'resume_model: saved_models/tiny_64_pretrain/tiny-resnet.epoch_20',
        'resume_model: saved_models/mnist_pretrain/model_last.pt.tar.epoch_10',
        'resume_model: saved_models/cifar_pretrain/model_last.pt.tar.epoch_200',
    ]
    
    # EXPS_DIR = './exps/extras'
    
    os.makedirs(EXPS_DIR, exist_ok=True)
    exp_number = 0

    for fl_total_participants in fl_total_participants_choices:
        for fl_no_models in fl_no_models_choices:
            for fl_dirichlet_alpha in fl_dirichlet_alpha_choices:
                for fl_number_of_adversaries in fl_number_of_adversaries_choices:
                    for fl_lr in fl_lr_choices:
                        for resume_model in resume_model_choices:

                            with open(root_file, 'r') as file :
                                filedata = file.read()
                            
                            exp_number += 1
                            str_write_to_file += f'## EXP ID: {exp_number:02d}\n'
                            pretrained_str = 'pretrained' if resume_model else 'no_pretrained'
                            print(f'`fl_total_participants: {fl_total_participants} fl_no_models: {fl_no_models} fl_dirichlet_alpha: {fl_dirichlet_alpha} fl_number_of_adversaries: {fl_number_of_adversaries} fl_lr: {fl_lr} resume_model: {resume_model}`\n\n')
                            str_write_to_file += f'fl_total_participants: {fl_total_participants} fl_no_models: {fl_no_models} fl_dirichlet_alpha: {fl_dirichlet_alpha} fl_number_of_adversaries: {fl_number_of_adversaries} fl_lr: {fl_lr} resume_model: {resume_model}\n\n'
                            filedata = filedata.replace('fl_total_participants: 100', f'fl_total_participants: {fl_total_participants}')
                            filedata = filedata.replace('fl_no_models: 10', f'fl_no_models: {fl_no_models}')
                            filedata = filedata.replace('fl_dirichlet_alpha: 0.5', f'fl_dirichlet_alpha: {fl_dirichlet_alpha}')
                            filedata = filedata.replace('fl_number_of_adversaries: 4', f'fl_number_of_adversaries: {fl_number_of_adversaries}')
                            filedata = filedata.replace('lr: 0.005', f'lr: {fl_lr}')
                            # print(filedata)

                            if not resume_model:
                                for resume_model_path in lis_resume_model:
                                    if resume_model_path in filedata:
                                        filedata = filedata.replace(resume_model_path, f'resume_model: \n')
                            # print?(filedata)
                            # exit(0)                                    
                            # print(len(filedata), type(filedata))  
                            # print('------------------------')
                            # write the file out again
                            
                            fn_write = f'{EXPS_DIR}/{name_exp}_fed_{fl_total_participants}_{fl_no_models}_{fl_number_of_adversaries}_{fl_dirichlet_alpha}_{fl_lr}_{pretrained_str}.yaml'

                            if not os.path.exists(fn_write):
                                with open(fn_write, 'w') as file:
                                    file.write(filedata)
                                
                            cmd = f'```bash\nCUDA_VISIBLE_DEVICES=0 python training.py --name {name_exp} --params {fn_write}\n```\n'
                            print(cmd)
                            str_write_to_file += f'{cmd}\n'
                            print('------------------------')
                            # if exp_number == 2:
                            #     exit(0)
                    

current_time = datetime.now().strftime('%Y.%b.%d')

generate_exps_file(root_file='./cifar_fed.yaml',
                   name_exp = 'cifar10', 
                   EXPS_DIR=f"./run_cifar10__{current_time}")


generate_exps_file(root_file='./mnist_fed.yaml',
                   name_exp = 'mnist', 
                   EXPS_DIR=f"./run_mnist__{current_time}")

generate_exps_file(root_file='./imagenet_fed.yaml',
                   name_exp = 'tiny-imagenet', 
                   EXPS_DIR=f"./run_tiny-imagenet__{current_time}")

with open(f'./run_exps__{current_time}.md', 'w') as file:
    file.write(str_write_to_file)

