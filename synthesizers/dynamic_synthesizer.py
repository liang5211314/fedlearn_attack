import random
import torch
from torchvision.transforms import transforms
import numpy as np
from synthesizers.synthesizer import Synthesizer
from tasks.task import Task
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision.utils as vutils
import os
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()
import cv2
nz = 255
numOfClasses = 10
BDSize = 5
class DynamicSynthesizer(Synthesizer):

    def __init__(self, task: Task):
        super().__init__(task)
        self.backdoor_label = self.params.backdoor_label
        self.device=self.params.device
        self.epochs=50
        self.bdModel = hiddenNet().to(self.device)


    def synthesize_inputs(self, batch, test=False, attack_portion=0.1):
        """
        Apply triggers to a portion of the batch using a trained model.
        """
        # 1. 加载训练好的模型
        model_path = "./result_models/cifar10_cnn.pth"
        model = Net().to(self.device)  # Ensure model is moved to the correct device
        model.load_state_dict(torch.load(model_path))  # Load the saved model weights
        model.eval()  # Set model to evaluation mode

        # 2. 生成带有后门触发器的攻击数据
        modified_images = []
        batch_size = batch.batch_size

        # 计算要攻击的数据比例
        num_attack_images = attack_portion

        # 3. 遍历批次中的每张图像
        for i in range(batch_size):
            image = batch.inputs[i]
            label = batch.labels[i]

            # 只有部分图像会被加入后门
            if i <= num_attack_images:
                # 生成后门触发器
                noise = torch.rand(1, 255).to(self.device)  # 随机噪声
                targetOneHotEncoding = convertToOneHotEncoding(torch.tensor([label]), numOfClasses).to(self.device)

                # 使用训练好的生成器生成后门触发器
                with torch.no_grad():
                    backdoor_trigger = self.bdModel(targetOneHotEncoding, noise).view(-1, 3, BDSize, BDSize)

                # 将生成的后门触发器插入到图像中
                modified_image = insertSingleBD(image.unsqueeze(0), backdoor_trigger, label).squeeze(0)
                modified_images.append(modified_image)
            else:
                # 如果不是攻击图像，则保留原始图像
                modified_images.append(image)

        # 4. 将修改后的图像组成新的批次
        batch.inputs = torch.stack(modified_images)

        return batch

    def synthesize_labels(self, batch, attack_portion=0.1):
        """
        Modify labels for the backdoor attack.
        """
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)
        return batch




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class hiddenNet(nn.Module):
    def __init__(self, numOfClasses=numOfClasses):
        super(hiddenNet, self).__init__()
        self.fc0 = nn.Linear(numOfClasses, 64)
        self.fc1 = nn.Linear(nz, 64)
        self.fc11 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3 * BDSize * BDSize)

    def forward(self, c, x):
        xc = self.fc0(c)
        xx = self.fc1(x)
        gen_input = torch.cat((xc, xx), -1)
        x = self.fc11(gen_input)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output = F.sigmoid(x)
        return output


def convertToOneHotEncoding(c, numOfClasses=numOfClasses):
    oneHotEncoding = (torch.zeros(c.shape[0], numOfClasses))
    oneHotEncoding[:, c] = 1
    oneHotEncoding = oneHotEncoding
    return oneHotEncoding


def transformImg(image, scale=1):
    transformIt = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    images = image.clone()
    for i in images:
        i = transformIt(i)
    return (images)


def insertSingleBD(image, BD, label, scale=1):
    transformIt = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 直接复制image，不进行in-place操作
    images = image.clone()

    new_images = []  # 存储带后门图像的列表
    for i, bdSingle in zip(images, BD):
        xPos = np.random.randint(20)
        if label < 5:
            x = 5 + xPos
            pos = (label * BDSize) + BDSize
        else:
            x = 30 - xPos
            pos = ((label - 5) * BDSize) + BDSize

        # 使用新张量来存储修改后的图像
        modified_image = i.clone()
        modified_image[:, (x - BDSize):x, (pos - BDSize):pos] = (scale * bdSingle)
        modified_image = transformIt(modified_image)
        new_images.append(modified_image)

    # 将带后门的图像拼接成新的张量
    return torch.stack(new_images)
def train(model, device, train_loader, optimizer, epoch, bdModel, optimizerBD):
    model.train()
    bdModel.train()

    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        # noise = torch.rand(batch_size, nz)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        optimizerBD.zero_grad()

        lossBD = 0
        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size).long().to(device) * i
            targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
            backDoors = (bdModel(targetOneHotEncoding, noise)).view(-1, 3, BDSize, BDSize)
            dataBD = insertSingleBD(data.detach(), backDoors, i)
            outputBD = model(dataBD)
            lossBD = lossBD + criterion(outputBD, targetBDBatch)
        lossBD.backward()
        optimizerBD.step()

        dataNorm = transformImg(data.detach())
        output = model(dataNorm)
        lossTarget = criterion(output, target)

        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size).long().to(device) * i
            targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
            backDoors = (bdModel(targetOneHotEncoding, noise)).view(-1, 3, BDSize, BDSize)

            dataBD = insertSingleBD(data, backDoors, i)
            outputBD = model(dataBD)

            lossTarget = lossTarget + criterion(outputBD, targetBDBatch)

        lossTarget.backward()
        optimizer.step()





