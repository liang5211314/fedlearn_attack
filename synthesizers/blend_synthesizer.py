import random
import torch
from torchvision.transforms import transforms
import numpy as np
from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()
import cv2

class BlendSynthesizer(Synthesizer):
    mask: float=0.3
    def __init__(self, task: Task):
        super().__init__(task)
        self.backdoor_label = self.params.backdoor_label


    def add_trigger(self, image, label, test=False):
        self.hallokitty = cv2.imread('./synthesizers/mask/halloKitty.png')
        pixel_max = 255
        self.hallokitty = self.hallokitty.astype(np.float64) / pixel_max
        self.hallokitty = torch.from_numpy(self.hallokitty)

        # 确保所有张量都在同一个设备上
        self.hallokitty = self.hallokitty.to(self.params.device)
        image = image.to(self.params.device)

        pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
        self.hallokitty *= pixel_max_dataset
        self.hallokitty = self.hallokitty.reshape(3, 32, 32)

        image = (1-self.mask)*self.hallokitty  + image * self.mask

        # 获取两个张量的最大值并进行比较
        max_pixel = torch.max(self.hallokitty).item() if torch.max(self.hallokitty).item() > torch.max(
            image).item() else torch.max(image).item()

        # 将超过 max_pixel 的值设为 max_pixel
        image[image > max_pixel] = max_pixel

        return image

    def synthesize_inputs(self, batch, test=False, attack_portion=0.1):
        """
        Apply triggers to a portion of the batch.
        """
        attack_count = attack_portion
        for i in range(attack_count):
            label = batch.labels[i].item()
            batch.inputs[i] = self.add_trigger(batch.inputs[i], label, test=test)
        return batch

    def synthesize_labels(self, batch, attack_portion=0.1):
        """
        Modify labels for the backdoor attack.
        """
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)
        return batch

