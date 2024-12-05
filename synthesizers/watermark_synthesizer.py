import random
import torch
from torchvision.transforms import transforms
import numpy as np
from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()
import cv2

class WaterMarkSynthesizer(Synthesizer):

    def __init__(self, task: Task):
        super().__init__(task)
        self.backdoor_label = self.params.backdoor_label

    def add_trigger(self, image, label, test=False):
        self.watermark = cv2.imread('./synthesizers/mask/watermark.png', cv2.IMREAD_GRAYSCALE)
        self.watermark = cv2.bitwise_not(self.watermark)
        self.watermark = cv2.resize(self.watermark, dsize=(image.shape[2], image.shape[1]),
                                    interpolation=cv2.INTER_CUBIC)  # Make sure dimensions match
        pixel_max = np.max(self.watermark)
        self.watermark = self.watermark.astype(np.float64) / pixel_max
        # cifar [0,1] else max>1
        pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1

        # Adjust watermark pixel values according to image max value
        self.watermark *= pixel_max_dataset

        # Ensure image tensor is moved to CPU before NumPy operations
        max_pixel = max(np.max(self.watermark), torch.max(image.cpu()).item())

        # Add watermark to image, ensuring both are on the same device (CPU or CUDA)
        image += torch.tensor(self.watermark, device=image.device)

        # Clamping the pixel values
        image = torch.clamp(image, max=max_pixel)

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

