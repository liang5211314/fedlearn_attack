import random
import torch
from torchvision.transforms import transforms

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class DBASynthesizer(Synthesizer):
    triggerX: int = 15
    triggerY: int = 15
    trigger_size: int = 5  # Trigger size is 5x5

    def __init__(self, task: Task):
        super().__init__(task)
        self.backdoor_label = self.params.backdoor_label

    def add_trigger(self, image, label, test=False):
        """
        Add DBA trigger to the image.
        """
        pixel_max = 0
        # pixel_max = max(1, torch.max(image))

        # Ensure that the trigger won't go out of bounds
        # if self.triggerX + self.trigger_size > image.shape[2]:
        #     self.triggerX = image.shape[2] - self.trigger_size
        # if self.triggerY + self.trigger_size > image.shape[1]:
        #     self.triggerY = image.shape[1] - self.trigger_size
        label = int(label % 4)
        if not test:
            if label == 0:
                image[:, self.triggerY:self.triggerY + 2, self.triggerX:self.triggerX + 2] = pixel_max
            elif label == 1:
                image[:, self.triggerY:self.triggerY + 2, self.triggerX + 2:self.triggerX + 5] = pixel_max
            elif label == 2:
                image[:, self.triggerY + 2:self.triggerY + 5, self.triggerX:self.triggerX + 2] = pixel_max
            elif label == 3:
                image[:, self.triggerY + 2:self.triggerY + 5, self.triggerX + 2:self.triggerX + 5] = pixel_max
            return image
        else:
            image[:, self.triggerY:self.triggerY + 2, self.triggerX:self.triggerX + 2] = pixel_max
            image[:, self.triggerY:self.triggerY + 2, self.triggerX + 2:self.triggerX + 5] = pixel_max
            image[:, self.triggerY + 2:self.triggerY + 5, self.triggerX:self.triggerX + 2] = pixel_max
            image[:, self.triggerY + 2:self.triggerY + 5, self.triggerX + 2:self.triggerX + 5] = pixel_max
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

