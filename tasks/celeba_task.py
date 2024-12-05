import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms
from torchvision import models
# from models.resnet import ResNet18  # Update to your actual model path
from tasks.task import Task
import torch.utils.data as data

class CelebaTask(Task):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def load_data(self):
        self.load_celeba_data()
        number_of_samples = []

        if self.params.fl_sample_dirichlet:
            split = min(self.params.fl_total_participants / 10, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                                                     indices_per_participant.items()])

        else:
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)

            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                                                     for pos in range(self.params.fl_total_participants)])

        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples

    def load_celeba_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.Resize((64, 64)),  # Resize images to 64x64
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                self.normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = CelebA_attr('train', transform_train)

        # self.train_dataset = torchvision.datasets.CelebA(
        #     root=self.params.data_path,
        #     split='train',
        #     download=True,
        #     transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=0)
        self.test_dataset = CelebA_attr('test', transform_train)

        # self.test_dataset = torchvision.datasets.CelebA(
        #     root=self.params.data_path,
        #     split='valid',
        #     download=True,
        #     transform=transform_test)

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = self.classes = list(range(8))#Adjust based on your labels
        return True

    def build_model(self) -> nn.Module:
        model = models.resnet101(pretrained=False) # Update num_classes if necessary
        return model
class CelebA_attr(data.Dataset):
    def __init__(self, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root='./attacks/condition_trigger_generate/data/', split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)
