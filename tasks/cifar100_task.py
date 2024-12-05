import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms
from torchvision import models
import torch
# Assuming ResNet18 for CIFAR-100 is already implemented and imported correctly
from models.resnet_cifar import ResNet18
from tasks.task import Task
from models.googlenet import googlenet

class cifar100Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()

        if self.params.fl_sample_dirichlet:
            # Sample indices for participants using Dirichlet distribution
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders, number_of_samples = zip(
                *[self.get_train(indices) for pos, indices in indices_per_participant.items()])

        else:
            # Sample indices for participants equally split
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)

            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                                                     for pos in range(self.params.fl_total_participants)])

        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=4)

        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False,
                                      num_workers=4)

        self.classes = self.train_dataset.classes  # CIFAR-100 has 100 classes

    def build_model(self) -> nn.Module:
        model = googlenet()
        # model sige= models.resnet34(pretrained=False, num_classes=100)
        # model.classifier = torch.nn.Linear(model.classifier.in_features, 100)
        # Assuming ResNet18 for CIFAR-100
        return model
