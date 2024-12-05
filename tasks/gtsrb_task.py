import os
import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tasks.task import Task
import os
import random
import torch.utils.data as data
from torchvision import models
import torch
import csv
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tasks.task import Task
from models.resnet_cifar import ResNet18

class gtsrbTask(Task):
    normalize = transforms.Normalize((0.5,), (0.5,))

    def load_data(self):
        self.load_gtsrb_data()
        number_of_samples = []

        if self.params.fl_sample_dirichlet:
            split = min(self.params.fl_total_participants / 43, 1)  # GTSRB has 43 classes
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                            indices_per_participant.items()])

        else:
            split = min(self.params.fl_total_participants / 43, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)

            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                            for pos in range(self.params.fl_total_participants)])

        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples

    def load_gtsrb_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                self.normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalize,
        ])

        # Load GTSRB dataset using the custom GTSRB class
        self.train_dataset = GTSRB(self.params, train=True, transforms=transform_train, min_width=0)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=0)

        self.test_dataset = GTSRB(self.params, train=False, transforms=transform_test, min_width=0)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = list(range(43))  # Update based on the GTSRB classes
        return True

    # def build_model(self) -> nn.Module:
    #     model = ResNet18(num_classes=len(self.classes))  # Ensure the model supports GTSRB classes
    #     return model
    def build_model(self) -> nn.Module:
        # model = googlenet()
        model=models.resnet101(pretrained=False,num_classes=43)
        model.linear = torch.nn.Linear(model.fc.in_features, 43)
        return model

    # def build_model(self) -> nn.Module:
    #      model=models.resnet101(pretrained=True)
    #      model.classifier = torch.nn.Linear(model.classifier.in_features, 43)
    #      # model.fc = nn.Linear(model.classifier.in_features, 43)
    #      return model
         # Ensure the model supports GTSRB classes
        # return model
class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms, data_root=None, min_width=0):
        super(GTSRB, self).__init__()
        if data_root is None:
            data_root = opt.data_root
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: #only load images if more than certain width
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label