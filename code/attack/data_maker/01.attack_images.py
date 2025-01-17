import os
import shutil
from dataclasses import dataclass
from typing import List

from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

import eagerpy as ep
import torch
import torchvision
from foolbox import PyTorchModel, Attack
from foolbox.attacks import (L2DeepFoolAttack, L2ProjectedGradientDescentAttack, \
    L1ProjectedGradientDescentAttack, LinfFastGradientAttack, LinfProjectedGradientDescentAttack,
                             L2AdditiveUniformNoiseAttack)

from utils.datasets import mean_std
from utils.environment_utils import get_num_workers
from utils.timing import timeit

from utils.stats import cifar10_mean, cifar10_std, cifar100_mean, cifar100_std


@dataclass
class DataSet:
    train_name: str
    test_name: str
    mean: List[float]
    std: List[float]
    train_loader: DataLoader
    test_loader: DataLoader


def get_cifar10():
    batch_size = 2000
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name_train = 'cifar10_train'
    trainset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    assert dataset_name_train.split('_', maxsplit=1)[0].lower() in trainset.__class__.__name__.lower()

    dataset_name_test = 'cifar10_test'
    testset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    assert dataset_name_test.split('_', maxsplit=1)[0].lower() in testset.__class__.__name__.lower()

    return DataSet(train_name=dataset_name_train,
                   test_name=dataset_name_test,
                   # values taken from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                   mean=cifar10_mean,
                   std=cifar10_std,
                   train_loader=trainloader,
                   test_loader=testloader)


def get_cifar100():
    batch_size = 2048
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name_train = 'cifar100_train'
    trainset = torchvision.datasets.CIFAR100(root='/tmp/data', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    assert dataset_name_train.split('_', maxsplit=1)[0].lower() in trainset.__class__.__name__.lower()

    dataset_name_test = 'cifar100_test'
    testset = torchvision.datasets.CIFAR100(root='/tmp/data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    assert dataset_name_test.split('_', maxsplit=1)[0].lower() in testset.__class__.__name__.lower()

    return DataSet(train_name=dataset_name_train,
                   test_name=dataset_name_test,
                   # values taken from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                   mean=cifar100_mean,
                   std=cifar100_std,
                   train_loader=trainloader,
                   test_loader=testloader)

def get_tinyimagenet200(data_path: str, device: torch.device):
    batch_size = 500
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name_train = 'tiny_imagenet_200_train'
    trainset = ImageFolder(os.path.join(data_path, "train"), transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=get_num_workers(trainset, batch_size))
    mean_train, std_train = mean_std(trainloader)
    mean_train, std_train = mean_train.to(device), std_train.to(device)

    dataset_name_test = 'tiny_imagenet_200_test'
    testset = ImageFolder(os.path.join(data_path, "test"), transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=get_num_workers(testset, batch_size))

    return DataSet(train_name=dataset_name_train,
                   test_name=dataset_name_test,
                   mean=mean_train,
                   std=std_train,
                   train_loader=trainloader,
                   test_loader=testloader)


class ImageAttacker:
    def __init__(self, model: PyTorchModel,
                 model_name: str,
                 attacker: Attack,
                 attacker_name: str,
                 dataloader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 output_root_dir: str,
                 epsilon: float,
                 device: torch.device):
        self.model = model
        self.model_name = model_name
        self.attacker = attacker
        self.attacker_name = attacker_name
        self.epsilon = epsilon
        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.output_root_dir = output_root_dir
        self.device = device

    def _attack_details_(self) -> str:
        return f'{self.model_name}_attacker={self.attacker_name}_epsilon={self.epsilon}'

    def output_directory(self) -> str:
        return os.path.join(self.output_root_dir, self._attack_details_())

    def attack_and_write(self, extension='png'):
        directory = self.output_directory()
        print(f'Recreating directory {directory}')
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
        index = 0
        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            images, labels = ep.astensors(*[images, labels])

            raw_advs, clipped_advs, success = self.attacker(self.model, images, labels, epsilons=[self.epsilon])
            for clipped_adv, label in zip(clipped_advs[0], labels):
                pil_image = transforms.ToPILImage()(torch.tensor(clipped_adv.numpy()))
                class_directory = os.path.join(directory, str(label.item()))
                os.makedirs(class_directory, exist_ok=True)
                full_file_path = os.path.join(class_directory, f'{self._attack_details_()}_{index}.{extension}')
                pil_image.save(full_file_path)
                index += 1


def get_attacker_name(attacker) -> str:
    attacker_name: str = attacker.__class__.__name__
    if attacker_name.endswith('Attack'):
        attacker_name = attacker_name[:-len('Attack')]
    return attacker_name

@timeit
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attacks = [
        LinfProjectedGradientDescentAttack(),
               LinfFastGradientAttack(),  # FGSM
               L1ProjectedGradientDescentAttack(),  # L1PGD
               L2ProjectedGradientDescentAttack(),  # L2PGD
        #        L2DeepFoolAttack(loss='crossentropy')
        # new
        L2AdditiveUniformNoiseAttack(),
    ]

    for attack in tqdm(attacks, desc='Attacks'):
        for data in (
                get_cifar10(),
                get_cifar100(),
                get_tinyimagenet200('../../../images/tiny_imagenet_200', device),
        ):
            epsilons = [
                #first stage
                # 0.0002,
                # 0.0005,
                # 0.0008,
                # 0.001,
                # 0.0015,

                # second stage
                # 0.002,
                # 0.003,
                0.01,
                0.03,
                # 0.3,

                # third stage
                # 0.5,
                # 1.0,
            ]

            for loader, dataset_name in zip((data.train_loader, data.test_loader), (data.train_name, data.test_name)):
                for epsilon in epsilons:
                    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
                    preprocessing = {'mean': data.mean, 'std': data.std, 'axis': -3}
                    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
                    attack_name = get_attacker_name(attack)
                    attacker = ImageAttacker(fmodel, 'ResNet18', attack, attack_name, loader, dataset_name,
                                             f'../../../images/attacked/{dataset_name}', epsilon=epsilon,
                                             device=device)
                    attacker.attack_and_write()


if __name__ == '__main__':
    main()

    # FGM = L2FastGradientAttack
    # FGSM = LinfFastGradientAttack
    # L1PGD = L1ProjectedGradientDescentAttack
    # L2PGD = L2ProjectedGradientDescentAttack
    # LinfPGD = LinfProjectedGradientDescentAttack
    # PGD = LinfPGD
    # MIFGSM = LinfMomentumIterativeFastGradientMethod
    #
    # L1AdamPGD = L1AdamProjectedGradientDescentAttack
    # L2AdamPGD = L2AdamProjectedGradientDescentAttack
    # LinfAdamPGD = LinfAdamProjectedGradientDescentAttack
    # AdamPGD = LinfAdamPGD
