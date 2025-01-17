import os.path
from datetime import datetime
from typing import Optional

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, \
    efficientnet_v2_l, EfficientNet_V2_L_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

import sys
import os

from utils.datasets import DiffDataset

sys.path.append(os.getcwd())

from utils.timing import timeit
from utils.conversions import get_optimizer_name, get_model_name, get_scheduler_name
from utils.environment_utils import is_linux_os, get_machine_name
from utils.stats import cifar10_mean, cifar10_std, cifar100_mean, cifar100_std, tiny_imagenet_200_mean, \
    tiny_imagenet_200_std

import logging


def get_dataset(path: str, dataset_type: str, difference: int) -> ImageFolder:
    if dataset_type == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
    elif dataset_type == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
    elif dataset_type == 'tiny_imagenet_200':
        mean = tiny_imagenet_200_mean
        std = tiny_imagenet_200_std
    else:
        raise ValueError(f'unknown dataset type {dataset_type}')

    normalize = transforms.Normalize(mean=mean, std=std)
    transformations = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageFolder(path, transform=transformations)

    if difference > 0:
        dataset = DiffDataset(dataset, difference=difference)

    return dataset


def get_loader(path: str, batch_size: int, shuffle: bool, dataset_type: str, difference: int) -> DataLoader:
    dataset = get_dataset(path, dataset_type, difference=difference)
    num_workers = 4 if is_linux_os() else 1
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dataloader


@torch.no_grad()
def get_loss(model: nn.Module,
             device: str,
             criterion: nn.Module,
             path: str,
             batch_size: int,
             dataset_type: str,
             difference: int) -> float:
    model.eval()
    loader = get_loader(path, batch_size=batch_size, shuffle=False, dataset_type=dataset_type, difference=difference)
    total_loss = 0
    n_items = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        predicted = model(data)
        if isinstance(criterion, BCELoss):
            labels = labels.unsqueeze(1).to(torch.float)
        loss = criterion(predicted, labels)
        total_loss += loss.detach().cpu().item()
        n_items += len(data)
    return total_loss / n_items


@timeit
def train_model(model: nn.Module,
                device: str,
                train_path: str,
                test_path: str,
                dataset_type: str,
                epochs: int,
                batch_size: int,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[LRScheduler],
                writer: SummaryWriter,
                log_interval: int = 1,
                difference: int = 0):
    model.train()

    train_loader = get_loader(train_path, batch_size=batch_size, shuffle=True, dataset_type=dataset_type,
                              difference=difference)
    criterion = CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), 'Training', initial=1):
        model.train()
        total_loss = 0
        n_items = 0
        for data, labels in train_loader:
            model.zero_grad()
            data, labels = data.to(device), labels.to(device)
            predicted = model(data)
            if isinstance(criterion, BCELoss):
                labels = labels.unsqueeze(1).to(torch.float)
            loss = criterion(predicted, labels)
            total_loss += loss.item()
            n_items += len(data)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            logging.info(f'scheduled lr={scheduler.get_last_lr()}')
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        logging.info(f'Average running loss for {difference=} at epoch {epoch}/{epochs}: {total_loss / n_items}')
        writer.add_scalar('Running loss/Train', total_loss / n_items, epoch)

        accurate_train_loss = get_loss(model, device, criterion, train_path, batch_size, dataset_type,
                                       difference=difference)
        logging.info(f'Average train loss for {difference=} at epoch {epoch}/{epochs}: {accurate_train_loss}')
        writer.add_scalar('Loss/Train', accurate_train_loss, epoch)

        if epoch % log_interval == 0:
            train_accuracy = test_model(model, device, train_path, batch_size, dataset_type, difference=difference)
            writer.add_scalar('Train accuracy/Train', train_accuracy, epoch)
            logging.info(f'Train accuracy/Train for {difference=} {train_path=} at epoch {epoch}: {train_accuracy}')

            test_accuracy = test_model(model, device, test_path, batch_size, dataset_type, difference=difference)
            writer.add_scalar('Test accuracy/Train', test_accuracy, epoch)
            logging.info(f'Test accuracy/Train for {difference=} {test_path=} at epoch {epoch}: {test_accuracy}')


@torch.no_grad()
def test_model(model: nn.Module, device: str, test_path: str, batch_size: int, dataset_type: str,
               difference: int = 0) -> float:
    model.eval()
    loader = get_loader(test_path, batch_size=batch_size, shuffle=False, dataset_type=dataset_type,
                        difference=difference)
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        prob = model(images)
        y_hat = prob.argmax(dim=-1)
        correct += torch.sum(labels == y_hat).item()
        total += len(images)
    test_accuracy = correct / total
    return test_accuracy


@torch.no_grad()
def get_model(device: str, model_type: str, dilation: Optional[int] = None) -> nn.Module:
    allowed_models = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_v2_l', 'efficientnet_v2_m']
    model_type = model_type.lower() if model_type else None
    assert model_type in allowed_models, f'Model {model_type} not in {allowed_models}'
    _model = None
    if model_type == 'resnet18':
        _model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet34':
        _model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet50':
        _model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_type == 'efficientnet_v2_l':
        _model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    elif model_type == 'efficientnet_v2_m':
        _model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    assert _model is not None

    dilation = 1 if dilation is None or dilation < 1 else dilation
    # Modify conv1 to suit CIFAR-10/CIFAR-100
    _model.conv1 = nn.Conv2d(3, 64,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False,
                             dilation=dilation)

    if model_type.startswith('resnet'):
        _model.fc = torch.nn.Linear(in_features=_model.fc.in_features, out_features=2)
    elif model_type.startswith('efficientnet'):
        _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, 2)
    else:
        raise ValueError(f'Unknown model type {model_type}')

    return _model.to(device)


@timeit
def main():
    machine_name = get_machine_name()
    logging.info(f'{machine_name=}')

    use_cuda: bool = True
    device: str = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    logging.info(f'{device=}')

    difference = 2  # 0 = original dataset; 1 = first order difference; 2 = second order difference
    differences = [0, 1, 2]
    for difference in tqdm(differences, total=len(differences)):
        logging.info(f'{difference=}')
        root_images_attack = '../../images/attacked_split'
        # root_images_attack = '../images/detect_attack_mixeddatasets'  # evaluate on two merged datasets
        # root_images_attack = '../../images/detect_attack_crossed'

        # experiment_dir = 'cifar10_ResNet18_attacker=LinfFastGradient_epsilon=0.01'   # FGSM, works fine genuine and 1st, 2nd order difference
        experiment_dirs = sorted([d for d in os.listdir(root_images_attack) if os.path.isdir(os.path.join(root_images_attack, d))])
        for experiment_dir in tqdm(experiment_dirs, initial=1):
            logging.info(f'{experiment_dir=}')

            # experiment_dir ='tiny_imagenet_200_ResNet18_attacker=LinfFastGradient_epsilon=0.01'
            # experiment_dir = 'cifar100_ResNet18_attacker=LinfFastGradient_epsilon=0.03'

            # experiment_dir = 'cifar10_ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01'  # hard, 0.5 test; hard for 1st order difference;; hard for 2nd order difference
            # experiment_dir = 'tiny_imagenet_200_ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01'
            # experiment_dir = 'cifar100_ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.03'

            # experiment_dir = 'cifar10_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01'  # works only (???) with 2nd order difference; 2024-03-22 13:44:08,999 [INFO] Test accuracy/Train at epoch 40: 0.95655
            # experiment_dir = 'tiny_imagenet_200_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01'
            # experiment_dir = 'cifar10_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.03'

            # experiment_dir = 'cifar10_ResNet18_attacker=L2DeepFool_epsilon=0.01'  # hard, 0.5 test for 2nd order; check what's inside
            # experiment_dir = 'tiny_imagenet_200_ResNet18_attacker=L2DeepFool_epsilon=0.01'

            # experiment_dir = 'cifar10_ResNet18_attacker=LinfProjectedGradientDescent_epsilon=0.01'  # works with 1st order difference, and with 2nd order difference
            # experiment_dir = 'tiny_imagenet_200_ResNet18_attacker=LinfProjectedGradientDescent_epsilon=0.01'

            # experiment_dir = 'cifar100_L2ProjectedGradientDescent_epsilon=0.01_L1ProjectedGradientDescent_epsilon=0.01'  # evaluate on two merged datasets
            # experiment_dir = 'tiny_imagenet_200_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01'  # evaluate on two merged datasets

            # experiment_dir = 'cifar100_train_L2PGD_eps=0.01_test_L1PGD_eps=0.01'

            # experiment_dir = 'cifar100_train_L1PGD_eps=0.01_test_L2PGD_eps=0.01'

            assert experiment_dir in os.listdir(root_images_attack), (f'{experiment_dir} is not a subdirectory of '
                                                                      f'{root_images_attack}')
            logging.info(f'{experiment_dir=}')

            train_path = f'{root_images_attack}/{experiment_dir}/train'
            test_path = f'{root_images_attack}/{experiment_dir}/test'

            logging.info(f'{train_path=}\n{test_path=}')

            dataset_type = 'cifar10'
            # dataset_type = 'tiny_imagenet_200'
            dataset_type = experiment_dir.split('_')[0] if experiment_dir.startswith('cifar') else 'tiny_imagenet_200'
            logging.info(f'{dataset_type=}')
            assert dataset_type + '_' in train_path and dataset_type + '_' in test_path

            epochs = 200

            batch_sizes = {
                'racheta1': 1500,
                'racheta2': 1500,
                'racheta4': 6000 if 'cifar' in dataset_type else 550,
                'racheta5': 6000 if 'cifar' in dataset_type else 550,
                'racheta10': 500,
            }
            batch_size = batch_sizes[machine_name]

            lr = 1e-2 if 'cifar' in dataset_type else 1e-3
            # model_type = 'resnet50'
            model_type = 'resnet18'

            logging.info(f'{epochs=} {batch_size= } {lr= } {model_type= } {difference=} ')

            now = datetime.now()
            now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
            dir_saved_models = './saved_models'
            os.makedirs(dir_saved_models, exist_ok=True)

            dilation = None

            model = get_model(device=device, model_type=model_type, dilation=dilation)
            logging.info(f'Model name: {get_model_name(model)}')
            logging.info(model)
            logging.info(summary(model))

            experiment_name = (f'_{experiment_dir}_{model_type}_'
                               'CrossEntropy_'
                               f'{now_str}_dilation={1 if dilation is None else dilation}_epochs={epochs}_batch={batch_size}_lr={lr}_diff={difference}')

            # optimizer = optim.Adam(model.parameters(), lr=lr)
            # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
            logging.info(f'Optimizer: {get_optimizer_name(optimizer)}')

            step_size = 10
            gamma = 0.9
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            logging.info(f'Scheduler: {get_scheduler_name(scheduler)}, {step_size=}, {gamma=}')
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4, verbose=True)

            experiment_name = (f'{experiment_name}_optimizer={get_optimizer_name(optimizer)}_'
                               f'scheduler={get_scheduler_name(scheduler)}')
            writer = SummaryWriter(comment=experiment_name)
            train_model(model, device, train_path, test_path, dataset_type=dataset_type, epochs=epochs, batch_size=batch_size,
                        optimizer=optimizer, scheduler=scheduler, writer=writer, log_interval=5, difference=difference)
            path_model = f'{dir_saved_models}/{experiment_name}.pth'
            torch.save(model.state_dict(), path_model)

            test_acc = test_model(model, device, test_path, batch_size, dataset_type=dataset_type, difference=difference)
            writer.add_scalar("Test", test_acc)
            writer.add_hparams(
                {
                    "lr": lr, "batch_size": batch_size, "epochs": epochs, "model": model_type, "device": device,
                    "optimizer": get_optimizer_name(optimizer),
                },
                {
                    "test_accuracy": test_acc
                }
            )
            writer.close()

        # check if cifar100_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01 does not work with difference=1 - not working
        # check if cifar100_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01 does work with difference=2


def setup_logger():
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f'detect_attack_{now_str}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{filename}"),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    setup_logger()
    main()
