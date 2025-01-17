import glob
import os.path
import shutil
from datetime import datetime
from typing import List

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.conversions import get_optimizer_name
from utils.environment_utils import is_linux_os, can_compile_torch_model
from utils.timing import timeit


# # imagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#


def get_dataset(path, mean, std):
    normalize = transforms.Normalize(mean, std)
    transformations = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageFolder(path, transform=transformations)
    return dataset


def get_loader(path, mean, std, batch_size, shuffle: bool):
    dataset = get_dataset(path, mean, std)
    num_workers = 8 if is_linux_os() else 1
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dataloader


@torch.no_grad()
def get_loss(model: nn.Module, device: str, criterion: nn.Module, path: str, mean: List[float], std: List[float],
             batch_size: int) -> float:
    model.eval()
    loader = get_loader(path, mean, std, batch_size=batch_size, shuffle=False)
    total_loss = 0
    n_items = 0
    for data, labels in tqdm(loader, 'Loss compute', initial=1):
        data, labels = data.to(device), labels.to(device)
        predicted = model(data)
        loss = criterion(predicted, labels)
        total_loss += loss.detach().cpu().item()
        n_items += len(data)
    return total_loss / n_items


@timeit
def train_model(model: nn.Module,
                device: str,
                train_path: str,
                test_path: str,
                mean: List[float],
                std: List[float],
                epochs: int,
                batch_size: int,
                optimizer: torch.optim.Optimizer,
                writer: SummaryWriter,
                log_interval: int = 1):
    model.train()

    train_loader = get_loader(train_path, mean, std, batch_size=batch_size, shuffle=True)
    criterion = CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), 'Training'):
        model.train()
        total_loss = 0
        n_items = 0
        for data, labels in tqdm(train_loader, 'Iterations', initial=1):
            model.zero_grad()
            data, labels = data.to(device), labels.to(device)
            predicted = model(data)

            loss = criterion(predicted, labels)
            total_loss += loss.item()
            n_items += len(data)
            loss.backward()
            optimizer.step()

        print(f'Average running loss at epoch {epoch}/{epochs}: {total_loss / n_items}')
        writer.add_scalar('Running loss/Train', total_loss / n_items, epoch)

        accurate_train_loss = get_loss(model, device, criterion, train_path, batch_size)
        print(f'Average train loss at epoch {epoch}/{epochs}: {accurate_train_loss}')
        writer.add_scalar('Loss/Train', accurate_train_loss, epoch)

        if epoch % log_interval == 0:
            train_accuracy = test_model(model, device, train_path, mean, std, batch_size)
            test_accuracy = test_model(model, device, test_path, mean, std, batch_size)

            writer.add_scalar('Train accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Test accuracy/Train', test_accuracy, epoch)

            print(f'Train accuracy/Train at epoch {epoch}: {train_accuracy}')
            print(f'Test accuracy/Train at epoch {epoch}: {test_accuracy}')


@timeit
@torch.no_grad()
def test_model(model: nn.Module,
               device: str,
               test_path: str,
               mean: List[float],
               std: List[float],
               batch_size: int) -> float:
    model.eval()
    loader = get_loader(test_path, mean, std, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Test', initial=1):
        images, labels = images.to(device), labels.to(device)
        y_hat = model(images).argmax(dim=-1)
        correct += torch.sum(labels == y_hat).item()
        total += len(images)
    test_accuracy = correct / total
    print(f'Test accuracy: {test_accuracy}')
    return test_accuracy


def get_model(device: str, model_type: str) -> nn.Module:
    allowed_models = ['resnet18', 'resnet34', 'resnet50']
    model_type = model_type.lower() if model_type else None
    assert model_type in allowed_models, f'Model {model_type} not in {allowed_models}'
    _model = None
    if model_type == 'resnet18':
        _model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet34':
        _model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet50':
        _model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    _model.fc = torch.nn.Linear(in_features=_model.fc.in_features, out_features=2)
    _model.to(device)
    return _model


def make_train_test_dirs(attacked_dirs: List[str], genuine_dirs: List[str], put_into: str):
    shutil.rmtree(put_into)
    os.mkdir(put_into)

    # copy attacked files
    for dir_w_attacked_files in attacked_dirs:
        parent = os.path.split(dir_w_attacked_files)[1]
        for name in glob.glob(f'{dir_w_attacked_files}/**/*.png', recursive=True):
            new_name = f'{parent}_{os.path.split(name)[1]}'
            new_path = os.path.join(put_into, 'attacked', new_name)
            shutil.copy(name, new_path)

    # coppy genuine files
    for dir_w_genuine_files in genuine_dirs:
        parent = os.path.split(dir_w_genuine_files)[1]
        for name in glob.glob(f'{dir_w_genuine_files}/**/*.png', recursive=True):
            new_name = f'{parent}_{os.path.split(name)[1]}'
            new_path = os.path.join(put_into, 'genuine', new_name)
            shutil.copy(name, new_path)


@timeit
def main():
    use_cuda: bool = True
    device: str = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    train_path = './images/detect_attack_split/train'
    test_path = './images/detect_attack_split/test'
    epochs = 1500
    batch_size = 128  # 3800
    lr = 1e-2  # larger lr? lr scheduler with decreasing of lr?
    model_type = "resnet50l"
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f'_shuffled_{model_type}_2classes_{now_str}_epochs={epochs}_batch={batch_size}_lr={lr}'
    dir_saved_models = './saved_models'
    if not os.path.exists(dir_saved_models):
        os.mkdir(dir_saved_models)
    model = get_model(device, model_type)
    if can_compile_torch_model(use_cuda=True, compile=False):
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    experiment_name = f'{experiment_name}_optimizer={get_optimizer_name(optimizer)}'
    writer = SummaryWriter(comment=experiment_name)
    train_model(model, device, train_path, test_path, mean=mean, std=std, epochs=epochs, batch_size=batch_size,
                optimizer=optimizer,
                writer=writer, log_interval=10)
    path_model = f'{dir_saved_models}/{experiment_name}.pth'
    torch.save(model.state_dict(), path_model)
    # model = get_model(device, model_type)
    # model.load_state_dict(torch.load(path_model))
    test_acc = test_model(model, device, test_path, batch_size)
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


if __name__ == '__main__':
    # main()
    make_train_test_dirs(
        [os.path.join(os.getcwd(), 'images/attacked/cifar10_train/ResNet18_attacker=FGSM_epsilon=0.001')],
        [os.path.join(os.getcwd(), 'images/attacked/cifar10_train/ResNet18_attacker=FGSM_epsilon=0.001')],
        os.path.join(os.getcwd(), 'images/detect_attack/cifar10/'))
