import os.path
import shutil
from collections import defaultdict

import cv2
import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from torchvision import transforms
from tqdm import tqdm

from utils.timing import timeit
from utils.patching import apply_patch_tensor, get_random_location

# TODO add logging


def compute_loss(model, activations, batch):
    model(batch)
    loss = 1.0
    for lst_values in activations.values():
        assert len(lst_values) == 1, f'got list of length {len(lst_values)}'
        loss = loss * torch.norm(lst_values[0], p=2) ** 2
    loss = -torch.log(loss)
    return loss, batch


def save_activation(activations, name):
    def hook(model, input, output):
        activations[name].append(output)
    return hook


def get_init_patch(device):
    patch_shape_cv = (35, 35, 3)

    # patch's contents is randomly defined
    patch = np.random.randint(low=0, high=255, size=patch_shape_cv, dtype=np.uint8)
    transformations = transforms.Compose([transforms.ToTensor()])
    patch = transformations(patch).to(device)
    patch.requires_grad = True
    return patch



@timeit
def attack_net(model, iterations, lr, patch, device) -> torch.Tensor:
    model.train()
    activations = defaultdict(list)

    model.layer1.register_forward_hook(save_activation(activations, 'layer1'))
    model.layer2.register_forward_hook(save_activation(activations, 'layer2'))
    model.layer3.register_forward_hook(save_activation(activations, 'layer3'))
    model.layer4.register_forward_hook(save_activation(activations, 'layer4'))
    model.fc.register_forward_hook(save_activation(activations, 'fc'))

    image_shape_cv = (224, 224, 3)
    patch_shape = patch.shape

    original_patch = patch.clone().detach()
    for _ in tqdm(range(iterations)):

        # the initial batch is black
        iz = torch.permute(torch.zeros(image_shape_cv), (2, 0, 1)).to(device)

        location = get_random_location(image_shape_cv, patch_shape)
        rotation = np.random.randint(1)

        iz = apply_patch_tensor(iz, patch, location, rotation).unsqueeze(0)
        assert iz.requires_grad

        loss, image = compute_loss(model=model,
                                   activations=activations,
                                   batch=iz)
        loss.backward()
        patch.data = patch.data - lr * patch.grad.data

        # reset grads and activations
        model.zero_grad()
        patch.grad.zero_()
        activations.clear()

    print(f'After {iterations} iters, the difference is {torch.norm(original_patch - patch, torch.inf)}')

    return patch


def get_model(device):
    # Using pretrained weights:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    # model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
    # model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device)
    return model


@timeit
def test_net(path, model, device) -> float:
    # Using pretrained weights:
    model.eval()

    dataset = get_dataset(path)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=16)
    correctly_classified = 0

    mapping = {0: 0, 1: 1, 8: 2}

    for data, label in tqdm(dataloader):
        data, label = data.to(device), label.to(device)
        y_resnet = torch.argmax(model(data), dim=1)
        y_mapping = torch.tensor([mapping.get(y_hat.item(), -1) for y_hat in y_resnet]).to(device)
        correctly_classified += torch.sum(label == y_mapping)
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return correctly_classified / len(dataset)


def get_dataset(path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = ImageFolder(path, transform=transformations)
    return dataset


@timeit
def alter(original_path, new_path, patch):
    assert os.path.exists(original_path) and os.path.isdir(original_path)

    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)

    patch_pil = transforms.ToPILImage()(patch)

    for subdir in os.listdir(original_path):
        altered_subdir = os.path.join(new_path, subdir)
        os.mkdir(altered_subdir)
        original_subdir = os.path.join(original_path, subdir)
        for file in os.listdir(original_subdir):
            image = cv2.imread(os.path.join(original_subdir, file))
            middle_1, middle_2 = image.shape[0] // 2, image.shape[1] // 2  # rename them accordingly
            pos_1, pos_2 = middle_1 - patch.shape[1] // 2, middle_2 - patch.shape[2] // 2
            image[pos_1:pos_1 + patch.shape[1], pos_2:pos_2 + patch.shape[2], :] = patch_pil
            new_filename = os.path.join(altered_subdir, file)
            cv2.imwrite(new_filename, image)


def alter_patch(original_path, new_path, patch):
    assert os.path.exists(original_path) and os.path.isdir(original_path)

    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)

    patch_pil = transforms.ToPILImage()(patch)

    for subdir in os.listdir(original_path):
        altered_subdir = os.path.join(new_path, subdir)
        os.mkdir(altered_subdir)
        original_subdir = os.path.join(original_path, subdir)
        for file in os.listdir(original_subdir):
            image = cv2.imread(os.path.join(original_subdir, file))
            middle_1, middle_2 = image.shape[0] // 2, image.shape[1] // 2  # rename them accordingly
            pos_1, pos_2 = middle_1 - patch.shape[1] // 2, middle_2 - patch.shape[2] // 2
            image[pos_1:pos_1 + patch.shape[1], pos_2:pos_2 + patch.shape[2], :] = patch_pil
            new_filename = os.path.join(altered_subdir, file)
            cv2.imwrite(new_filename, image)

@timeit
def main():
    _device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {_device}')

    model = get_model(_device)

    path_test_original = './images/imagenet_3_classes/'
    path_patched_with_initial_patch = './images/imagenet_3_classes_initial_patch'
    path_patched_with_elaborated_patch = './images/imagenet_3_classes_elaborated_patch'
    acc_original = test_net(path_test_original, model, _device)
    print(f'Accuracy before attack: {acc_original}')

    _patch = get_init_patch(_device)
    alter(original_path=path_test_original, new_path=path_patched_with_initial_patch, patch=_patch)
    acc_initial_patch = test_net(path_patched_with_initial_patch, model, _device)
    print(f'Accuracy before attack, with initial patch: {acc_initial_patch}')

    _patch = attack_net(model, 100000, -1e+0, _patch, _device)
    alter(original_path=path_test_original, new_path=path_patched_with_elaborated_patch, patch=_patch)
    acc_after_attack = test_net(path_patched_with_elaborated_patch, model, _device)
    print(f'Accuracy after attack, with elaborated patch: {acc_after_attack}')
    print(f'Difference: {acc_after_attack - acc_initial_patch}')


if __name__ == "__main__":
    main()
