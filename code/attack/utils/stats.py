from typing import Tuple

import numpy as np
from torchvision import datasets


def compute_mean_std(data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    assert data_type in ['cifar10', 'cifar100'], f'Unknown datatype: {data_type}'
    # load CIFAR
    if data_type == 'cifar10':
        train_data = datasets.CIFAR10('./cifar10_data', train=True, download=True)
    else:
        train_data = datasets.CIFAR100('./cifar10_data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    x = x / 255.0
    # calculate the mean and std along the (0, 1) axes
    train_mean = np.mean(x, axis=(0, 1))
    train_std = np.std(x, axis=(0, 1))
    # the mean and std
    return train_mean, train_std


cifar10_mean = np.array([0.49139968, 0.48215841, 0.44653091], dtype=np.float32)
cifar10_std = np.array([0.24703223, 0.24348513, 0.26158784], dtype=np.float32)

cifar100_mean = np.array([0.50707516, 0.48654887, 0.44091784], dtype=np.float32)
cifar100_std = np.array([0.26733429, 0.25643846, 0.27615047], dtype=np.float32)

tiny_imagenet_200_mean = np.array([0.4802, 0.4481, 0.3975])
tiny_imagenet_200_std = np.array([0.2302, 0.2265, 0.2262])

if __name__ == '__main__':
    print(f'mean and std for cifar10: {compute_mean_std("cifar10")}')
    print(f'mean and std for cifar100: {compute_mean_std("cifar100")}')
