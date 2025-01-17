import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def mean_std(loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and std for a dataset, channelwise.
    Source: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/2?u=kuzand

    Args:
        loader: a data loader objecy

    Returns:
        mean and std for the whole dataset, channelwise
    """
    data_0 = loader.dataset.__getitem__(0)[0]
    assert data_0.ndim == 3, f'data shape {data_0.shape} is not 3'
    channels = data_0.shape[0]
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


class DiffDataset(Dataset):
    def __init__(self, original_dataset: Dataset, difference: int = 1, dim=1) -> None:
        assert difference >= 1, f'Should be differentiation of at least degree 1, got {difference=}'
        super().__init__()
        self.original_dataset = original_dataset
        self.difference = difference
        self.dim = dim

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index):
        item, class_index = self.original_dataset.__getitem__(index)
        item = torch.diff(item, n=self.difference, dim=self.dim)
        return item, class_index


if __name__ == '__main__':
    print(os.getcwd())
    batch_size = 2000
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImageFolder(os.path.join('../../../images/tiny_imagenet_200', 'train'), transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=10)
    mean, std = mean_std(trainloader)
    print(f'mean={mean}, std={std}')

