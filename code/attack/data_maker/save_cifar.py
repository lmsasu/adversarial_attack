"""
Author: Lucian Sasu
Date: 2023-12-24

Saves cifar10 and cifar100 original files intro directories
"""


import os.path
import shutil
from typing import Optional

import torchvision
from tqdm import tqdm



def write_dataset(datasetname: Optional[str], path: str) -> None:
    datasetname = datasetname.lower() if datasetname else None

    datasets = {'cifar10_train': torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True),
                'cifar10_test': torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True),
                'cifar100_train': torchvision.datasets.CIFAR100(root='/tmp/data', train=True, download=True),
                'cifar100_test': torchvision.datasets.CIFAR100(root='/tmp/data', train=False, download=True)}

    assert datasetname in datasets, f'The required dataset {datasetname} is not present in {datasets.keys()}'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    file_id = 0
    for file, label in tqdm(datasets[datasetname], desc=datasetname):
        save_into = os.path.join(path, str(label))
        os.makedirs(save_into, exist_ok=True)
        file.save(os.path.join(save_into, f'{datasetname}_class={label}_{file_id}.png'))
        file_id += 1


if __name__ == '__main__':
    write_dataset('cifar10_train', '../images/cifar/cifar10_train')
    write_dataset('cifar10_test', '../images/cifar/cifar10_test')
    write_dataset('cifar100_train', '../images/cifar/cifar100_train')
    write_dataset('cifar100_test', '../images/cifar/cifar100_test')
