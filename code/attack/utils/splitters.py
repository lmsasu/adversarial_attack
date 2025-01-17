import os
import shutil

import splitfolders
from torchvision.datasets import ImageFolder


def split_train_test(input_path:str, output_path:str) -> None:
    assert os.path.exists(input_path), f'{input_path} does not exist or is not a directory'
    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)
    assert os.path.exists(output_path), f'{input_path} does not exist or is not a directory'
    # TODO mage stratified sampling
    # dataset = ImageFolder(input_path)
    # for file, label in dataset:
    #     print(type(file), type(label))
    #     break
    splitfolders.ratio(input_path, seed=1337, output=output_path, ratio=(0.6, 0.0, 0.4))


if __name__ == '__main__':
    input_path = '../images/detect_attack/'
    output_path = '../images/detect_attack_split'
    split_train_test(input_path, output_path)
