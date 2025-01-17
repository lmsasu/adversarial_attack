import pickle
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary

fwd_activations = dict()

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        fwd_activations['relu1'] = []
        fwd_activations['relu2'] = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        fwd_activations['relu1'].append(x.detach().cpu().data.numpy())
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        fwd_activations['relu2'].append(x.detach().cpu().data.numpy())
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_save(train_loader, path='./saved_models/my_net.pch'):
    net = Net().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        torch.save(net.state_dict(), path)
    print('Finished Training')
    summary(net)
    return net


def save_activation(activations, name):
    def hook(model, input, output):
        activations[name].append(output.cpu().detach().data.numpy())
    return hook


def restore(path):
    net = Net()
    net.load_state_dict(torch.load(path))
    # summary(net)
    return net


def forward(net, test_loader):
    for i, data in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        inputs = inputs.to('cuda')
        # forward + backward + optimize
        net(inputs)

    print('Finished testing')
    return net


def save_activations_pickle(activations, path):
    with open(path, 'wb') as f:
        pickle.dump(activations, f)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    train_set = torchvision.datasets.CIFAR10(root='/tmp/data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

    test_set = torchvision.datasets.CIFAR10(root='/tmp/data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = './saved_models/my_net.pch'

    _ = train_save(train_loader, PATH)
    net = restore(PATH)
    net.to('cuda')

    activations = defaultdict(list)
    # register hooks
    # it works
    # net.relu1.register_forward_hook(save_activation('relu1'))
    # net.relu2.register_forward_hook(save_activation('relu2'))

    for name, module in net.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(save_activation(activations, name))

    forward(net, test_loader)

    path1 = './saved_models/activations.pkl'
    path2 = './saved_models/activations_hook.pkl'
    save_activations_pickle(fwd_activations, path1)
    save_activations_pickle(activations, path2)

    restored_fwd_activations = pickle.load(open(path1, 'rb'))
    restored_activations = pickle.load(open(path2, 'rb'))

    assert set(restored_fwd_activations.keys()) == set(restored_activations.keys())
    for key in restored_fwd_activations:
        print(f'Checking key {key}...', end='')
        val1 = restored_fwd_activations[key]
        val2 = restored_activations[key]
        assert len(val1) == len(val2)
        assert all(np.all(np.allclose(item1, item2) for item1, item2 in zip(val1, val2)))
        print('done')

    print('Comparison done')

