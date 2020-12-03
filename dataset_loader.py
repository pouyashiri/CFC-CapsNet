import pickle
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10
import torchnet as tnt

def get_iterator(dset, bsize, mode):

        if dset == 'svhn':
            if mode is True:
                dataset = SVHN(root='./data', download=True, split="train")
            elif mode is False:
                dataset = SVHN(root='./data', download=True, split="test")
            data = torch.tensor(dataset.data)
            labels = torch.tensor(dataset.labels)


        elif dset == 'mnist':
            dataset = MNIST(root='./data', download=True, train=mode)
            data = getattr(dataset, 'train_data' if mode else 'test_data')
            labels = getattr(dataset, 'train_labels' if mode else 'test_labels')

        elif dset == 'fmnist':
            dataset = FashionMNIST(root='./data', download=True, train=mode)

            data = getattr(dataset, 'train_data' if mode else 'test_data')
            labels = getattr(dataset, 'train_labels' if mode else 'test_labels')

        elif dset == 'cifar':
            dataset = CIFAR10(root='./dataF', download=True, train=mode)
            data = np.transpose(getattr(dataset, 'data'), (0, 3, 1, 2))
            labels = getattr(dataset, 'targets')

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=bsize, num_workers=4, shuffle=mode)