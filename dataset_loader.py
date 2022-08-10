import pickle
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10
import torchnet as tnt
import scipy.io
import torchvision
from Affnist_dataset import AffNISTDataset
from torch.utils.data import Dataset, DataLoader


def load_affnist_trans_test():
    transformed_path = 'affnist/transformed'
    x_test = []
    y_test = []
    for i in range(32):
        test = scipy.io.loadmat(f'{transformed_path}/test_batches/{i + 1}.mat')
        x_test_temp = np.transpose(test['affNISTdata'][0][0][2], (1, 0)).reshape(10000, 40, 40)
        y_test_temp = test['affNISTdata'][0][0][5].reshape(10000)
        x_test.append(x_test_temp)
        y_test.append(y_test_temp)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return (x_test, y_test)


def get_iterator(dset, bsize, mode):
    if dset == 'aff_expanded':
        tensor_dataset = AffNISTDataset('affnist/expanded', mode)
        return DataLoader(tensor_dataset, batch_size=bsize, num_workers=4, shuffle=False)

    elif dset == 'aff_trans_test':
        (data, labels) = load_affnist_trans_test()


    elif dset == 'svhn':
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
        # print(f'SHAPE={np.shape(labels)}')

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
