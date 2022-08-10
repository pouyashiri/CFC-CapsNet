from __future__ import print_function, division
import pickle
import tarfile
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py


class AffNISTDataset(Dataset):
    def __init__(self, root_folder, mode):
        self.root_folder = root_folder
        self.x = ''
        self.y = ''
        self.mode = mode
        if mode == False:
            with h5py.File(f'{self.root_folder}/affnist_expanded_test', 'r') as file:
                self.x = file['x_test'][:]
                self.y = file['y_test'][:]
        else:
            with open(f'{self.root_folder}/affnist_expanded_train','rb') as file:
                (self.x, self.y) = pickle.load(file)
                


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return [torch.Tensor(self.x[item]), torch.LongTensor(self.y[item].reshape(1,))]

