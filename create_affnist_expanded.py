"""
Author: Pouya Shiri, pouyashiri@gmail.com
Create the shuffled expanded AffNIST dataset (out of AffNIST centered dataset).
Placing 28x28 images randomly on a 40x40 grid.
Sources required: 'training_and_validation.mat' and 'test.mat'
https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/

usage: python create_affnist_expanded.py outputFolder
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import tarfile
import os
import sys
import os
import h5py
import requests
import zipfile


expanded_path = 'affnist/expanded'
transformed_path = 'affnist/transformed'

if not os.path.exists(expanded_path):
    os.makedirs(expanded_path)

if not os.path.exists(transformed_path):
    os.makedirs(transformed_path)


print('Downloading AffNIST Sources...')
url = 'https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training_and_validation.mat.zip'
r = requests.get(url, allow_redirects=True)

open(f'{expanded_path}/training_and_validation.zip', 'wb').write(r.content)


with zipfile.ZipFile(f'{expanded_path}/training_and_validation.zip', 'r') as zip_ref:
    zip_ref.extractall(expanded_path)

os.remove(f'{expanded_path}/training_and_validation.zip')

url = 'https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/test.mat.zip'
r = requests.get(url, allow_redirects=True)

open(f'{expanded_path}/test.zip', 'wb').write(r.content)

with zipfile.ZipFile(f'{expanded_path}/test.zip', 'r') as zip_ref:
    zip_ref.extractall(expanded_path)

os.remove(f'{expanded_path}/test.zip')


url = 'https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test_batches.zip'
r = requests.get(url, allow_redirects=True)

open(f'{transformed_path}/test.zip', 'wb').write(r.content)

with zipfile.ZipFile(f'{transformed_path}/test.zip', 'r') as zip_ref:
    zip_ref.extractall(transformed_path)

os.remove(f'{transformed_path}/test.zip')

print('Download Successful!')




train = scipy.io.loadmat(f'{expanded_path}/training_and_validation.mat')
test = scipy.io.loadmat(f'{expanded_path}/test.mat')

x_train = np.transpose(train['affNISTdata'][0][0][2], (1,0)).reshape(60000,40,40)
y_train = train['affNISTdata'][0][0][5]

random_60000_id = np.random.permutation(60000)

batch_size = 1000
current_index = 0

cnt=1
while current_index != 60:
    x_train_part = x_train[random_60000_id[current_index*batch_size:(current_index+1)*batch_size]]
    y_train_part = y_train[0][random_60000_id[current_index * batch_size:(current_index + 1) * batch_size]]
    random_169_id = np.random.permutation(169)

    img_list = []
    lbl_list = []

    for i in tqdm(range(169)):
        id_169 = random_169_id[i]
        x = int(id_169/13)-6
        y = int(id_169%13)-6
        for j in range(batch_size):
            new_image = np.zeros((40, 40), dtype=np.short)
            new_image[6 + x:34 + x, 6 + y:34 + y] = x_train_part[j][6:34, 6:34]
            lbl = y_train_part[j]
            img_list.append(new_image)
            lbl_list.append(lbl)
    imgs = np.stack(img_list, axis=0)
    lbls = np.stack(lbl_list, axis=0)
    filename = f'{expanded_path}/affnist_expanded_train_{cnt}'
    cnt += 1
    res = (imgs, lbls)
    with open(filename,'wb') as file:
        pickle.dump(res, file)

    current_index += 1

x_test = np.transpose(test['affNISTdata'][0][0][2], (1, 0)).reshape(10000, 40, 40)
y_test = test['affNISTdata'][0][0][5]

img_list = []
lbl_list = []
for i in range(-6, 7):
    for j in range(-6, 7):

        for test_cnt in tqdm(range(x_test.shape[0])):
            new_image = np.zeros((40, 40), dtype=np.short)
            new_image[6 + i:34 + i, 6 + j:34 + j] = x_test[test_cnt][6:34, 6:34]
            lbl = y_test[0][test_cnt]
            img_list.append(new_image)
            lbl_list.append(lbl)
imgs = np.stack(img_list, axis=0)
lbls = np.stack(lbl_list, axis=0)
res = (imgs, lbls)
filename = f'{expanded_path}/affnist_expanded_test.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('x_test', data=imgs)
    file.create_dataset('y_test', data=lbls)