import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import pandas as pd

#calculate max_len
def cal_max_len(data):
    max_len = 0
    for d in data:
        max_len = max(len(d), max_len)
    print('sequence_max_len:: ', max_len)
    for p in range(3):
        if max_len % 4 == 0:
            break
        else: max_len += 1
    print('modified sequence_max_len:: ', max_len)

    return max_len

#equalize length
def equal(data, max_len):
    dataset = []
    for d in data:
        d = np.array(d)
        if len(d) < max_len:
            dataset.append(torch.cat([torch.tensor(d), torch.zeros((max_len - d.shape[-2]),d.shape[-1])]))
        else:
            dataset.append(torch.tensor(d[:max_len]))
    return dataset

#dataset
class superDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

#read data
def read_data(data_path, batch_size, fold):
    df = pd.read_csv(data_path)

    num_augmented = len(list(set(df[df['is_test']==0]['MaterialID'].values)))//len(list(set(df[df['target']==1]['MaterialID'].values)))

    augmented = []
    i = 0
    for ID in list(set(df[df['target']==1]['MaterialID'].values)):
        for _ in range(num_augmented):
            tdata = df[df['MaterialID']==ID]
            tdata['MaterialID']=10000+i
            tdata['is_test']=0
            i += 1
            for _ in range(3):
                t = random.randint(1, 10)
                t0 = random.randint(0, tdata.shape[0]-t)
                tdata.iloc[t0:t0+t, 3:-2] = 0
            augmented.append(tdata)

    augmented_dataset = pd.concat(augmented).reset_index(drop=True)
    raw_train_data = df[df['is_test']==0]
    raw_test_data = df[df['is_test']==1]


    all_train_normal_data = []
    all_train_normal_label = []
    for ID in list(set(raw_train_data['MaterialID'].values)):
        all_train_normal_data.append(raw_train_data[raw_train_data['MaterialID']==ID].iloc[:, 3:-2])
        all_train_normal_label.append(raw_train_data[raw_train_data['MaterialID']==ID].iloc[-1, -1])

    input_size = np.array(all_train_normal_data[0]).shape[-1]  # input_size


    all_train_abnormal_data = []
    all_train_abnormal_label = []
    for ID in list(set(augmented_dataset['MaterialID'].values)):
        all_train_abnormal_data.append(augmented_dataset[augmented_dataset['MaterialID']==ID].iloc[:, 3:-2])
        all_train_abnormal_label.append(augmented_dataset[augmented_dataset['MaterialID']==ID].iloc[-1, -1])


    test_data = []
    test_label = []
    for ID in list(set(raw_test_data['MaterialID'].values)):
        test_data.append(raw_test_data[raw_test_data['MaterialID']==ID].iloc[:, 3:-2])
        test_label.append(raw_test_data[raw_test_data['MaterialID']==ID].iloc[-1, -1])


    train_data = all_train_normal_data[:int(len(all_train_normal_data)*0.2*(fold-1))]+all_train_normal_data[int(len(all_train_normal_data)*0.2*fold):]+all_train_abnormal_data[:int(len(all_train_abnormal_data)*0.2*(fold-1))]+all_train_abnormal_data[int(len(all_train_abnormal_data)*0.2*fold):]
    valid_data = all_train_normal_data[int(len(all_train_normal_data)*0.2*(fold-1)):int(len(all_train_normal_data)*0.2*fold)]+all_train_abnormal_data[int(len(all_train_abnormal_data)*0.2*(fold-1)):int(len(all_train_abnormal_data)*0.2*fold)]
    train_label = all_train_normal_label[:int(len(all_train_normal_label)*0.2*(fold-1))]+all_train_normal_label[int(len(all_train_normal_label)*0.2*fold):]+all_train_abnormal_label[:int(len(all_train_abnormal_label)*0.2*(fold-1))]+all_train_abnormal_label[int(len(all_train_abnormal_label)*0.2*fold):]
    valid_label = all_train_normal_label[int(len(all_train_normal_label)*0.2*(fold-1)):int(len(all_train_normal_label)*0.2*fold)]+all_train_abnormal_label[int(len(all_train_abnormal_label)*0.2*(fold-1)):int(len(all_train_abnormal_label)*0.2*fold)]
    
    max_len = cal_max_len(train_data)
    train_data = equal(train_data, max_len)
    valid_data = equal(valid_data, max_len)
    test_data = equal(test_data, max_len)

    #dataset
    train_dataset = superDataset(train_data, train_label)
    valid_dataset = superDataset(valid_data, valid_label)
    test_dataset = superDataset(test_data, test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, valid_loader, test_loader, input_size, max_len

#read all data
def read_all_data(data_path, max_len):
    df = pd.read_csv(data_path)

    num_augmented = len(list(set(df[df['is_test']==0]['MaterialID'].values)))//len(list(set(df[df['target']==1]['MaterialID'].values)))

    augmented = []
    i = 0
    for ID in list(set(df[df['target']==1]['MaterialID'].values)):
        for _ in range(num_augmented):
            tdata = df[df['MaterialID']==ID]
            tdata['MaterialID']=10000+i
            tdata['is_test']=0
            i += 1
            for _ in range(3):
                t = random.randint(1, 10)
                t0 = random.randint(0, tdata.shape[0]-t)
                tdata.iloc[t0:t0+t, 3:-2] = 0
            augmented.append(tdata)

    all_augmented_dataset = pd.concat([df, pd.concat(augmented)]).reset_index(drop=True)

    all_data = []
    all_label = []
    for ID in list(set(all_augmented_dataset['MaterialID'].values)):
        all_data.append(all_augmented_dataset[all_augmented_dataset['MaterialID']==ID].iloc[:, 3:-2])
        all_label.append(all_augmented_dataset[all_augmented_dataset['MaterialID']==ID].iloc[-1, -1])

    all_data = equal(all_data, max_len)

    all_dataset = superDataset(all_data, all_label)
    all_loader = DataLoader(all_dataset, batch_size=1)

    return all_loader




