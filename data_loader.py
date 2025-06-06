import json
import logging
import os
import csv
import random
import numpy as np
import torch
import h5py
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample
import random
from itertools import combinations
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_data(dataset, data_dir, client_num_in_total, client_num_warm):
    set_seed(42)
    if dataset == 'movielens100k':
        data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', names=['uid', 'iid', 'y', 'timestamp'])
        data = data.drop('timestamp', axis=1)
        rating_counts = data['y'].value_counts(normalize=True).sort_index()
        user_data = pd.read_csv(
            os.path.join(data_dir, 'u.user'),
            names=['uid', 'age', 'gender', 'occupation', 'zip'],
            sep="|", engine='python'
        )
        item_data = pd.read_csv(
            os.path.join(data_dir, 'u.item'),
            names=['iid', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'],
            sep="|", engine='python', encoding="latin-1"
        )

        merged_data = pd.merge(data, user_data, on='uid')
        merged_data = pd.merge(merged_data, item_data, on='iid')
        merged_data = merged_data.drop(columns='IMDb_URL', axis=1)
        merged_data = merged_data.drop(columns='video_release_date', axis=1)
        label_encoders = {}
        for col in ['age', 'gender', 'occupation', 'zip', 'title', 'release_date']:
            le = LabelEncoder()
            merged_data[col] = le.fit_transform(merged_data[col])
            label_encoders[col] = le


    data = merged_data[['uid', 'iid', 'y', 'age', 'gender', 'occupation', 'zip', 'title', 'release_date', 'unknown', 'Action',
                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']]
    data['uid'] = data['uid'] - 1
    data['iid'] = data['iid'] - 1

    all_users = data['uid'].unique()
    all_items = data['iid'].unique()

    num_clients = client_num_in_total

    clients = {i: {'data': pd.DataFrame(), 'train_data': pd.DataFrame(), 'test_data': pd.DataFrame()} for i in range(num_clients)}

    pd.set_option('display.max_columns', None)
    for client_id in range(client_num_in_total):
        client_data = data[data['uid'] % client_num_in_total == client_id]
        clients[client_id]['data'] = client_data
        if client_id < client_num_warm:
            train_data = data[data['uid'] % client_num_in_total == client_id]
            train_data['uid'] = 0
            train_size = int(len(train_data) * 0.8)
            clients[client_id]['train_data'] = train_data[:train_size]
            clients[client_id]['test_data'] = train_data[train_size:]
        else:
            test_data = data[data['uid'] % client_num_in_total == client_id]
            test_data['uid'] = 0
            test_size = int(len(test_data) * 0.2)
            clients[client_id]['train_data'] = test_data[:test_size]
            clients[client_id]['test_data'] = test_data[test_size:]
    return data, clients, rating_counts


def batch_data_train(data, batch_size):
    X = torch.tensor(data[['uid', 'iid', 'age', 'gender', 'occupation', 'zip', 'title', 'release_date', 'unknown', 'Action',
                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']].values, dtype=torch.long)
    y = torch.tensor(data['y'].values, dtype=torch.long)
    # if self.use_cuda:
    X = X.cuda()
    y = y.cuda()
    dataset = TensorDataset(X, y)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return data_iter

def batch_data_test(data ,batch_size):
    X = torch.tensor(data[['uid', 'iid', 'age', 'gender', 'occupation', 'zip', 'title', 'release_date', 'unknown', 'Action',
                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']].values, dtype=torch.long)
    y = torch.tensor(data['y'].values, dtype=torch.long)
    # if self.use_cuda:
    X = X.cuda()
    y = y.cuda()
    dataset = TensorDataset(X, y)
    data_iter = DataLoader(dataset, batch_size, shuffle = True)
    return data_iter

def load_partition_data(batch_size,
                        client_num_in_total,
                        client_num_warm,
                        data_dir,
                        dataset,
                        device):
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0

    data, clients, rating_counts = read_data(dataset, data_dir, client_num_in_total, client_num_warm)

    logging.info("loading data...")

    for client_id in range(client_num_in_total):
        user_train_data_num = len(clients[client_id]['train_data'])
        user_test_data_num = len(clients[client_id]['test_data'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_id] = user_train_data_num

        train_batch = batch_data_train(clients[client_id]['train_data'], batch_size)
        test_batch = batch_data_test(clients[client_id]['test_data'], batch_size)
        train_data_global += train_batch
        test_data_global += test_batch

        train_data_local_dict[client_id] = train_batch
        test_data_local_dict[client_id] = test_batch
        client_idx += 1

    return rating_counts, clients, train_data_num, test_data_num, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict
