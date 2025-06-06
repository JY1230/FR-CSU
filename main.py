import argparse
import logging
import os
import random
import sys
import datetime
import numpy as np
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data_loader import load_partition_data
from model.resnet import FRCSUModel
from frcsu_api import FRCSU
from trainer.my_model_trainer_frcsu import MyModelTrainer as MyModelTrainerCLS

import warnings

warnings.filterwarnings('ignore')

def prepare():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--model', type=str, default='FR-CSU', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='movielens100k', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='././data/',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training')

    parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                        help='learning rate')
    
    parser.add_argument('--lr_attribute', type=float, default=0.1, metavar='LR')

    parser.add_argument('--lr_mlp', type=float, default=0.05, metavar='LR')

    parser.add_argument('--lr_mapping', type=float, default=0.1, metavar='LR')

    parser.add_argument('--lr_output', type=float, default=0.05, metavar='LR')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0)

    parser.add_argument('--client_num', type=int, default=943, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_warm', type=int, default=800, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_cold', type=int, default=143, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=100, metavar='NN',
                        help='number of workers')

    parser.add_argument('--hidden_dim', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--baseline', default="FRCSU",
                        help='Training model')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we should use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--task', default='ml_100k', help='ml_100k: movielens100k'
                                                    'ml_1m: movielens1m')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    config = vars(args)
    return args, config


def load_data(args, dataset_name, device):
    logging.info("load_data. dataset_name = %s" % dataset_name)
    rating_counts, client_data_all, train_data_num, test_data_num, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict\
    = load_partition_data(args.batch_size, args.client_num, args.client_num_warm, args.data_dir, dataset_name, device)

    dataset = [client_data_all, train_data_num, test_data_num,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict]

    return dataset

def create_model(args, dataset_name, model_name, hidden_dim):
    logging.info("create_model. model_name = %s, hidden_dim = %s" % (model_name, hidden_dim))
    model = None
    if model_name == "FR-CSU":
        if dataset_name == 'ml_100k':
            model = FRCSUModel(uid_all=1, iid_all=1682, emb_dim=hidden_dim)
    return model

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    args, config = prepare()

    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset, device)
    model = create_model(args, dataset_name=args.task, model_name=args.model, hidden_dim=args.hidden_dim)
    model_trainer = MyModelTrainerCLS(model, args, config)

    if args.baseline == "FRCSU":
        FRCSU_API = FRCSU(dataset, device, args, model_trainer)
        FRCSU_API.train()


