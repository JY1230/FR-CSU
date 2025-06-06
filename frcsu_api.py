import copy
import logging
import random
import numpy as np
import pandas as pd
import torch
import math
from math import sqrt
from utils import transform_list_to_tensor

from client_frcsu import Client


class FRCSU(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [client_data_all, train_data_num, test_data_num,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset
        self.client_all = client_data_all
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_indexes = []
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_rmse = []
        self.test_rmse = []
        self.train_mae = []
        self.test_mae = []
        self.train_mse = []
        self.test_mse = []
        self.model_trainer = model_trainer
        self.results = {'test_frcsu_mae_min': 10, 'test_frcsu_rmse_min': 10}

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, copy.deepcopy(model_trainer))
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae_min']:
            self.results[phase + '_mae_min'] = mae
        if rmse < self.results[phase + '_rmse_min']:
            self.results[phase + '_rmse_min'] = rmse

    def train(self):
        for round_idx in range(self.args.comm_round):
            w_global = self.model_trainer.get_model_params()

            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []
            self._client_sampling(round_idx, self.args.client_num,
                                  self.args.client_num_per_round)
            # logging.info("client_indexes = " + str(self.client_indexes))

            for idx in self.client_indexes:
                client_idx = idx
                for i in self.client_list:
                    if i.client_idx == client_idx:
                        client = i

                if client.client_idx < self.args.client_num_warm:
                    w_client_local = client.model_trainer.get_model_params()

                    w_client_local['global_item.weight'] = copy.deepcopy(w_global['embedding.iid_embedding.weight'])
                    w_client_local['embedding.iid_embedding.weight'] = copy.deepcopy(w_global['embedding.iid_embedding.weight'])
                    w_client_local['user_attribute.weight'] = copy.deepcopy(w_global['user_attribute.weight'])

                    weight = client.train(copy.deepcopy(w_client_local), mapping=False)
                    w_locals.append((client.local_sample_number, copy.deepcopy(weight)))
                else:
                    w_client_local = client.model_trainer.get_model_params()
                    weight = client.train(copy.deepcopy(w_client_local), mapping=True)

            w_global = self._aggregate(w_locals)

            if (round_idx + 1) % 2 == 0:
                self._local_test_on_all_clients(round_idx, w_global)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            self.client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            self.client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]

        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        global_model_params = averaged_params
        self.model_trainer.set_model_params(global_model_params)

        return global_model_params

    def _local_test_on_all_clients(self, round_idx, w_global):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics_frcsu = {
            'num_samples': [],
            'losses': [],
            'mae': [],
        }
        
        test_metrics_frcsu = {
            'num_samples': [],
            'losses': [],
            'mae': [],
        }

        for client in self.client_list[self.args.client_num_warm:]:
            w_client_test = client.model_trainer.get_model_params()
            w_client_test['global_item.weight'] = copy.deepcopy(w_global['embedding.iid_embedding.weight'])
            w_client_test['embedding.iid_embedding.weight'] = copy.deepcopy(w_global['embedding.iid_embedding.weight'])
            w_client_test['user_attribute.weight'] = copy.deepcopy(w_global['user_attribute.weight'])
            test_local_metrics_frcsu = client.local_test(True, copy.deepcopy(w_client_test))

            test_metrics_frcsu['num_samples'].append(copy.deepcopy(test_local_metrics_frcsu['test_total']))
            test_metrics_frcsu['losses'].append(copy.deepcopy(test_local_metrics_frcsu['test_loss']))
            test_metrics_frcsu['mae'].append(copy.deepcopy(test_local_metrics_frcsu['test_mae']))
        
        test_loss_frcsu = math.sqrt(sum(test_metrics_frcsu['losses'])/ sum(test_metrics_frcsu['num_samples']))
        test_mae_frcsu = sum(test_metrics_frcsu['mae']) / sum(test_metrics_frcsu['num_samples'])

        stats = {'test_frcsu_mae': test_mae_frcsu,
                 'test_frcsu_rmse': test_loss_frcsu}
        logging.info(stats)

        self.test_mae.append(test_mae_frcsu)
        self.test_rmse.append(test_loss_frcsu)

        self.update_results(test_mae_frcsu, test_loss_frcsu, 'test_frcsu')       
        logging.info(self.results)


