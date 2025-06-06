import logging
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn.functional as F
import copy

class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.id = 0

    def train(self, w_global, mapping):
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)
        if mapping:
            self.model_trainer.train_cold(self.local_training_data, self.device, self.args)
            self.model_trainer.train_map(self.local_training_data, self.device, self.args)
        else:
            self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset, w_client_test):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_client_test)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics