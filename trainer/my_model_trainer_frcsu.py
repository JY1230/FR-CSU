import logging
import traceback
import copy
import torch
import os.path
import json
import time
import tqdm
import random
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from trainer.model_trainer import ModelTrainer
from torch.utils.data import DataLoader, TensorDataset

class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args, config):
        self.model = model
        self.id = 0
        self.args = args
        self.config = config
        self.epoch = config['epoch']
        self.lr = config['lr']
        self.wd = config['wd']
        self.lr_attribute = config['lr_attribute']
        self.lr_mlp = config['lr_mlp']
        self.lr_mapping = config['lr_mapping']
        self.lr_output = config['lr_output']

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, data_loader, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model
            model.to(device)
            model.train()

            # train and update
            criterion = torch.nn.MSELoss().to(device)
            optimizer_warm = torch.optim.Adam([{'params': model.embedding.parameters(), 'lr': self.lr},
                                               {'params': model.user_attribute.parameters(), 'lr':self.lr_attribute},
                                               {'params': model.mlp.parameters(), 'lr': self.lr_mlp},
                                               {'params': model.user_pre.parameters(), 'lr': self.lr_attribute}],
                                              weight_decay=self.wd)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_warm,
                                                                   mode="min",
                                                                   factor=0.2,
                                                                   patience=2)

            epoch_loss = []
            for epoch in range(args.epoch):
                batch_loss = []
                model.train()
                for X, y in data_loader:
                    X, y = X.to(device), y.to(device)
                    model.zero_grad()
                    pred, uid_emb, user_repre, mlp_score = model(X, stage='train_frcsu_warm')
                    y = y.squeeze().float()
                    y = y.view_as(pred)
                    loss = criterion(pred, y) + criterion(mlp_score, y) + criterion(uid_emb, user_repre)
                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer_warm.step()

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
            #     logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            # logging.info("--------------------------------------------------------------------------")

        except Exception as e:
            logging.error(traceback.format_exc())


    def train_cold(self, data_loader, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model

            model.to(device)
            model.train()

            # train and update
            criterion = torch.nn.MSELoss().to(device)

            optimizer_cold = torch.optim.Adam([{'params': model.embedding.parameters(), 'lr': self.lr},
                                               {'params': model.affine_output.parameters(), 'lr': self.lr_output},
                                               {'params': model.mlp.parameters(), 'lr': self.lr_mlp}
                                               ], weight_decay=self.wd)


            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cold,
                                                                   mode="min",
                                                                   factor=0.2,
                                                                   patience=2)

            epoch_loss = []
            for epoch in range(args.epoch):
                batch_loss = []
                model.train()
                for X, y in data_loader:
                    X, y = X.to(device), y.to(device)
                    model.zero_grad()
                    pred = model(X, stage='train_frcsu_cold')
                    y = y.squeeze().float()
                    y = y.view_as(pred)
                    loss = criterion(pred, y)

                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer_cold.step()

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
            #     logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            # logging.info("--------------------------------------------------------------------------")

        except Exception as e:
            logging.error(traceback.format_exc())


    def train_map(self, data_loader, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model

            model.to(device)
            model.train()

            # train and update
            criterion = torch.nn.MSELoss().to(device)
            optimizer_cold = torch.optim.Adam([
                                               {'params': model.mapping.parameters(), 'lr': self.lr_mapping}
                                               ], weight_decay=self.wd)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cold,
                                                                   mode="min",
                                                                   factor=0.2,
                                                                   patience=2)

            epoch_loss = []
            for epoch in range(args.epoch):
                batch_loss = []
                model.train()
                for X, y in data_loader:
                    X, y = X.to(device), y.to(device)
                    model.zero_grad()

                    item_client, item_global = model(X, stage='frcsu_map')
                    loss = criterion(item_global, item_client)

                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer_cold.step()

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))

        except Exception as e:
            logging.error(traceback.format_exc())

    def test(self, data_loader, device, args):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_loss': 0,
            'test_mae': 0,
            'test_total': 0,
        }

        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        all_targets, all_predicts = list(), list()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X, stage='test_frcsu')
                all_targets.extend(y.tolist())
                all_predicts.extend(pred.tolist())
                targets = y.tolist()
                predicts = pred.tolist()
                targets = torch.tensor(targets).float()
                predicts = torch.tensor(predicts)
                metrics['test_loss'] += mse_loss(targets, predicts).item() * targets.size(0)
                metrics['test_mae'] += loss(targets, predicts).item() * targets.size(0)
                metrics['test_total'] += targets.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
