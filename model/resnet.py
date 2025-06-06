import logging
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.float_embedding import FloatEmbedding
from model.embeddings_100k import item_100k, user_100k

class LookupEmbedding(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0])
        iid_emb = self.iid_embedding(x[:, 1])
        return uid_emb, iid_emb

class FloatLookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = FloatEmbedding(iid_all, emb_dim)

    def forward(self, x, stage=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.dtype is torch.float32:
            uid_idx = x[:, 0].type(torch.LongTensor).to(device)
            uid_emb = self.uid_embedding(uid_idx)
            iid_emb = self.iid_embedding(x[:, 1:])
            iid_emb = torch.sum(iid_emb, dim=1).unsqueeze(1)
            if stage is None:
                emb = torch.cat([uid_emb, iid_emb], dim=1)
                return emb
            if stage is 'save_inversed_iid':
                a = uid_idx.unsqueeze(2)
                # uid_idx_expanded = uid_idx.unsqueeze(2).expand(-1, -1, iid_emb.size(2))
                # emb = torch.cat([uid_idx_expanded, iid_emb], dim=2)
                emb = torch.cat([uid_idx.unsqueeze(2), iid_emb], dim=2)
                return emb
        else:
            uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
            iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], dim=1)
            return emb


class FRCSUModel(torch.nn.Module):
    # TODO change
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.user_attribute = torch.nn.Linear(emb_dim, emb_dim, False)
        self.user_pre = user_100k()
        self.global_item = torch.nn.Embedding(iid_all, emb_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, 1)
        )

        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.affine_output = torch.nn.Linear(2, 1, False)

    def forward(self, x, stage):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage in ['train_frcsu_warm']:
            uid_emb, iid_emb = self.embedding.forward(x)
            emb = torch.sum(uid_emb * iid_emb, dim=1)
            user_pre = self.user_pre.forward(x)
            user_repre = self.user_attribute(user_pre)
            mlp_score = self.mlp(torch.cat([user_repre, iid_emb], dim=1))
            return emb, uid_emb, user_repre, mlp_score
        elif stage in ['train_frcsu_cold']:
            uid_emb, iid_emb = self.embedding.forward(x)
            emb = torch.sum(uid_emb * iid_emb, dim=1).unsqueeze(1)
            mlp_score = self.mlp(torch.cat([uid_emb, iid_emb], dim=1))
            emb = self.affine_output(torch.cat([emb, mlp_score], dim=1))
            return emb.squeeze(1)
        elif stage in ['frcsu_map']:
            iid_emb = self.embedding.iid_embedding(x[:, 1])
            iid_emb_global = self.global_item(x[:, 1])
            iid_emb_map = self.mapping.forward(iid_emb)
            return iid_emb_global, iid_emb_map
        elif stage == 'test_frcsu':
            uid_emb, iid_emb = self.embedding.forward(x)
            user_pre = self.user_pre.forward(x)
            user_repre = self.user_attribute(user_pre)
            iid_map_emb = self.mapping.forward(iid_emb)
            mlp_score = self.mlp(torch.cat([user_repre, iid_map_emb], dim=1))
            emb = torch.sum(uid_emb * iid_emb, dim=1).unsqueeze(1)
            x = self.affine_output(torch.cat([emb, mlp_score], dim=1))
            return x.squeeze(1)



