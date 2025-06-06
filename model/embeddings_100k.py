import torch
import torch.nn as nn
import torch.nn.functional as F

class item_100k(torch.nn.Module):
    def __init__(self):
        super(item_100k, self).__init__()
        self.num_title = 1664
        self.num_release_date = 241
        self.embedding_dim = 10

        self.embedding_genre = torch.nn.Linear(19, self.embedding_dim, False)

        self.embedding_title = torch.nn.Embedding(
            num_embeddings=self.num_title,
            embedding_dim=self.embedding_dim
        )

        self.embedding_release_date = torch.nn.Embedding(
            num_embeddings=self.num_release_date,
            embedding_dim=self.embedding_dim
        )

        self.genre_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))
        self.title_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))
        self.release_date_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))


    def forward(self, x):
        genre_emb = self.embedding_genre(x[:, 8:].float()) / torch.sum(x[:, 8:].float(), 1).view(-1, 1)
        title_emb = self.embedding_title(x[:,6])
        release_dates_emb = self.embedding_release_date(x[:,7])
        genre_emb_weighted = genre_emb * self.genre_weight
        title_emb_weighted = title_emb * self.title_weight
        release_dates_emb_weighted = release_dates_emb * self.release_date_weight

        total_weight = self.genre_weight + self.title_weight + self.release_date_weight
        weighted_avg_emb = (genre_emb_weighted + title_emb_weighted + release_dates_emb_weighted) / total_weight

        return weighted_avg_emb

class user_100k(torch.nn.Module):
    def __init__(self):
        super(user_100k, self).__init__()
        self.num_gender = 2
        self.num_age = 61
        self.num_occupation = 21
        self.num_zipcode = 795
        self.embedding_dim = 10

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

        self.gender_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))
        self.age_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))
        self.occupation_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))
        self.area_weight = torch.nn.Parameter(torch.ones(self.embedding_dim))

    def forward(self, x):
        gender_emb = self.embedding_gender(x[:,3])
        age_emb = self.embedding_age(x[:,2])
        occupation_emb = self.embedding_occupation(x[:,4])
        area_emb = self.embedding_area(x[:,5])

        gender_emb_weighted = gender_emb * self.gender_weight
        age_emb_weighted = age_emb * self.age_weight
        occupation_emb_weighted = occupation_emb * self.occupation_weight
        area_emb_weighted = area_emb * self.area_weight

        total_weight = self.gender_weight + self.age_weight + self.occupation_weight + self.area_weight
        weighted_avg_emb = (gender_emb_weighted + age_emb_weighted + occupation_emb_weighted + area_emb_weighted) / total_weight

        return weighted_avg_emb








