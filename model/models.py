#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 10:31
# @Author  : yuheng Ye
# @FileName: models.py
# @Software: PyCharm
import sys
# print(sys.path)
sys.path.insert(0,'../')
from copy import deepcopy
from typing import List, Optional, Tuple
from model.backbones import xcit_tiny12, efficientnet_b0, efficientnet_b2, efficientnet_b4
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F

from LSM.utils.loss_utils import NTXentLoss


class FEncoder(nn.Module):
    def __init__(self, emd_size, in_chans, hidden_size):
        super().__init__()

        self.ext = nn.Sequential(

            nn.Conv1d(in_chans, 64, 1, 1), nn.BatchNorm1d(
                64), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 1, 1, 1), nn.BatchNorm1d(
                1), nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(hidden_size, 128), nn.BatchNorm1d(
                128), nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, emd_size), nn.BatchNorm1d(
                emd_size), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        f = self.ext(x)
        return f

class SSL(nn.Module):
    def __init__(self, backbone, in_chans, patch_len, K=65536, m=0.999, T=0.07, dim=192, num_classes=11, eps=1E-8, alpha=0.5,
                 diss=40, beta_1 =0.9 , beta_2 =0.1):
        super().__init__()
        self.encoder_q = backbone  # Encoder
        self.encoder_f = FEncoder(dim, in_chans, patch_len)
        self.patch_len = patch_len

        self.eps = eps
        self.temperature = 0.05

        self.p_mu = nn.Sequential(nn.Linear(dim, dim),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(dim, dim))

        self.cla_centers = nn.Parameter(torch.randn(
            1, diss, dim*2), requires_grad=True)

        self.p_logvar = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim),
            nn.Tanh())

        self.cla_head = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.contrast_head1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim)
        )

        self.contrast_head2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim)
        )

        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        self.criterion = NTXentLoss()

    def fft(self, x):
        data = x
        x = x[:, 0] + 1j * x[:, 1]
        x = torch.fft.fft(x)
        x = torch.stack((x.real, x.imag), dim=1)
        return x

    def contrast_logits(self, embd1, embd2, embd3):
        feat1 = F.normalize(self.contrast_head1(embd1), dim=1)

        feat2 = F.normalize(self.contrast_head2(embd2), dim=1)

        feat3 = F.normalize(self.contrast_head1(embd3), dim=1)
        return feat1, feat2, feat3

    def contra_loss(self, feat1, feat2, feat3):
        logit1, logit2, logit3 = self.contrast_logits(feat1, feat2, feat3)

        loss = self.pair_con_loss(logit1, logit2) + \
            self.pair_con_loss(logit1, logit3)

        # return loss, logit1, logit2
        return loss, logit2, logit3

    def pair_con_loss(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2,
                                  dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(
            torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        Ng = neg.sum(dim=-1)

        loss_pos = (- torch.log(pos / (Ng + pos))).mean()

        return loss_pos

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum(
            (embeddings.unsqueeze(1) - self.cla_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def cla_prob_loss(self, embeddings):
        porb = self.get_cluster_prob(embeddings)
        target = self.target_distribution(porb).detach()
        loss = self.kl_div((porb + 1e-08).log(), target) / porb.shape[0]
        return loss

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / (torch.sum(batch, 0) + self.eps)
        return (weight.t() / torch.sum(weight, 1)).t()

    def get_mu_logvar(self, x_samples1, x_samples2):
        mu1, logvar1 = self.p_mu(x_samples1), self.p_logvar(x_samples1)
        mu2, logvar2 = self.p_mu(x_samples2), self.p_logvar(x_samples2)
        return mu1, logvar1, mu2, logvar2

    def loglikeli(self, x1_samples, x2_samples):  # unnormalized loglikelihood
        mu1, logvar1, mu2, logvar2 = self.get_mu_logvar(x1_samples, x2_samples)
        like1 = (-(mu1 - x2_samples) ** 2 / logvar1.exp() - logvar1).mean()
        like2 = (-(mu2 - x1_samples) ** 2 / logvar2.exp() - logvar2).mean()
        return like1 + like2

    def mi_loss(self, logit2, logit3):
        return - self.loglikeli(logit2, logit3)

    # train_one_step
    def forward(self, data_t, data_t_a):
        f_aug_x = self.fft(data_t_a)
        feat1 = self.encoder_q(data_t)
        feat1 = F.normalize(feat1, dim=1)
        feat2 = self.encoder_f(f_aug_x)
        feat2 = F.normalize(feat2, dim=1)
        loss = self.criterion(feat1,feat2)
        return loss

    def train_one_cla_step(self, data_t, y, criterion):
        with torch.no_grad():
            feat1 = self.encoder_q(data_t)
            feat2 = self.encoder_f(self.fft(data_t))

            feat2 = F.normalize(feat2, dim=1)
            feat1 = F.normalize(feat1, dim=1)

        y_ = self.cla_head(torch.cat((feat1, feat2), dim=-1))

        loss = criterion(y_, y)
        return loss

    @torch.no_grad()
    def predict(self, x):
        patchs = x.shape[-1] // self.patch_len
        x = x[:, :, :patchs *
                     self.patch_len].reshape(x.shape[0], x.shape[1], patchs, self.patch_len)
        x = x.permute(0, 2, 1, 3)
        batchs = x.shape[0]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        with torch.no_grad():
            feature_q = self.encoder_q(x)
            feature_k = self.encoder_f(self.fft(x))

            feature_q = F.normalize(feature_q, dim=1)
            feature_k = F.normalize(feature_k, dim=1)

            feature = torch.cat((feature_q, feature_k), dim=-1).reshape(batchs, patchs, -1).sum(-2)
            y = self.cla_head(torch.cat((feature_q, feature_k), dim=-1))
            y = torch.softmax(
                0.1 * y.reshape(batchs, patchs, y.shape[-1]), dim=-1)
            y_max, _ = torch.max(y, dim=-1, keepdim=True)
            y = (y * (y_max)).sum(-2)

        return feature, y

def get_model(backbone, framework, patch_len, emb_size,in_chans, classes,centers,backbone_pretrain=True):
    # Config backbone model
    if backbone == 'efficient_net_b0':
        backbone_net = efficientnet_b0()
    elif backbone == 'efficient_net_b2':
        backbone_net = efficientnet_b2()
    elif backbone == 'efficient_net_b4':
        backbone_net = efficientnet_b4()
    elif backbone == 'resnet50':
        backbone_net = models.__dict__[backbone],
    elif backbone == 'xciT':
        backbone_net = xcit_tiny12(num_classes=classes, in_chans=in_chans, patch_len=patch_len, pretrained=backbone_pretrain, path="./model/xcit_tiny12_online.pt")


    model = SSL(backbone=backbone_net, in_chans=in_chans, patch_len=patch_len, num_classes=classes, dim=emb_size, diss=centers)
    return model