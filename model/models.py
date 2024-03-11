#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 10:31
# @Author  : yuheng Ye
# @FileName: models.py
# @Software: PyCharm
import sys
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
    def __init__(self, backbone, in_chans, patch_len, dim=192, num_classes=11, eps=1E-8):
        super().__init__()
        self.encoder_q = backbone  # Encoder
        self.encoder_f = FEncoder(dim, in_chans, patch_len)
        self.patch_len = patch_len
        self.eps = eps
        self.temperature = 0.05
        self.cla_head = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.criterion = NTXentLoss()

    def fft(self, x):
        data = x
        x = x[:, 0] + 1j * x[:, 1]
        x = torch.fft.fft(x)
        x = torch.stack((x.real, x.imag), dim=1)
        return x

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

def get_model(backbone, patch_len, emb_size,in_chans, classes,backbone_pretrain=True):
    # Config backbone model
    if backbone == 'xciT':
        backbone_net = xcit_tiny12(num_classes=classes, in_chans=in_chans, patch_len=patch_len, pretrained=backbone_pretrain, path="./model/xcit_tiny12_online.pt")

    model = SSL(backbone=backbone_net, in_chans=in_chans, patch_len=patch_len, num_classes=classes, dim=emb_size)
    return model