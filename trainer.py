# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, clip, n_iter=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.n_updates = n_iter

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, srcs, tgts, refs):
        self.optimizer.zero_grad()
        loss = self.model.loss(self.criterion, srcs, tgts, refs)

        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.n_updates += 1
        return loss
