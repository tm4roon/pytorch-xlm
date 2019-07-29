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


# def get_task(method):
#     if method in ['casual', 'translation']:
#         return CasualLM
#     elif method in ['masked', 'maskedtranslation']:
#         return MaskedLM
# 
# 
# class Task(object):
#     def __init__(self, model, criterion):
#         self.model = model
#         self.criterion = criterion
# 
#     def loss(self, srcs, tgts, refs):
#         raise NotImplementedError
# 
# 
# class CasualLM(Task):
#     def __init__(self, model, criterion):
#         super().__init__(model, criterion)
#         self.task_name = 'CLM'
# 
#     def loss(self, srcs, tgts, refs):
#         slen, bsz = srcs.size()
# 
#         if tgts is None:
#             outs = self.model(srcs)
#             loss = self.criterion(
#                 outs.view(outs.size(0)*bsz, -1),
#                 refs.view(-1)
#             )
#         else: # translation
#             outs = self.model(srcs, tgts[:-1])
#             loss = self.criterion(
#                 outs[slen:].view(-1, outs.size(2)), 
#                 tgts.view(-1)
#             )
#         return loss
# 
# 
# class MaskedLM(Task):
#     def __init__(self, model, criterion, sampling=0.15, masked_rate=0.8,
#                  replaced_rate=0.1, unchanged_rate=0.1):
#         super().__init__(model, criterion)
#         self.task_name = 'MLM'
#         self.vocabsize = model.decoder.vocabsize
#         self.pad_idx = model.pad_idx
#         self.bos_idx = model.bos_idx
#         self.eos_idx = model.eos_idx
#         self.mask_idx = model.mask_idx
#         self.sampling = sampling
#         
#         assert masked_rate + replaced_rate + unchanged_rate == 1.0, \
#             'summation of perturabation_rates must be 1.0' 
#         self.masked_rate = masked_rate
#         self.replaced_rate = replaced_rate
#         self.unchaged_rate = unchanged_rate
#    
#     def loss(self, srcs, tgts, refs):
#         inputs = srcs if tgts is None else torch.cat((srcs, tgts))
#         slen, bsz = inputs.size()
#         sampler = self._sampling(inputs)
# 
#         rnd = torch.rand((slen, bsz), device=inputs.device)
#         mask = (self.masked_rate >= rnd) & sampler
# 
#         refs = torch.where(
#             ~mask, 
#             torch.ones_like(inputs) * self.pad_idx,
#             inputs,
#         )
#         
#         # replace mask tokens
#         inputs = torch.where(
#             (self.masked_rate >= rnd) & mask,
#             torch.ones_like(inputs) * self.mask_idx, 
#             inputs,
#         )
#    
#         # replace random tokens
#         th = self.masked_rate + self.replaced_rate
#         inputs = torch.where(
#             (th >= rnd) & (rnd > self.masked_rate) & sampler, 
#             torch.randint_like(inputs, self.mask_idx+1, self.vocabsize),
#             inputs,
#         )
# 
#         outs = self.model(inputs).view(slen*bsz, -1)
#         loss = self.criterion(outs, refs.view(-1))
#         return loss
# 
#     def _sampling(self, inputs):
#         slen, bsz = inputs.size()
#         rnd = -torch.rand((slen, bsz))
#         mask = rnd.ge(-self.sampling)
#         mask[inputs <= self.mask_idx] = 0 # special tokens are not sampled
#         return mask.to(inputs.device)
