# -*- coding: utf-8 -*-

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import TransformerDecoderLayer
    
from .positional_embedding import SinusoidalPositionalEmbedding
from .multihead_attention import MultiheadAttn
from .utils import Linear


def get_model(task):
    if task == 'causal':
        return CasualLM
    elif task in ['masked', 'translation']:
        return MaskedLM


def fill_ninf(t):
    return t.float().fill_(float('-inf')).type_as(t)


class BaseLM(nn.Module):
    def __init__(self, field, args):
        super(BaseLM, self).__init__()
        self.field = field
        self.vocabsize = len(field.vocab.itos)
        self.pad_idx = field.vocab.stoi['<pad>']
        self.bos_idx = field.vocab.stoi['<bos>']
        self.eos_idx = field.vocab.stoi['<eos>']
        self.sep_idx = field.vocab.stoi['<sep>']
        self.mask_idx = field.vocab.stoi['<mask>']
        self.bidirectional = args.bidirectional


        self.dropout = args.dropout
        self.embed_dim = args.embed_dim
        self.n_layers = args.layers

        self.w_embed = nn.Embedding(self.vocabsize, self.embed_dim) \
            if field.vocab.vectors is None \
            else nn.Embedding.from_pretrained(field.vocab.vectors, freeze=True)
        self.p_embed = SinusoidalPositionalEmbedding(self.embed_dim, self.pad_idx)
        self.embed_scale = math.sqrt(self.embed_dim)

        self.fwd_layers = nn.ModuleList(
            [TransformerDecoderLayer(args, no_encoder_attn=True) for _ in range(self.n_layers)])
        if args.bidirectional:
            self.bwd_layers = nn.ModuleList(
                [TransformerDecoderLayer(args, no_encoder_attn=True) for _ in range(self.n_layers)])

        self.out_projection = Linear(self.embed_dim, self.vocabsize)

    def forward(self, srcs, tgts, refs):
        raise NotImplementedError

    def loss(self, criterion, srcs, tgts, refs):
        raise NotImplementedError
        


class CausalLM(BaseLM):
    def __init__(self, field, args):
        super(CasualLM, self).__init__(field, args)
        self.task_name = 'CLM'
        assert not self.bidirectional, 'CausalLM does not contain `bidirectional` option'

    def forward(self, inputs, incremental_state=None):
        # embed positions
        positions = self.p_embed(inputs, incremental_state=None)

        x = self.w_embed(inputs)
        x *= self.embed_scale
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding mask
        fwd_pad_mask = inputs.eq(self.pad_idx)
        if not fwd_pad_mask.any():
            fwd_pad_mask = None

        self_attn_mask = self.buffered_future_mask(x)

        # decoder layers
        for layer in self.fwd_layers:
            x = layer(x, None, None, self_attn_mask, fwd_pad_mask, incremental_state)
        x = self.out_projection(x)
        return x

    def loss(self, criterion, srcs, tgts, refs):
        slen, bsz = srcs.size()
        outs = self.forward(srcs)
        loss = criterion(
            outs.view(outs.size(0)*bsz, -1),
            refs.view(-1)
        )
        return loss

    def buffered_future_mask(self, tensor, delim_idx=1):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None:
            self._future_mask = torch.triu(fill_ninf(tensor.new(dim, dim)), delim_idx)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                fill_ninf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class MaskedLM(BaseLM):
    def __init__(self, field, args, sampling=0.15, masked_rate=0.8,
                 replaced_rate=0.1, unchanged_rate=0.1):
        super(MaskedLM, self).__init__(field, args)
        self.task_name = 'MLM'
        self.sampling = sampling
        
        assert masked_rate + replaced_rate + unchanged_rate == 1.0, \
            'summation of perturabation_rates must be 1.0' 
        self.masked_rate = masked_rate
        self.replaced_rate = replaced_rate
        self.unchaged_rate = unchanged_rate

    def forward(self, inputs, incremental_state=None):
        # embed positions
        positions = self.p_embed(inputs, incremental_state=None)

        x = self.w_embed(inputs)
        x *= self.embed_scale
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding mask
        pad_mask = inputs.eq(self.pad_idx)
        if not pad_mask.any():
            pad_mask = None

        self_attn_mask = None if self.bidirectional else self.buffered_future_mask(x)

        for layer in self.fwd_layers:
            x = layer(x, None, None, self_attn_mask, pad_mask, incremental_state)
        x = self.out_projection(x)
        return x
 
    def loss(self, criterion, srcs, tgts, refs=None):
        refs = srcs if tgts is None else torch.cat((srcs, tgts))
        slen, bsz = refs.size()
        sampler = self._sampling(refs)

        rnd = torch.rand((slen, bsz), device=refs.device)
        mask = (self.masked_rate >= rnd) & sampler

        # replace mask tokens
        inputs = torch.where(
            (self.masked_rate >= rnd) & mask,
            torch.ones_like(refs) * self.mask_idx, 
            refs,
        )
   
        # replace random tokens
        th = self.masked_rate + self.replaced_rate
        inputs = torch.where(
            (th >= rnd) & (rnd > self.masked_rate) & sampler, 
            torch.randint_like(inputs, self.mask_idx+1, self.vocabsize),
            inputs,
        )
        import pdb; pdb.set_trace()
        outs = self.forward(inputs).view(slen*bsz, -1)
        loss = criterion(outs, refs.view(-1))
        return loss

    def _sampling(self, inputs):
        slen, bsz = inputs.size()
        rnd = -torch.rand((slen, bsz))
        mask = rnd.ge(-self.sampling)
        mask[inputs <= self.mask_idx] = 0 # special tokens are not sampled
        return mask.to(inputs.device)

    def buffered_future_mask(self, tensor, delim_idx=1):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None:
            self._future_mask = torch.triu(fill_ninf(tensor.new(dim, dim)), delim_idx)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                fill_ninf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
