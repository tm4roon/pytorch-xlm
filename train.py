# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
import torch.optim as optim


import options
import utils

from models.transformer import (
    get_model,
    CausalLM,
    MaskedLM,
)

monolingual_tasks = ['causal', 'masked']


class Trainer(object):
    def __init__(self, model, criterion, optimizer, clip, n_iter=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
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


def main(args):
    device = torch.device('cuda' if args.gpu  else 'cpu')

    if args.re_training is None:
        TEXT = data.Field(
            lower=True, 
            init_token='<bos>', 
            eos_token='<eos>'
        )
    else: 
        basedir, _ = os.path.split(args.re_training)
        path = os.path.join(basedir, 'text.field')
        TEXT = utils.load_field(path)

    fields = [('text', TEXT)] if args.task in monolingual_tasks \
                else [('src', TEXT), ('tgt', TEXT)]

    slen_filter = lambda x: args.src_minlen <= len(x.src) <= args.src_maxlen \
                         and args.tgt_minlen <= len(x.tgt) <= args.tgt_maxlen

    # load training data
    if args.task == 'translation':
        train_data = data.TabularDataset(
                path=args.train,
                format='tsv',
                fields=fields,
                filter_pred=slen_filter,
        )
    else: # `causal`, `masked`
        train_data = datasets.LanguageModelingDataset(
            path=args.train, 
            text_field=TEXT, 
            newline_eos=True
        )

    # set Vocabulary object
    if args.re_training is None:
        TEXT.build_vocab(
            train_data, 
            min_freq=args.min_freq, 
            specials=['<sep>', '<mask>'], 
        )
        if args.embed_path:
            vectors = utils.load_vector(args.embed_path)
            TEXT.vocab.load_vectors(vectors)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # save a field object
    with open(os.path.join(args.savedir, 'text.field'), 'wb') as fout:
        dill.dump(TEXT, fout)
    utils.save_vocab(args.savedir, TEXT)

    # set training iterator
    if args.task == 'translation':
        train_iter = data.BucketIterator(
            train_data, 
            batch_size=args.batch_size,
            sort_within_batch=True,
            sort_key= lambda x: len(x.src),
            repeat=False,
        )
    else: # `causal`, `masked`
        train_iter = data.BPTTIterator(
            train_data, 
            batch_size=args.batch_size, 
            bptt_len=args.bptt_len,
            train=True, 
            repeat=False, 
            shuffle=True,
        )

    print(f'| [text] Dictionary: {len(TEXT.vocab.itos)} types')
    print('')

    print(f'train: {args.train}')
    for name, field in fields:
        n_tokens, n_unk = utils.get_statics(train_iter, name, field)
        print(f'| [{name}] {n_tokens} tokens,', end='')
        print(f' coverage: {100*(n_tokens-n_unk)/n_tokens:.{4}}%')
    print('')

    # build a model
    model_class = get_model(args.task)

    if  args.re_training is None:
        epoch = 1
        iteration = 0
        best_loss = math.inf
        model = model_class(TEXT, args).to(device)
    else:
        load_vars = torch.load(args.re_training)
        epoch = load_vars['epoch'] + 1
        iteration = load_vars['iteration']
        best_loss = load_vars['best_loss']
        lm_args, lm_weights = load_vars['args'], load_vars['weights']
        model = model_class(TEXT, lm_args)
        model.load_state_dict(lm_weights)
        model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi['<pad>'])
    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)
    trainer = Trainer(model, criterion, optimizer, args.clip, iteration)

    # show the details of model and optimizer
    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')

    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    assert not(max_epoch == math.inf and max_update == math.inf), \
        'Please set `--max-epoch` or `--max-update`.'
 
    while epoch <= max_epoch and trainer.n_updates <= max_update:
        # training
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            for samples in pbar:
                if args.task in monolingual_tasks:
                    srcs = samples.text.to(device)
                    tgts = None
                    refs = None if args.task == 'masked' \
                            else samples.target.to(device)
                else:
                    srcs = samples.src.to(device)
                    tgts = samples.tgt.to(device)
                    refs = None
                loss = trainer.step(srcs, tgts, refs)
                train_loss += loss.item()

                # setting of progressbar
                pbar.set_description(f'epoch {str(epoch).zfill(3)}')
                progress_state = OrderedDict(
                    task=args.task,
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=srcs.size(1),
                    lr=trainer.get_lr(), 
                    clip=args.clip, 
                    num_updates=trainer.n_updates)
                pbar.set_postfix(progress_state)
        train_loss /= len(train_iter)

        print(f'| epoch {str(epoch).zfill(3)} | train ', end='') 
        print(f'| loss {train_loss:.{4}} ', end='')
        print(f'| ppl {math.exp(train_loss):.{4}} ', end='')
        print(f'| lr {trainer.get_lr():.1e} ', end='')
        print(f'| clip {args.clip} ', end='')
        print(f'| num_updates {trainer.n_updates} |')
        
        # saving model
        save_vars = {
            'epoch': epoch,
            'iteration': trainer.n_updates,
            'best_loss': train_loss if train_loss < best_loss else best_loss,
            'args': args, 
            'weights': model.state_dict()
        }

        if train_loss < best_loss:
            best_loss = train_loss
            filename = os.path.join(args.savedir, 'checkpoint_best.pt') 
            torch.save(save_vars, filename)
        if epoch % args.save_epoch == 0:
            filename = os.path.join(args.savedir, f'checkpoint_{epoch}.pt') 
            torch.save(save_vars, filename)
        filename = os.path.join(args.savedir, 'checkpoint_last.pt') 
        torch.save(save_vars, filename)

        # update
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('''
        An implementation of cross-lingual language model pre-training (XLM).
    ''')

    options.train_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
