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

from trainer import Trainer

import options
import utils

from models.transformer import (
    get_model,
    CausalLM,
    MaskedLM,
)


class LMDataset(data.Dataset):
    def __init__(self, path, text_field, encoding='utf-8', **kwargs):
        fields = [('text', text_field)]
        text = []
        with io.open(path, encoding=encoding) as f:
            examples = [data.Example.fromlist([line], fields) for line in f]
        super(LMDataset, self).__init__(examples, fields, **kwargs)


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
        path = os.path.join(basedir, 'src.field')
        TEXT = utils.load_field(path)

    fields = [('text', TEXT)] \
             if args.task in ['causal', 'masked'] \
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
        valid_data = data.TabularDataset(
            path=args.valid,
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
        valid_data = datasets.LanguageModelingDataset(
            path=args.valid, 
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
    utils.save_field(args.savedir, fields)
    utils.save_vocab(args.savedir, fields)

    # set training iterator
    if args.task == 'translation':
        train_iter = data.BucketIterator(
            train_data, 
            batch_size=args.batch_size,
            sort_within_batch=True,
            sort_key= lambda x: len(x.src),
            repeat=False,
        )
        valid_iter = data.BucketIterator(
            valid_data, 
            batch_size=args.batch_size,
            sort_within_batch=True,
            sort_key= lambda x: len(x.src),
            train=False,
            repeat=False,
            shuffle=False
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
        valid_iter = data.BPTTIterator(
            valid_data, 
            batch_size=args.batch_size, 
            bptt_len=args.bptt_len,
            train=False,
            repeat=False, 
            shuffle=False,
        )

    print(f'| [text] Dictionary: {len(TEXT.vocab.itos)} types')
    print('')

    for iter_name, iterator in [('train', train_iter), ('valid', valid_iter)]:
        file_path = args.train if iter_name == 'train' else args.valid
        data_object = train_data if iter_name == 'train' else valid_data
        print(f' {iter_name}: {file_path}')
        for name, field in fields:
            n_tokens, n_unk = utils.get_statics(iterator, name, field)
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
        model.load_state_dict(lm_weight)

    criterion = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi['<pad>'])
    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip, iteration)

    # show the details of model and optimizer
    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')
 
    max_epoch = (args.max_epoch or math.inf) + epoch
    max_update = (args.max_update or math.inf) + iteration

    while epoch < max_epoch and trainer.n_updates < max_update and args.min_lr < trainer.get_lr():
        # training
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            for samples in pbar:
                if args.task == 'translation':
                    srcs = samples.src.to(device)
                    tgts = samples.tgt.to(device)
                    refs = None
                else: # `causal`, `masked`
                    srcs = samples.text.to(device)
                    tgts = None
                    refs = None if args.task == 'masked' else samples.target.to(device)
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
        
        # validation
        if args.valid is not None:    
            valid_loss = 0.0
            trainer.model.eval()
            for samples in valid_iter:
                if args.task == 'translation':
                    srcs = samples.src.to(device)
                    tgts = samples.tgt.to(device)
                    refs = None
                else:
                    srcs = samples.text.to(device)
                    tgts = None
                    refs = None if args.task == 'masked' else samples.target.to(device)
                loss = trainer.step(srcs, tgts, refs)
                valid_loss += loss.item()
            valid_loss /= len(valid_iter)

            print(f'| epoch {str(epoch).zfill(3)} | valid ', end='') 
            print(f'| loss {valid_loss:.{4}} ', end='')
            print(f'| ppl {math.exp(valid_loss):.{4}} ', end='')
            print(f'| lr {trainer.get_lr():.1e} ', end='')
            print(f'| clip {args.clip} ', end='')
            print(f'| num_updates {trainer.n_updates} |')

        # saving model
        save_vars = {
            'epoch': epoch,
            'iteration': trainer.n_updates,
            'best_loss': valid_loss if valid_loss < best_loss else best_loss,
            'args': args, 
            'weights': model.state_dict()
        }

        if valid_loss < best_loss:
            filename = os.path.join(args.savedir, 'checkpoint_best.pt') 
            torch.save(save_vars, filename)
        if epoch % args.save_epoch == 0:
            filename = os.path.join(args.savedir, f'checkpoint_{epoch}.pt') 
            torch.save(save_vars, filename)
        filename = os.path.join(args.savedir, 'checkpoint_last.pt') 
        torch.save(save_vars, filename)

        # update
        trainer.scheduler.step(train_loss)
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('''
    ''')

    options.train_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
