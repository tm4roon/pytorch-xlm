# -*- coding: utf-8 -*-


def train_opts(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--task', default='causal',
        choices=['causal', 'masked', 'translation'],
        help='training method')
    group.add_argument('--re-training', default=None,
        help='path to trained model')
    group.add_argument('--train', required=True,
        help='filename of the train data')
    group.add_argument('--src-minlen', type=int, default=0,
        help='minimum sentence length of source side')
    group.add_argument('--tgt-minlen', type=int, default=0,
        help='minimum sentence length of target side')
    group.add_argument('--src-maxlen', type=int, default=100,
        help='maximum sentence length of source side')
    group.add_argument('--tgt-maxlen', type=int, default=100,
        help='maximum sentence length of target side')
    group.add_argument('--batch-size', type=int, default=64, 
        help='batch size')
    group.add_argument('--bptt-len', type=int, default=50,
        help='length of sequences for backpropagation through time')
    group.add_argument('--savedir', default='./checkpoints', 
        help='path to save models')
    group.add_argument('--max-epoch', type=int, default=0, 
        help='number of epochs')
    group.add_argument('--max-update', type=int, default=0,
        help='number of updates')
    group.add_argument('--lr', type=float, default=0.25,
        help='learning rate')
    group.add_argument('--min-lr', type=float, default=1e-5, 
        help='minimum learning rate')
    group.add_argument('--clip', type=float, default=1.0,
        help='gradient cliping')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    group.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad'],
        default='sgd', help='optimizer')
    group.add_argument('--save-epoch', type=int, default=10)


def model_opts(parser):
    group = parser.add_argument_group('Model\'s hyper-parameters')
    group.add_argument('--embed-dim', type=int, default=256,
        help='dimension of word embeddings of decoder')
    group.add_argument('--embed-path', default=None,
        help='pre-trained word embeddings')
    group.add_argument('--min-freq', type=int, default=0,
        help='map words appearing less than threshold times to unknown')
    group.add_argument('--hidden-dim', type=int, default=1024,
        help='number of hidden units per decoder layer')
    group.add_argument('--layers', type=int, default=4,
        help='number of layers')
    group.add_argument('--heads', type=int, default=8,
        help='number of attention heads')
    group.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 means no dropout)')
    group.add_argument('--activation-dropout', type=float, default=0.1,
        help='dropout after activation fucntion in self attention')
    group.add_argument('--attention-dropout', type=float, default=0.1,
        help='dropout in self attention')
    group.add_argument('--bidirectional', action='store_true')
    group.add_argument('--tied-embed', action='store_true',
        help='tie the word embedding and softmax weight')
    return group
