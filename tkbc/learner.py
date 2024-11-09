# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from collections import defaultdict
from typing import Dict
import logging
import torch
from torch import optim
import pandas as pd
import os
import json
import numpy as np
import time
from sklearn.decomposition import PCA
import pickle

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, BiTComplEx
from regularizers import N3, Lambda3, DURA
from utils import load_model, save_model, set_seed

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'BiTComplEx'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--seed', default=42, type=int,
    help="Random seed"
)
parser.add_argument(
    '--load', default=None, type=str,
    help="Resume from weight"
)
parser.add_argument(
    '--predict', default=None, type=int
)


args = parser.parse_args()

set_seed(args.seed)

dataset = TemporalDataset(args.dataset)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'BiTComplEx': BiTComplEx(sizes, args.rank)
}[args.model]
model = model.cuda()


opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)

# result = {
#     'Epoch': [],
#     'Split': [],
#     'MRR': [],
#     'Hits@1': [],
#     'Hits@3': [],
#     'Hits@10': [],
#     'Head_MRR': [],
#     'Head_Hits@1': [],
#     'Head_Hits@3': [],
#     'Head_Hits@10': [],
#     'Tail_MRR': [],
#     'Tail_Hits@1': [],
#     'Tail_Hits@3': [],
#     'Tail_Hits@10': []
# }

result = defaultdict(list)

def append_log(epoch, split_result):
    for split in ['train', 'valid', 'test']:
        result['Epoch'].append(epoch)
        result['Split'].append(split)
        for metric in split_result[split].keys():
            result[metric].append(split_result[split][metric])
        result['Loss'].append(split_result['loss'])

def append_interval_log(epoch, split_result):
    for split in ['train', 'valid', 'test']:
        processed = {}
        for metric in split_result[split].keys():
            if metric.startswith('hits@_'):
                postfix = metric.replace('hits@', '')
                processed['Hits@1' + postfix] = split_result[split][metric][0].item()
                processed['Hits@3' + postfix] = split_result[split][metric][1].item()
                processed['Hits@10' + postfix] = split_result[split][metric][2].item()
            else:
                processed[metric] = split_result[split][metric]
        split_result[split] = processed
        result['Epoch'].append(epoch)
        result['Split'].append(split)
        result['Loss'].append(split_result['loss'])
        max_metric_sz = 0
        for metric in split_result[split].keys():
            result[metric].append(split_result[split][metric])
            if len(result[metric]) > max_metric_sz:
                max_metric_sz = len(result[metric])
        for metric in result.keys():
            if len(result[metric]) == 1 and metric in split_result[split] and max_metric_sz > 1:
                result[metric] = ((max_metric_sz - len(result[metric])) * [None]) + result[metric]
            elif len(result[metric]) < max_metric_sz:
                result[metric].append(None)

best_test_mrr = 0
best_results = {}
epoch_time = []

if not os.path.exists('./weights/'):
    os.makedirs('./weights')

model_start_time = round(time.time() * 1000)

start_epoch = 0
if args.load is not None:
    model, opt, result, start_epoch = load_model(f'./weights/{args.load}', model, opt)
    start_epoch += 1
    model_start_time = args.load.split('_')[-1]
print(f'Start from epoch {start_epoch}')
print(result)

start_ent_pca_file_name = f'./tkbc/output/{args.model}_{args.dataset}_{args.rank}_{args.learning_rate}_{args.emb_reg}_{args.time_reg}_ent_init.pkl'
all_entities = list(dataset.ent2id.values())

if args.predict is not None:
    ex = torch.from_numpy(np.array([dataset.get_examples('test').astype('int64')[args.predict]], dtype=np.int64)).cuda()
    ranks = model.get_ranking(ex, dataset.to_skip['rhs'], batch_size=1)
    print("RESULT: ")
    print((ranks == 1).nonzero(as_tuple=True)[0])
    exit(0)

for epoch in range(start_epoch, args.max_epochs):
    start_epoch = round(time.time() * 1000)
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    loss = None
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        loss = optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        loss = optimizer.epoch(examples)

    epoch_time.append(round(time.time() * 1000) - start_epoch)

    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {
            'MRR': m, 
            'Hits@1': h[0].item(), 
            'Hits@3': h[1].item(), 
            'Hits@10': h[2].item(),
            'Head_MRR': mrrs['lhs'],
            'Head_Hits@1': hits['lhs'][0].item(), 
            'Head_Hits@3': hits['lhs'][1].item(), 
            'Head_Hits@10': hits['lhs'][2].item(),
            'Tail_MRR': mrrs['rhs'],
            'Tail_Hits@1': hits['rhs'][0].item(), 
            'Tail_Hits@3': hits['rhs'][1].item(), 
            'Tail_Hits@10': hits['rhs'][2].item()
        }

    if epoch <= 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            epoch_result = {
                'train': train,
                'valid': valid,
                'test': test,
                'loss': loss
            }
            append_interval_log(epoch, epoch_result)
            if (best_test_mrr < test['MRR_all']):
                best_test_mrr = test['MRR_all']
                best_results = epoch_result
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            epoch_result = {
                'train': train,
                'valid': valid,
                'test': test,
                'loss': loss
            }
            append_log(epoch, epoch_result)
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            print("train: ", train['MRR'])
            if (best_test_mrr < test['MRR']):
                best_test_mrr = test['MRR']
                best_results = epoch_result
    
    save_model(f'./weights/{args.model}_{args.dataset}_{args.rank}_{model_start_time}', args, model, optimizer.optimizer, result, epoch)

best_results['avg_millis_per_epoch'] = np.mean(epoch_time)
print('BEST RESULTS: ', best_results)

if not os.path.exists('./tkbc/output'):
    os.makedirs('./tkbc/output')

result_file_name = f'./tkbc/output/{args.model}_{args.dataset}_{args.rank}_{args.learning_rate}_{args.emb_reg}_{args.time_reg}.csv'
best_result_file_name = f'./tkbc/output/{args.model}_{args.dataset}_{args.rank}_{args.learning_rate}_{args.emb_reg}_{args.time_reg}_best.json'
distance_file_name = f'./tkbc/output/{args.model}_{args.dataset}_{args.rank}_{args.learning_rate}_{args.emb_reg}_{args.time_reg}_distances.pkl'
ent_pca_file_name = f'./tkbc/output/{args.model}_{args.dataset}_{args.rank}_{args.learning_rate}_{args.emb_reg}_{args.time_reg}_ent.pkl'

output_files = [result_file_name, best_result_file_name, distance_file_name, ent_pca_file_name]

for file_name in output_files:
    if (os.path.exists(file_name)):
        os.remove(file_name)

with open(best_result_file_name, 'w') as f:
    json.dump(best_results, f)

try:
    df = pd.DataFrame(data=result)
    df.to_csv(result_file_name)
except Exception:
    c = {}
    for key in result.keys():
        c[key] = len(result[key])
    print(c)

