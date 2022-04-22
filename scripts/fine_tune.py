import os
import math
import json
import torch
import numpy
import random
import argparse
import subprocess
from transformers import BertConfig, TrainingArguments, Trainer, get_scheduler

from realbuglearn import config
from realbuglearn.data.dataset import MyDataset, MyDataCollator
from realbuglearn.model.bert_models import MyBertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--model', type=str, dest='model', required=True)
    parser.add_argument('--dataset', type=str, dest='dataset', choices=['real', 'synthetic', 'contrastive'], required=True)

    parser.add_argument('--target', type=str, dest='target', choices=['cls', 'loc', 'loc-tgt', 'cls-locpos', 'cls-locpos-tgt', 'cls-locpos-tgt-contrastive'], default='cls-locpos-tgt', required=False)

    parser.add_argument('--cls_loss', type=str, dest='cls_loss', choices=['ce', 'focal'], default='focal', required=False)
    parser.add_argument('--ptr_loss', type=str, dest='ptr_loss', choices=['original', 'focal'], default='original', required=False)
    parser.add_argument('--contrastive_weight', type=float, dest='contrastive_weight', default=None, required=False)
    parser.add_argument('--pred_head', type=str, dest='pred_head', choices=['flat', 'cls-locpos-tgt', 'cls-tgt-locpos', 'locpos-cls-tgt', 'locpos-tgt-cls', 'tgt-cls-locpos', 'tgt-locpos-cls'], default=None, required=False)

    parser.add_argument('--train_percent', type=int, dest='train_percent', default=100, required=False)
    parser.add_argument('--neg_pos_ratio', type=int, dest='neg_pos_ratio', default=-1, required=False)
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=None, required=False)
    parser.add_argument('--in_dir', type=str, dest='in_dir', default='../../dataset', required=False)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../fine-tuned', required=False)
    parser.add_argument('--pretrained', type=str, dest='pretrained', default='../../pretrained/pretrained-epoch-2-20210711/length-512/pytorch_model.bin', required=False)
    parser.add_argument('--config', type=str, dest='config', default='../../pretrained/pretrained-epoch-2-20210711/length-512/config.json', required=False)
    parser.add_argument('--seed', type=int, dest='seed', default=42)
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)

def get_lr():
    if args.dataset in ('synthetic', 'contrastive'):
        lr = 1e-5
        if args.task == 'var-misuse':
            lr = 1e-6
    elif args.dataset == 'real':
        lr = 1e-6
    else:
        assert(False)

    return lr

def get_num_epochs():
    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    else:
        if args.dataset in ('synthetic', 'contrastive'):
            num_epochs = 1
        elif args.dataset == 'real':
            if args.task == 'wrong-binary-operator':
                num_epochs = 2
            elif args.task == 'var-misuse':
                num_epochs = 2
            elif args.task == 'argument-swap':
                num_epochs = 1
            else:
                assert(False)
        else:
            assert(False)

    return num_epochs

def get_contrastive_weight():
    if args.contrastive_weight is not None:
        weight = args.contrastive_weight
    else:
        if args.task == 'wrong-binary-operator':
            weight = 4
        elif args.task == 'var-misuse':
            weight = 0.5
        elif args.task == 'argument-swap':
            weight = 0.5
        else:
            assert(False)

    return weight

def get_pred_head():
    if args.pred_head is not None:
        pred_head = args.pred_head
    else:
        if args.task == 'wrong-binary-operator':
            pred_head = 'tgt-locpos-cls'
        elif args.task == 'var-misuse':
            pred_head = 'cls-locpos-tgt'
        elif args.task == 'argument-swap':
            pred_head = 'tgt-locpos-cls'
        else:
            assert(False)

    return pred_head


def main():
    target = args.target
    if args.dataset == 'contrastive':
        target = 'cls-locpos-tgt-contrastive'
    dataset = MyDataset(os.path.join(args.in_dir, args.task, f'{args.dataset}.train.dataset'))
    if args.train_percent != 100:
        dataset.subset_percent_by_repo(args.train_percent)
    dataset.prepare(target)
    dataset.shuffle()
    if args.neg_pos_ratio != -1:
        dataset.neg_pos_ratio(args.neg_pos_ratio)

    lr = get_lr()
    num_epochs = get_num_epochs()
    bert_config = BertConfig.from_json_file(args.config)
    bert_config.task = args.task
    bert_config.target = target
    bert_config.cls_loss = args.cls_loss
    bert_config.ptr_loss = args.ptr_loss
    bert_config.pred_head = get_pred_head()
    bert_config.contrastive_weight = get_contrastive_weight()
    model = MyBertModel.from_pretrained(args.pretrained, config=bert_config).to(device)
    save_dir = os.path.join(args.out_dir, args.task, target, args.model)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'fine_tune.args'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, sort_keys=True)+'\n')

    train_batch_size = config.TRAIN_BATCH_SIZE
    if 'contrastive' in target and train_batch_size > 1:
        train_batch_size = int(train_batch_size / 2)

    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=0,
        per_device_train_batch_size=train_batch_size,
        do_eval=False,
        logging_dir=save_dir,
        logging_strategy='steps',
        logging_steps=config.LOGGING_STEPS,
        save_strategy='epoch',
        save_total_limit=1
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=MyDataCollator()
    )
    trainer.train()

if __name__ == '__main__':
    main()