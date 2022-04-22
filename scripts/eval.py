import os
import torch
import numpy
import pickle
import random
import argparse
from tqdm import tqdm
from transformers import BertConfig

import torch.nn.functional as F

from realbuglearn import config
from realbuglearn.data.dataset import MyDataset, MyDataCollator
from realbuglearn.model.bert_models import MyBertModel
from realbuglearn.model.metrics import MyMetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--model', type=str, dest='model', required=True)

    parser.add_argument('--target', type=str, dest='target', choices=['cls', 'loc', 'loc-tgt', 'cls-locpos', 'cls-locpos-tgt', 'cls-locpos-tgt-contrastive'], default='cls-locpos-tgt', required=False)
    parser.add_argument('--eval_percent', type=int, dest='eval_percent', default=100, required=False)
    parser.add_argument('--probs_file', type=str, dest='probs_file', default=None, required=False, help="File to save the prediction probabilities to.")

    parser.add_argument('--split', type=str, dest='split', choices=['train', 'val', 'test'], default='val', required=False)
    parser.add_argument('--model_dir', type=str, dest='model_dir', default='../../fine-tuned', required=False)
    parser.add_argument('--dataset_dir', type=str, dest='dataset_dir', default='../../dataset', required=False)
    parser.add_argument('--seed', type=int, dest='seed', default=42)
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)

class ProbsSaver:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        if args.probs_file is not None:
            self.probs_file = args.probs_file + "-" + self.checkpoint + ".pt"
            print("Storing probabilities in {}".format(self.probs_file))
        self.res = {}
    
    def flush(self):
        if args.probs_file is None: return

        with open(self.probs_file, 'wb') as f:
            pickle.dump(self.res, f)

    def add(self, outputs, inputs):
        if args.probs_file is None: return

        res = {}
        res["user"] = inputs["user"]
        res["repo"] = inputs["repo"]
        res["path"] = inputs["path"]
        res["code"] = inputs["code"]
        
        res["cls_labels"] = inputs["cls_labels"].tolist()
        res["loc_correct_masks"] = inputs["loc_correct_masks"].tolist()
        res["tgt_correct_masks"] = inputs["tgt_correct_masks"].tolist()

        logits_cls, logits_loc, logits_tgt = outputs.logits_cls, outputs.logits_loc, outputs.logits_tgt

        cls = torch.argmax(logits_cls, dim=1).tolist()
        if 'cls' not in res: res['cls'] = list()
        res['cls'] += cls
        probs_cls = F.softmax(logits_cls, dim=1).tolist()
        if 'cls_prob' not in res: res['cls_prob'] = list()
        for i, label in enumerate(cls):
            res['cls_prob'].append(probs_cls[i][label])

        loc = torch.argmax(logits_loc, dim=1).tolist()
        if 'loc' not in res: res['loc'] = list()
        res['loc'] += loc
        probs_loc = F.softmax(logits_loc, dim=1).tolist()
        if 'loc_prob' not in res: res['loc_prob'] = list()
        for i, label in enumerate(loc):
            res['loc_prob'].append(probs_loc[i][label])

        tgt = torch.argmax(logits_tgt, dim=1).tolist()
        if 'tgt' not in res: res['tgt'] = list()
        res['tgt'] += tgt
        probs_tgt = F.softmax(logits_tgt, dim=1).tolist()
        if 'tgt_prob' not in res: res['tgt_prob'] = list()
        for i, label in enumerate(tgt):
            res['tgt_prob'].append(probs_tgt[i][label])

        self._add_res(res)

    def _add_res(self, r):
        for key in r.keys():
            if key not in self.res.keys(): 
                self.res[key] = list()
            self.res[key] += r[key]

def eval_model(model_dir, dataloader):
    print(model_dir)
    bert_config = BertConfig.from_json_file(os.path.join(model_dir, 'config.json'))
    if args.target == 'cls-locpos-tgt-contrastive':
        bert_config.target = 'cls-locpos-tgt'
    model = MyBertModel.from_pretrained(model_dir, config=bert_config).to(device)
    model.eval()

    num_batches, pbar, metrics = len(dataloader), tqdm(enumerate(dataloader)), MyMetrics(args.task, 'cls-locpos-tgt' if args.target == 'cls-locpos-tgt-contrastive' else args.target)
    probs_saver = ProbsSaver(os.path.basename(model_dir))

    for batch_idx, inputs in pbar:
        with torch.no_grad():
            for key, val in inputs.items():
                if torch.is_tensor(val):
                    inputs[key] = val.to(device)
            outputs = model(**inputs)
            metrics.add(outputs, inputs)
            probs_saver.add(outputs, inputs)
            pbar.set_description(f'batch {batch_idx+1}/{num_batches}, {str(metrics)}')
    
    probs_saver.flush()
    print(metrics)

def main():
    eval_dataset = MyDataset(os.path.join(args.dataset_dir, args.task, f'real.{args.split}.dataset'))
    if args.eval_percent != 100:
        eval_dataset.subset_percent_by_repo(args.eval_percent)
    eval_dataset.prepare(args.target)
    dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=MyDataCollator(), batch_size=config.TEST_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model_dir = os.path.join(args.model_dir, args.task, args.target, args.model)
    checkpoints = []
    for d in os.listdir(model_dir):
        if d.startswith('checkpoint-'):
            checkpoints.append(int(d[11:]))

    for checkpoint in sorted(checkpoints):
        eval_model(os.path.join(model_dir, f'checkpoint-{checkpoint}'), dataloader)

if __name__ == '__main__':
    main()