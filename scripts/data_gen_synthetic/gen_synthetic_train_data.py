import os
import json
import argparse
import libcst as cst
from tqdm import tqdm
import termcolor
import numpy as np
import pickle

from realbuglearn.data.multiprocessing_utils import map_reduce

from realbuglearn.tokenizer.hf_tokenizer import CuBertHugTokenizer
from realbuglearn.data.rewrite import visit, VarMisuseExtractor, WrongBinOpExtractor, ArgSwapExtractor, NoOpFoundException
from realbuglearn.data.sparse_sample import sparsify_sample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--contrastive', action='store_true', default=False, required=False)
    parser.add_argument('--random_by_key', action='store_true', default=False, required=False)
    parser.add_argument('--no-parallel', dest='no_parallel', action='store_true', default=False)
    parser.add_argument('--in_dir', type=str, dest='in_dir', default='../../../dataset', required=False)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../dataset', required=False)
    parser.add_argument('--vocab', type=str, dest='vocab', default='../../../pretrained/pretrained-epoch-2-20210711/20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt', required=False)
    parser.add_argument('--seed', type=int, dest='seed', default=42)
    args = parser.parse_args()
    return args

args = get_args()
tokenizer = CuBertHugTokenizer(args.vocab)
np.random.seed(args.seed)

def generate_synthetic_sample(line):
    j = json.loads(line)
    code = j['code']

    if args.task == 'var-misuse':
        extractor = VarMisuseExtractor(code, tokenizer)
    elif args.task == 'wrong-binary-operator':
        extractor = WrongBinOpExtractor(code, tokenizer)
    elif args.task == 'argument-swap':
        extractor = ArgSwapExtractor(code, tokenizer)
    else:
        assert False, "Not a supported task {}".format(args.task)

    visit(code, extractor)

    if args.random_by_key:
        keys = list(sorted(extractor.ops.keys()))
        tgt_key = keys[np.random.randint(0, len(keys))]
        ops = extractor.ops[tgt_key]
        tgt_op = ops[np.random.randint(0, len(ops))]
    else:
        # collect all possible rewrite operations on 'code'
        all_ops = []
        for key in sorted(extractor.ops.keys()):
            ops = extractor.ops[key]
            for op in ops:
                all_ops += [op]
        # apply some random choice of operation
        tgt_op = all_ops[np.random.randint(0, len(all_ops))]

    try:
        # synthetic positive sample
        pos_sample = tgt_op.tensorize(code, tokenizer)
        pos_sample['user'], pos_sample['repo'], pos_sample['path'] = j['user'], j['repo'], j['path']
        # negative sample
        neg_sample = extractor.tensorize(tokenizer, with_label=True)
        neg_sample['user'], neg_sample['repo'], neg_sample['path'] = j['user'], j['repo'], j['path']

        return sparsify_sample(pos_sample), sparsify_sample(neg_sample)
    except NoOpFoundException:
        print("Could not apply sampled rewrite operation at {} -> {} in {}/{}/{}".format(tgt_op.loc_range.start, tgt_op.loc_range.end, j["user"], j["repo"], j["path"]))
        return []
    except Exception as e:
        print("Failed to generate synthetic sample for {}/{}/{}".format(j["user"], j["repo"], j["path"]))
        return []

def main():
    data_path = os.path.join(args.in_dir, args.task, f'bigcode.train.jsontxt')
    if not args.contrastive:
        out_file = os.path.join(args.in_dir, args.task, f'synthetic.train.dataset')
    else:
        out_file = os.path.join(args.in_dir, args.task, f'contrastive.train.dataset')
    
    with open(data_path) as f:
        lines = f.readlines()

    def collect_pos_and_neg_sample(samples, extracted_samples):
        if len(extracted_samples) == 0: return samples
        if not args.contrastive:
            samples += extracted_samples
            return samples

        pos_sample, neg_sample = extracted_samples
        sample = dict()
        for key in pos_sample:
            sample[key+'1'] = pos_sample[key]
        for key in neg_sample:
            sample[key+'2'] = neg_sample[key]
        samples.append(sample)

        return samples

    # generate samples in parallel
    samples = map_reduce(lines, generate_synthetic_sample, list(), collect_pos_and_neg_sample, desc="Generating synthetic bugs...", no_parallel=args.no_parallel)

    # write samples to disk
    dataset = dict()
    for sample in tqdm(samples, desc="Writing samples to disk..."):
        for key in sample:
            # if key == "tgt_candidate_masks": continue
            if key not in dataset:
                dataset[key] = list()
            dataset[key].append(sample[key])
    with open(out_file, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    main()