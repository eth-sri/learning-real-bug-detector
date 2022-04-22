import os
import sys
import json
import argparse
from tqdm import tqdm

from genbug.data.multiprocessing_utils import map_reduce
from genbug.data.cst_utils import iter_lines_in_dir, get_info, parse_with_indent_err
from genbug.data.rewrite import VarMisuseExtractor, WrongBinOpExtractor, ArgSwapExtractor, visit
from genbug.tokenizer.hf_tokenizer import CuBertHugTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--in_dir', type=str, dest='in_dir', default='../../../dataset', required=False)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../dataset', required=False)
    parser.add_argument('--vocab', type=str, dest='vocab', default='../../../pretrained/pretrained-epoch-2-20210711/20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt', required=False)
    args = parser.parse_args()
    return args

args = get_args()
sys.setrecursionlimit(10000)
tokenizer = CuBertHugTokenizer(args.vocab)

def check(line):
    j = json.loads(line)
    user, repo, path, func = get_info(j['info'], args.task)
    code = j['function'].strip()
    code = parse_with_indent_err(code)
    if code is None:
        return None
    else:
        try:
            if args.task == 'var-misuse':
                extractor = VarMisuseExtractor(code, tokenizer)
            elif args.task == 'wrong-binary-operator':
                extractor = WrongBinOpExtractor(code, tokenizer)
            elif args.task == 'argument-swap':
                extractor = ArgSwapExtractor(code, tokenizer)
            else:
                assert(False)
            visit(code, extractor)
            len_ops = 0
            for ops in extractor.ops.values():
                len_ops += len(ops)
            if len_ops == 0: return None
            out_j = {'user': user, 'repo': repo, 'path': path, 'func': func, 'code': code}
            return json.dumps(out_j, sort_keys=True)
        except:
            return None

def reduce(reduce_args, new_res):
    f_out, total, success, fail = reduce_args
    total += 1
    if new_res is None:
        fail += 1
    else:
        success += 1
        f_out.write(new_res+'\n')
    return f_out, total, success, fail

def main():
    data_dir = os.path.join(args.in_dir, args.task)
    total, success, fail = 0, 0, 0

    jsontxt_files = [fname for fname in sorted(os.listdir(data_dir)) if 'jsontxt' in fname]
    print("Scanning {} for samples".format(jsontxt_files))

    all_lines = list(iter_lines_in_dir(data_dir))
    correct_lines = []
    for line in all_lines:
        j = json.loads(line)
        if j['label'] == 'Correct':
            correct_lines.append(line)

    f_out = open(os.path.join(args.out_dir, args.task, 'cubert.jsontxt'), 'w')
    _, total, success, fail = map_reduce(correct_lines, check, (f_out, 0, 0, 0), reduce)
    f_out.close()
    print(f'total {total}, success {success}, fail {fail}')

if __name__ == '__main__':
    main()