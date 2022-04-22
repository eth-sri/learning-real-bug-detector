import os
import json
import random
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--in_dir', type=str, dest='in_dir', default='../../../dataset', required=False)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../dataset', required=False)
    parser.add_argument('--seed', type=int, dest='seed', default=42)
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)

def main():
    real_repos_path = os.path.join(args.in_dir, args.task, 'repos_to_exclude.txt')
    real_repos = set()
    with open(real_repos_path) as f:
        for line in f.readlines():
            user, repo = line.strip().split(' ')
            real_repos.add((user, repo))

    cubert_dataset_path = os.path.join(args.in_dir, args.task, 'cubert.jsontxt')
    big_code_repos = dict()
    with open(cubert_dataset_path) as f:
        for line in f.readlines():
            try:
                j = json.loads(line)
                user, repo = j['user'], j['repo']
                if (user, repo) in real_repos: continue
                if (user, repo) not in big_code_repos:
                    big_code_repos[(user, repo)] = list()
                big_code_repos[(user, repo)].append(line)
            except: 
                print("error with line '{}'".format(line))

    repos = list(sorted(big_code_repos.keys()))
    random.shuffle(repos)
    train_percent = 100
    if args.task == 'var-misuse':
        train_percent = 43
    elif args.task == 'wrong-binary-operator':
        train_percent = 85
    elif args.task == 'argument-swap':
        train_percent = 50
    else:
        assert(False)

    split_point = round(len(repos)*train_percent/100)
    filtered_repos = repos[split_point:]
    repos = repos[:split_point]

    out_path = os.path.join(args.out_dir, args.task, f'bigcode.filtered.jsontxt')
    with open(out_path, 'w') as f:
        for user, repo in tqdm(sorted(filtered_repos)):
            for line in big_code_repos[(user, repo)]:
                f.write(line)

if __name__ == '__main__':
    main()