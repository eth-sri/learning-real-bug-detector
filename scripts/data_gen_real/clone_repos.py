import os
import sys
import json
import argparse
import subprocess
from tqdm import tqdm

from realbuglearn.data.multiprocessing_utils import map_reduce

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, dest='in_file', required=True)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../repos', required=False)
    args = parser.parse_args()
    return args

args = get_args()

def clone(user_repo):
    user, repo = user_repo
    out_dir = os.path.join(args.out_dir, user)
    os.makedirs(out_dir, exist_ok=True)
    url = f'git@github.com:{user}/{repo}.git'
    cmd = f'git clone {url}'
    r = subprocess.call(cmd, shell=True, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r != 0:
        print(url)
        sys.stdout.flush()
    return f"{user}/{repo}"

def main():
    repos = set()
    with open(args.in_file) as f:
        for line in f:
            if line.strip() == "": continue
            user, repo = line[:-1].strip().split(' ')
            repos.add((user, repo))
    repos = list(sorted(repos))

    map_reduce(repos, clone, None, lambda a, b: None)
    
    cloned_repos = set()
    for user in os.listdir(args.out_dir):
        for repo in os.listdir(os.path.join(args.out_dir, user)):
            cloned_repos.add((user, repo))

    with open(args.in_file, 'w') as f:
        for user, repo in sorted(cloned_repos):
            f.write(f'{user} {repo}\n')

if __name__ == '__main__':
    main()