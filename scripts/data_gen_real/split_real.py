import os
import json
import sys
import random
import pickle
import argparse
from tqdm import tqdm

from libcst._exceptions import ParserSyntaxError

from realbuglearn.tokenizer.hf_tokenizer import CuBertHugTokenizer
from realbuglearn.data.rewrite import DataConstructionException, VarMisuseExtractor, WrongBinOpExtractor, ArgSwapExtractor, visit
from realbuglearn.data.multiprocessing_utils import map_reduce
from realbuglearn.data.deduplicate import deduplicate, ensure_dependencies as deduplicate_ensure_dependencies
from realbuglearn.data.sparse_sample import sparsify_sample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--train_ratio', type=float, dest='train_ratio', default=0.5, required=False)
    parser.add_argument('--val_ratio', type=float, dest='val_ratio', default=0.25, required=False)
    parser.add_argument('--pos_dir', type=str, dest='pos_dir', default='../../../real-bugs', required=False)
    parser.add_argument('--neg_dir', type=str, dest='neg_dir', default='../../../dataset', required=False)
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../dataset', required=False)
    parser.add_argument('--neg_sampling_p', type=float, dest='neg_sampling_p', default=1.0, required=False, help='probability of sampling a negative sample from the negative dataset')
    parser.add_argument('--exclude_user', type=str, dest='exclude_user', default='', required=False)
    parser.add_argument('--skip_filter_near_duplicates', dest='skip_filter_near_duplicates', action='store_true', default=False)
    parser.add_argument('--vocab', type=str, dest='vocab', default='../../../pretrained/pretrained-epoch-2-20210711/20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt', required=False)
    parser.add_argument('--seed', type=int, dest='seed', default=42, required=False)
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)

tokenizer = CuBertHugTokenizer(args.vocab)

def dump_repos(repos, path):
    with open(path, "w") as f:
        for user, repo in sorted(repos):
            f.write(f'{user} {repo}\n')

def dump_dataset(repos, repo_to_real, repo_to_bigcode, path):
    dataset = dict()
    for user, repo in repos:
        real = repo_to_real[(user, repo)]
        for key in real:
            if key not in dataset:
                dataset[key] = list()
            dataset[key] += real[key]
        if not (user, repo) in repo_to_bigcode:
            print("info: no additional negative samples could be sourced from {}/{} which does contain files with real bugs.".format(user, repo))
            continue
        bigcode = repo_to_bigcode[(user, repo)]
        for key in bigcode:
            if key not in dataset:
                dataset[key] = list()
            dataset[key] += bigcode[key]

    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

# collect samples and corresponding paths of real bugs
real_bugs_dataset = dict()
real_bugs_paths = set()
repos_to_exclude = set()

def try_extract_candidates(line):
    j = json.loads(line)
    user, repo, code, path = j['user'], j['repo'], j['code'], j['path']
    
    if (user, repo) not in real_bugs_dataset: return []

    # filter out files used as positive sample
    if (user, repo, path) in real_bugs_paths: return []

    if args.task == 'var-misuse':
        extractor = VarMisuseExtractor(code, tokenizer)
    elif args.task == 'wrong-binary-operator':
        extractor = WrongBinOpExtractor(code, tokenizer)
    elif args.task == 'argument-swap':
        extractor = ArgSwapExtractor(code, tokenizer)
    else:
        assert(False)

    try:
        visit(code, extractor)
        res = extractor.tensorize(tokenizer, with_label=True)
        res['user'], res['repo'], res['path'] = user, repo, path

        if random.random() <= args.neg_sampling_p:
            return [((user, repo), res)]
        else:
            return []
    except ParserSyntaxError:
        print("skip sample: parser error")
        return [] # skip any non-parseable files in cubert.jsontxt
    except DataConstructionException:
        print("skip sample: data construction error")
        return []
    except Exception as e:
        print("unknown exception when extracting from {} {} {}".format(user, repo, len(extractor.ops)))
        return []

def add_to_dataset(dataset, user, repo, sample):
    if (user,repo) not in dataset.keys():
        dataset[(user,repo)] = dict()
    sample = sparsify_sample(sample)
    for key in sample.keys():
        if key not in dataset[(user,repo)].keys():
            dataset[(user,repo)][key] = list()
        dataset[(user,repo)][key].append(sample[key])

def main():
    deduplicate_ensure_dependencies()
    pos_dir = os.path.join(args.pos_dir, args.task, '1')

    # collect all real-bug samples extracted by inspecting the git history
    all_real_bug_samples = list()

    for user in tqdm(os.listdir(pos_dir), desc="Scanning repositories with real bugs...", leave=False):
        if user == args.exclude_user: continue
        
        for repo in tqdm(list(os.listdir(os.path.join(pos_dir, user, '2'))), leave=False, desc=user):
            p = os.path.join(pos_dir, user, '2', repo, 'stdout')
            with open(p) as f:
                lines = f.readlines()
            for line in tqdm(lines, leave=False, desc="{}/{}".format(user, repo)):
                if line.strip() == '': continue

                data = json.loads(line)
                repos_to_exclude.add((data["user"], data["repo"]))
                real_bugs_paths.add((data["user"], data["repo"], data["path"]))

                if "code" not in data.keys(): continue
                assert data["cls_labels"] == 0 or data["cls_labels"] == 1

                all_real_bug_samples.append(data)

    # separate pos/neg cases, otherwise the individual pos/neg pairs will be detected as duplicates
    pos_real_bugs = [s for s in all_real_bug_samples if s["cls_labels"] == 1]
    neg_real_bugs = [s for s in all_real_bug_samples if s["cls_labels"] == 0]

    if not args.skip_filter_near_duplicates:
        # remove near-duplicates from positive samples
        num_non_duplicate = 0
        for sample in tqdm(deduplicate(pos_real_bugs, "code", lambda i,s: i), desc="Removing near-duplicates from real, positive samples"):
            add_to_dataset(real_bugs_dataset, sample['user'], sample['repo'], sample)
            num_non_duplicate += 1
        # for sample in tqdm(deduplicate(neg_real_bugs, "code", lambda i,s: i), desc="Removing near-duplicates from real, negative samples"):
        #     add_to_dataset(real_bugs_dataset, sample['user'], sample['repo'], sample)
        #     num_non_duplicate += 1
        print("{}/{} non-duplicate samples".format(num_non_duplicate, len(pos_real_bugs)))
    else:
        print("Skipping near-duplicate removal (--skip_filter_near_duplicates)")
        for sample in all_real_bug_samples:
            add_to_dataset(real_bugs_dataset, sample['user'], sample['repo'], sample)
        print("{} real-bug samples".format(len(all_real_bug_samples)))

    # collect additional negatives samples by scanning all repositories with real bugs for files that were not used yet
    negative_samples_dataset = dict()
    cubert_path = os.path.join(args.neg_dir, args.task, 'cubert.jsontxt')
    with open(cubert_path) as f:
        lines = f.readlines()

    def reduce(negative_samples_dataset, new_candidates):
        for (user,repo), candidate in new_candidates:
            candidate = sparsify_sample(candidate)
            if (user, repo) not in negative_samples_dataset:
                negative_samples_dataset[(user, repo)] = dict()
            for key in candidate:
                if key not in negative_samples_dataset[(user, repo)]:
                    negative_samples_dataset[(user, repo)][key] = list()
                negative_samples_dataset[(user, repo)][key].append(candidate[key])
        return negative_samples_dataset
    
    negative_samples_dataset = map_reduce(list(lines), try_extract_candidates, negative_samples_dataset, reduce, desc="Parsing candidates")

    repos = list(sorted(real_bugs_dataset.keys()))
    random.shuffle(repos)
    train_split_point = round(len(repos)*args.train_ratio)
    val_split_point = round(len(repos)*args.val_ratio) + train_split_point
    train_repos, val_repos, test_repos = repos[:train_split_point], repos[train_split_point:val_split_point], repos[val_split_point:]

    dump_repos(repos_to_exclude, os.path.join(args.out_dir, args.task, 'repos_to_exclude.txt'))
    dump_repos(repos, os.path.join(args.out_dir, args.task, 'real_repos.txt'))
    dump_repos(train_repos, os.path.join(args.out_dir, args.task, 'train_repos.txt'))
    dump_repos(val_repos, os.path.join(args.out_dir, args.task, 'val_repos.txt'))
    dump_repos(test_repos, os.path.join(args.out_dir, args.task, 'test_repos.txt'))

    dump_dataset(train_repos, real_bugs_dataset, negative_samples_dataset, os.path.join(args.out_dir, args.task, 'real.train.dataset'))
    dump_dataset(val_repos, real_bugs_dataset, negative_samples_dataset, os.path.join(args.out_dir, args.task, 'real.val.dataset'))
    dump_dataset(test_repos, real_bugs_dataset, negative_samples_dataset, os.path.join(args.out_dir, args.task, 'real.test.dataset'))

if __name__ == '__main__':
    main()