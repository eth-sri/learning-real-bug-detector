import json
import os
import sys
import argparse
import lizard
import hashlib
from libcst._exceptions import ParserSyntaxError
from multiprocessing import Pool
from tqdm import tqdm

from realbuglearn.data.rewrite import DataConstructionException, VarMisuseExtractor, WrongBinOpExtractor, ArgSwapExtractor, visit
from realbuglearn.data.deduplicate import deduplicate, ensure_dependencies as dedup_ensure_dependencies
from realbuglearn.tokenizer.hf_tokenizer import CuBertHugTokenizer
from realbuglearn.data.multiprocessing_utils import map_reduce
from realbuglearn.data.cst_utils import get_func_src

"""
This scripts generates an unlabeled dataset in the CuBERT .jsontxt format, given the original
py150_files and the deduplicated selection of files as used for the CuBERT fine-tuning datasets.
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--py150_files', type=str, dest='py150_files', default='../../../py150_files')
    parser.add_argument('--ethpy150open_files', type=str, dest='ethpy150open_files', default='../../../eth_py150_open')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default='../../../dataset', required=False)
    parser.add_argument('--subset-only', default=False, action="store_true", dest='subset_only', help="only processes a small subset of files (for testing purposes)")
    parser.add_argument('--skip-near-duplicate-removal', default=False, action="store_true", dest='skip_near_duplicate_removal', help="disables the (time-intensive) removal of near duplicate samples")
    parser.add_argument('--vocab', type=str, dest='vocab', default='../../../pretrained/pretrained-epoch-2-20210711/20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt', required=False)
    args = parser.parse_args()
    return args

args = get_args()
# early termination if dependencies are missing
dedup_ensure_dependencies()

def get_dataset_relative_path(filepath):
    prefix = os.path.join(args.py150_files, "data") + "/"
    return filepath[len(prefix):]

def get_user_repo(filepath):
    components = get_dataset_relative_path(filepath).split('/')
    return components[0], components[1]

tokenizer = CuBertHugTokenizer(args.vocab)

# checks the given function to be suitable for inclusion and returns if so
def scan_fct(text, fct, filepath, file):
    num_tokens.append(fct.token_count)
    
    try:
        try:
            if args.task == 'wrong-binary-operator':
                extractor = WrongBinOpExtractor(text, tokenizer)
            elif args.task == 'var-misuse':
                extractor = VarMisuseExtractor(text, tokenizer)
            elif args.task == 'argument-swap':
                extractor = ArgSwapExtractor(text, tokenizer)
            else:
                assert(False)
            visit(text, extractor)
            if len(extractor.ops) == 0: return []
        except ParserSyntaxError:
            # skip files with syntax errors (e.g. Python 2 syntax)
            return []
        except DataConstructionException:
            return []

        user, repo = get_user_repo(filepath)
        dataset_rel_path = get_dataset_relative_path(filepath)

        return [{
            "function": text,
            "label": "Correct",
            "info": "dataset/ETHPy150Open {}/{}/{} {}.{}".format(user, repo, dataset_rel_path, file, fct.name),
        }]

    except (SyntaxError, UnicodeEncodeError, ValueError):
        pass
    except Exception as e:
        print("Failed to scan function {} in file {}: ".format(fct.name, filepath), e)
    return []

# scans a file for suitable functions
def scan_file(filepath):
    with open(filepath, "r") as fct_file:
        code = fct_file.read()
    
    try:
        analysis = lizard.analyze_file.analyze_source_code("test.py", code)
    
        file = filepath.split('/')[-1]
        res = []
        for fct in analysis.function_list:
            fct_src = get_func_src(code, fct.start_line, fct.end_line)
            res += scan_fct(fct_src, fct, filepath, file)
        return res
    except UnicodeEncodeError:
        return []

# collect set of paths included in ethpy150open
manifest_files = [os.path.join(args.ethpy150open_files, m) for m in ["train__manifest.json", "dev__manifest.json", "eval__manifest.json"]]
included_files = set()

for m in manifest_files:
    with open(m, "r") as f:
        eth150open_repos = json.load(f)
        included_files = included_files.union(set(list([file["filepath"] for file in eth150open_repos])))

num_tokens = []

included_files = sorted(list(included_files))

def process_file_parallel(filepath, *a):
    try:
        # skip non Python files
        if not filepath.endswith(".py"): return []
        # skip non-existing files
        filepath = os.path.join(args.py150_files, "data", filepath)
        if not os.path.exists(filepath):
            print("Included file not found: {}".format(filepath))
            return []
        # otherwise, scan the file
        return scan_file(filepath)
    except UnicodeEncodeError as e:
        print("unicode error with {}".format(filepath.encode('utf-8')))
        return []

# first scan and collect all files in parallel
pool = Pool()
pbar = tqdm(pool.imap_unordered(process_file_parallel, included_files, chunksize=16), total=len(included_files))
samples = []

try:
    for i, scan_result in enumerate(pbar):
        for sample in scan_result: 
            samples.append(sample)
        pbar.set_description("{} samples collected.".format(len(samples)))
        
        # early exit if in --subset-only mode
        if len(samples) > 1000 and args.subset_only:
            pool.terminate()
            break
except KeyboardInterrupt:
    pool.terminate()

pool.close()
pool.join()

num_original_samples = len(samples)

# filter near-duplicates
if not args.skip_near_duplicate_removal:
    print("Removing near-duplicates from resulting data...")
    filtered_samples = []
    for sample in tqdm(deduplicate(samples, "function", key=lambda i,s: s["info"]), desc="Removing near-duplicates"):
        filtered_samples.append(sample)
    samples = filtered_samples
    print("{}/{} after removing near-duplicates".format(len(samples), num_original_samples))
else:
    print("Skipping near-duplicate removal (--skip-near-duplicate-removal")


# get hash of each function
def get_hash_parallel(s):
    return hashlib.sha1(s['function'].encode("utf-8")).hexdigest(), s["info"]
def collect_hashes_in_set(code_objs, hash_and_info):
    h,info = hash_and_info
    code_objs[h] = info
    return code_objs

code_objs = map_reduce(samples, get_hash_parallel, dict(), collect_hashes_in_set, desc="Hashing function code...")

# filter duplicate functions
def get_filtered_result(s):
    h = hashlib.sha1(s['function'].encode("utf-8")).hexdigest()
    if code_objs[h] == s["info"]: 
        return [s]
    return []
def collect_non_duplicate_results(res_samples, to_add):
    res_samples += to_add
    return res_samples

exact_filtered_samples = map_reduce(samples, get_filtered_result, list(), collect_non_duplicate_results, desc="Filtering exact duplicates...")

print("{}/{} after filtering exact matches ({} exact duplicates)".format(len(exact_filtered_samples), num_original_samples, len(samples) - len(exact_filtered_samples)))
samples = exact_filtered_samples

# write collected and cleaned data to disk
out_file = os.path.join(args.out_dir, args.task, 'gen.jsontxt')
with open(out_file, "w") as f:
    for s in samples:
        json.dump(s, f, sort_keys=True)
        f.write("\n")