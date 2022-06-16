import os
import json
import argparse
import pydriller
import libcst as cst
from libcst._exceptions import ParserSyntaxError

from realbuglearn.data.rewrite import CodeRange, DataConstructionException, StrSpanPositionProvider, WrongBinOpExtractor, VarMisuseExtractor, ArgSwapExtractor, get_op_pad, visit, diff
from realbuglearn.tokenizer.hf_tokenizer import CuBertHugTokenizer
from realbuglearn.data.cst_utils import get_func_src

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, dest='task', choices=['var-misuse', 'wrong-binary-operator', 'argument-swap'], required=True)
    parser.add_argument('--user', type=str, dest='user', required=True)
    parser.add_argument('--repo', type=str, dest='repo', required=True)
    parser.add_argument('--commit', type=str, dest='commit', default=None, required=False)
    parser.add_argument('--in_dir', type=str, dest='in_dir', default='../../../repos', required=False)
    parser.add_argument('--vocab', type=str, dest='vocab', default='../../../pretrained/pretrained-epoch-2-20210711/20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt', required=False)
    args = parser.parse_args()
    return args

args = get_args()

def exclude_path_json(user, repo, path):
    return {'user': user, 'repo': repo, 'path': path, 'info': 'path-to-exclude'}

def handle_wrong_bin_op(old_src, new_src, tokenizer, file):
    try:
        old_module, new_module = cst.parse_module(old_src), cst.parse_module(new_src)
    except ParserSyntaxError:
        return
    old_wrapper, new_wrapper = cst.MetadataWrapper(old_module), cst.MetadataWrapper(new_module)
    is_diff, old_res, new_res = diff(old_wrapper.module, new_wrapper.module)
    if not is_diff: return
    if not (type(old_res) in WrongBinOpExtractor.Arithmetics and type(new_res) in WrongBinOpExtractor.Arithmetics) and \
        not (type(old_res) in WrongBinOpExtractor.Comparisons and type(new_res) in WrongBinOpExtractor.Comparisons) and \
        not (type(old_res) in WrongBinOpExtractor.Booleans and type(new_res) in WrongBinOpExtractor.Booleans):
        return

    new_provider = new_wrapper.resolve(StrSpanPositionProvider)
    new_range = CodeRange(new_provider[new_res])
    new_extractor = WrongBinOpExtractor(new_src, tokenizer)
    visit(new_src, new_extractor)
    pad_left, pad_right = get_op_pad(type(new_res), new_range, type(old_res), new_src)
    tgt_code = pad_left + WrongBinOpExtractor.BinOp_To_Str[type(old_res)] + pad_right
    tgt_op = new_extractor.get_op(loc_range=new_range, tgt_code=tgt_code)

    if tgt_op is not None:
        try:
            pos_j = tgt_op.tensorize(new_src, tokenizer)
            pos_j['user'] = args.user
            pos_j['repo'] = args.repo
            pos_j['path'] = file.old_path

            neg_j = new_extractor.tensorize(tokenizer, with_label=True)
            neg_j['user'] = args.user
            neg_j['repo'] = args.repo
            neg_j['path'] = file.new_path

            print(json.dumps(pos_j))
            print(json.dumps(neg_j))
        except DataConstructionException:
            print(json.dumps(exclude_path_json(args.user, args.repo, file.old_path)))
            print(json.dumps(exclude_path_json(args.user, args.repo, file.new_path)))
    else:
        print(json.dumps(exclude_path_json(args.user, args.repo, file.old_path)))
        print(json.dumps(exclude_path_json(args.user, args.repo, file.new_path)))

def handle_var_misuse(old_src, new_src, tokenizer, file):
    try:
        old_module, new_module = cst.parse_module(old_src), cst.parse_module(new_src)
    except ParserSyntaxError:
        return
    old_wrapper, new_wrapper = cst.MetadataWrapper(old_module), cst.MetadataWrapper(new_module)
    is_diff, old_res, new_res = diff(old_wrapper.module, new_wrapper.module)
    if not is_diff: return
    if not isinstance(old_res, cst.Name) or not isinstance(new_res, cst.Name): return

    new_provider = new_wrapper.resolve(StrSpanPositionProvider)
    new_range = CodeRange(new_provider[new_res])
    new_extractor = VarMisuseExtractor(new_src, tokenizer)
    visit(new_src, new_extractor)
    tgt_code = old_res.value
    tgt_op = new_extractor.get_op(loc_range=new_range, tgt_code=tgt_code)
    if tgt_op is None: return

    try:
        pos_j = tgt_op.tensorize(new_src, tokenizer)
        pos_j['user'] = args.user
        pos_j['repo'] = args.repo
        pos_j['path'] = file.old_path

        neg_j = new_extractor.tensorize(tokenizer, with_label=True)
        neg_j['user'] = args.user
        neg_j['repo'] = args.repo
        neg_j['path'] = file.new_path

        print(json.dumps(pos_j))
        print(json.dumps(neg_j))
    except DataConstructionException:
        print(json.dumps(exclude_path_json(args.user, args.repo, file.old_path)))
        print(json.dumps(exclude_path_json(args.user, args.repo, file.new_path)))

def is_arg_swap(arg_diffs):
    if len(arg_diffs) != 2: return False
    d1, d2 = arg_diffs[0], arg_diffs[1]
    return not diff(d1[0], d2[1])[0] and not diff(d1[1], d2[0])[0]

def handle_arg_swap(old_src, new_src, tokenizer, file, func_def_changes):
    try:
        old_module, new_module = cst.parse_module(old_src), cst.parse_module(new_src)
    except ParserSyntaxError:
        return
    old_wrapper, new_wrapper = cst.MetadataWrapper(old_module), cst.MetadataWrapper(new_module)
    new_provider = new_wrapper.resolve(StrSpanPositionProvider)
    is_diff, old_res, new_res = diff(old_wrapper.module, new_wrapper.module)
    if not is_diff: return
    if not isinstance(old_res, cst.Call) or not isinstance(new_res, cst.Call): return
    if diff(old_res.func, new_res.func)[0]: return
    old_args, new_args = old_res.args, new_res.args
    if len(old_args) != len(new_args): return
    if len(old_args) <= 1: return
    arg_swap_func_name = CodeRange(new_provider[new_res.func]).within(new_src)
    for class_name, func_name in func_def_changes:
        if arg_swap_func_name == class_name:
            return
        if func_name in arg_swap_func_name or arg_swap_func_name in func_name:
            return

    arg_diffs = []
    for old_exp, new_exp in zip(old_args, new_args):
        is_diff, old_exp_res, new_exp_res = diff(old_exp, new_exp)
        if is_diff:
            arg_diffs.append([old_exp_res, new_exp_res])

    if not is_arg_swap(arg_diffs): 
        return

    arg1_range = CodeRange(new_provider[arg_diffs[0][1]])
    arg2_range = CodeRange(new_provider[arg_diffs[1][1]])
    new_extractor = ArgSwapExtractor(new_src, tokenizer)
    visit(new_src, new_extractor)
    tgt_op = new_extractor.get_op(arg1_range=arg1_range, arg2_range=arg2_range)

    if tgt_op is not None:
        try:
            pos_j = tgt_op.tensorize(new_src, tokenizer)
            pos_j['user'] = args.user
            pos_j['repo'] = args.repo
            pos_j['path'] = file.old_path

            neg_j = new_extractor.tensorize(tokenizer, with_label=True)
            neg_j['user'] = args.user
            neg_j['repo'] = args.repo
            neg_j['path'] = file.new_path

            # from genbug.data.cst_utils import side_by_side
            # from termcolor import colored
            # print(side_by_side([old_src, new_src[:arg1_range.start]+colored(new_src[arg1_range.start:arg1_range.end], 'green')+new_src[arg1_range.end:arg2_range.start]+colored(new_src[arg2_range.start:arg2_range.end], 'yellow')+new_src[arg2_range.end:]]))
            print(json.dumps(pos_j))
            print(json.dumps(neg_j))
        except DataConstructionException:
            print(json.dumps(exclude_path_json(args.user, args.repo, file.old_path)))
            print(json.dumps(exclude_path_json(args.user, args.repo, file.new_path)))
    else:
        print(json.dumps(exclude_path_json(args.user, args.repo, file.old_path)))
        print(json.dumps(exclude_path_json(args.user, args.repo, file.new_path)))

class FuncDefExtractor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, code, file_path, added_lines):
        self.code = code
        self.file_path = file_path
        self.added_lines = added_lines
        self.res = []
        self.class_stack = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name.value)

    def leave_ClassDef(self, node):
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        pos = self.get_metadata(cst.metadata.PositionProvider, node.params)
        for line in range(pos.start.line, pos.end.line+1):
            if line in self.added_lines:
                class_name = None if len(self.class_stack) == 0 else self.class_stack[-1]
                func_name = node.name.value
                if node.name.value in ('__call__', '__init__'):
                    func_name = class_name.lower()
                self.res.append((class_name, func_name))

def extract_func_def_changes(commit):
    func_def_changes = []
    for file in commit.modified_files:
        if not file.filename.endswith('.py'): continue
        try:
            added_lines = set(map(lambda t: t[0], file.diff_parsed['added']))
            code = file.source_code
            extrator = FuncDefExtractor(code, file.new_path, added_lines)
            tree = cst.parse_module(code)
            wrapper = cst.MetadataWrapper(tree)
            wrapper.visit(extrator)
            func_def_changes += extrator.res
        except:
            pass
    return func_def_changes

def extract_from_file(file, tokenizer, added, func_def_changes):
    if file.deleted_lines != 1: return
    if file.added_lines != 1: return
    if not file.filename.endswith('.py'): return
    if file.old_path != file.new_path: return

    deleted_line_no = file.diff_parsed['deleted'][0][0]
    added_line_no = file.diff_parsed['added'][0][0]
    old_methods = [m for m in file.methods_before if m.start_line <= deleted_line_no < m.end_line]
    new_methods = [m for m in file.methods if m.start_line <= added_line_no < m.end_line]
    if len(old_methods) != 1 or len(new_methods) != 1: return
    new_method, old_method = new_methods[0], old_methods[0]
    old_method_src = get_func_src(file.source_code_before, old_method.start_line, old_method.end_line)
    new_method_src = get_func_src(file.source_code, new_method.start_line, new_method.end_line)

    if (old_method_src, new_method_src) in added: return
    if (new_method_src, old_method_src) in added: return
    added.add((old_method_src, new_method_src))

    if args.task == 'wrong-binary-operator':
        handle_wrong_bin_op(old_method_src, new_method_src, tokenizer, file)
    elif args.task == 'var-misuse':
        handle_var_misuse(old_method_src, new_method_src, tokenizer, file)
    elif args.task == 'argument-swap':
        handle_arg_swap(old_method_src, new_method_src, tokenizer, file, func_def_changes)
    else:
        assert(False)

def main():
    tokenizer = CuBertHugTokenizer(args.vocab)
    repo_path = os.path.join(args.in_dir, args.user, args.repo)
    repo = pydriller.Repository(repo_path, include_refs=True)
    added = set()
    for commit in repo.traverse_commits():
        if args.commit is None or args.commit != commit.hash:
            continue
        if args.task == 'argument-swap':
            func_def_changes = extract_func_def_changes(commit)
        else:
            func_def_changes = None
        for file in commit.modified_files:
            extract_from_file(file, tokenizer, added, func_def_changes)

if __name__ == '__main__':
    main()