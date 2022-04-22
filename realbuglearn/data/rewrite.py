import abc
from itertools import combinations
from termcolor import colored
from typing import Sequence
from dataclasses import fields
import libcst as cst
from libcst import CSTNode, BaseMetadataProvider, Module
from libcst.metadata import ParentNodeProvider, ScopeProvider, Scope, CodeSpan
from libcst.metadata.scope_provider import ComprehensionScope, FunctionScope, GlobalScope, QualifiedNameSource
from libcst.metadata.span_provider import SpanProvidingCodegenState

from .. import config

class DataConstructionException(Exception):
    pass

class NoMaskSetException(DataConstructionException):
    pass

class NoOpFoundException(DataConstructionException):
    pass

# adapted from https://github.com/Instagram/LibCST/blob/main/libcst/_nodes/deep_equals.py
def is_sequence(a):
    return isinstance(a, Sequence) and not isinstance(a, (str, bytes))

def diff(a: object, b: object):
    if isinstance(a, cst.CSTNode) and isinstance(b, CSTNode):
        return diff_node(a, b)
    elif is_sequence(a) and is_sequence(b):
        return diff_sequence(a, b)
    elif type(a) == type(b):
        return a != b, None, None
    else:
        return a != b, a, b

EXCLUDED = [cst.SimpleWhitespace, cst.Newline, cst.Comment, cst.TrailingWhitespace, cst.ParenthesizedWhitespace]

def diff_node(a, b):
    if type(a) in EXCLUDED or type(b) in EXCLUDED:
        return False, None, None
    if type(a) is not type(b):
        return True, a, b

    count_diff, a_res, b_res = 0, None, None
    for field in (f for f in fields(a) if f.compare is True):
        a_value = getattr(a, field.name)
        b_value = getattr(b, field.name)
        diff_field, a_res_tmp, b_res_tmp = diff(a_value, b_value)
        if diff_field:
            # print(type(a), field.name, type(a_value), type(b_value), type(a_res_tmp), type(b_res_tmp))
            count_diff += 1
            a_res, b_res = a_res_tmp, b_res_tmp
    
    # if count_diff != 0:
    #     print(count_diff, type(a_res), type(b_res))
    if count_diff == 0:
        return False, None, None
    elif count_diff == 1:
        if a_res is None and b_res is None:
            return True, a, b
        else:
            return True, a_res, b_res
    else:
        return True, a, b

def diff_sequence(a, b):
    if len(a) != len(b):
        return True, None, None

    count_diff, a_res, b_res = 0, None, None
    for a_elm, b_elm in zip(a, b):
        diff_elm, a_res_tmp, b_res_tmp = diff(a_elm, b_elm)
        if diff_elm:
            count_diff += 1
            a_res, b_res = a_res_tmp, b_res_tmp

    if count_diff == 0:
        return False, None, None
    elif count_diff == 1:
        return True, a_res, b_res
    else:
        return True, None, None

def visit(code, visitor):
    tree = cst.parse_module(code)
    wrapper = cst.MetadataWrapper(tree)
    wrapper.visit(visitor)

def flatten(d):
    res = list()
    for key in sorted(d):
        if isinstance(d[key], list):
            res += d[key]
        elif isinstance(d[key], set):
            res += list(sorted(d[key]))
        else:
            assert(False)
    return res

def intersect(start1, end1, start2, end2):
    return end1 > start2 and end2 > start1

def ranges_to_mask(tokenizer_d, sorted_ranges, code):
    index, mask, any_mask_elem_set = 0, [0] * config.MAX_SEQ_LEN, False
    # assuming ranges is sorted
    for r in sorted_ranges:
        while index < config.MAX_SEQ_LEN:
            r_start, r_end = r.start, r.end
            char_start, char_end = tokenizer_d['char_starts'][index], tokenizer_d['char_ends'][index]
            if intersect(r_start, r_end, char_start, char_end):
                break
            index += 1
        assert(code[r_start:r_end].startswith(code[char_start:char_end]))
        if index < config.MAX_SEQ_LEN:
            any_mask_elem_set = True
            mask[index] = 1
    if not any_mask_elem_set: raise NoMaskSetException
    return mask

def visualize_varmisuse(code, loc_range, tgt_ranges):
    loc_color, tgt_color = 'green', 'yellow'
    ranges = flatten(tgt_ranges)
    ranges.append(loc_range)
    ranges = list(sorted(ranges))
    res, offset = '', 0
    for r in ranges:
        start, end = r.start - offset, r.end - offset
        res += code[:start]
        color = tgt_color
        if r == loc_range:
            color = loc_color
        res += colored(code[start:end], color)
        code = code[end:]
        offset += end
    res += code
    print(res)

class StrSpanPositionProvider(BaseMetadataProvider[CodeSpan]):
    def _gen_impl(self, module: Module) -> None:
        state = SpanProvidingCodegenState(
            default_indent=module.default_indent,
            default_newline=module.default_newline,
            provider=self,
            get_length=len,
        )
        module._codegen(state)

def index_to_coord(s, index):
    line, col = 1, 0
    for i, c in enumerate(s):
        if i == index: break
        col += 1
        if c == '\n':
            line += 1
            col = 0
    return line, col

class CodeRange:

    def __init__(self, code_span):
        self.start = code_span.start
        self.end = code_span.start + code_span.length

    def within(self, code):
        assert(len(code) >= self.end)
        return code[self.start:self.end]

    def before(self, code):
        assert(len(code) >= self.start)
        return code[:self.start] 

    def after(self, code):
        assert(len(code) >= self.end)
        return code[self.end:]

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __lt__(self, other):
        return self.start < other.start

    def __str__(self):
        return f'CodeRange({self.start} {self.end})'

    def __len__(self):
        return self.end - self.start

    def to_libcst_code_range(self, code):
        start_line, start_column = index_to_coord(code, self.start)
        end_line, end_column = index_to_coord(code, self.end)
        return (start_line, start_column), (end_line, end_column)

class RewriteOp:

    def __init__(self, loc_range, tgt_code):
        self.loc_range = loc_range
        self.tgt_code = tgt_code

    @abc.abstractclassmethod
    def tensorize(self, code, tokenizer):
        raise NotImplementedError

    def __str__(self):
        return f'RewriteOP({self.loc_range}, {self.tgt_code})'

    @abc.abstractclassmethod
    def to_json(self, code):
        raise NotImplementedError

class VarMisuseOp(RewriteOp):

    def __init__(self, loc_range, tgt_code, loc_code, candidates):
        super().__init__(loc_range, tgt_code)
        self.loc_code = loc_code
        self.candidates = candidates

    def tensorize(self, code, tokenizer):
        assert(self.loc_range.within(code) == self.loc_code)
        new_code = self.loc_range.before(code) + self.tgt_code + self.loc_range.after(code)
        extractor = VarMisuseExtractor(new_code, tokenizer)
        visit(new_code, extractor)
        loc_range = CodeRange(CodeSpan(self.loc_range.start, len(self.tgt_code)))
        tgt_op = extractor.get_op(loc_range=loc_range, tgt_code=self.loc_code)
        if tgt_op is None: raise NoOpFoundException
        d = extractor.tensorize(tokenizer)
        loc_correct_mask = ranges_to_mask(d, [loc_range], new_code)
        tgt_ranges = extractor.ops[loc_range][0].candidates[self.loc_code]
        tgt_correct_mask = ranges_to_mask(d, list(sorted(tgt_ranges)), new_code)
        d['cls_labels'] = 1
        d['loc_correct_masks'] = loc_correct_mask
        d['tgt_correct_masks'] = tgt_correct_mask
        d['code'] = new_code
        d['op'] = tgt_op.to_json(new_code)
        return d

    def __str__(self):
        return f'VarMisuseOp({self.loc_range}, \'{self.loc_code}\'->\'{self.tgt_code}\')'

    def to_json(self, code):
        d = dict()
        d['type'] = 'var-misuse'
        (loc_start_line, loc_start_col), (loc_end_line, loc_end_col) = self.loc_range.to_libcst_code_range(code)
        d['loc_start'] = [loc_start_line, loc_start_col]
        d['loc_end'] = [loc_end_line, loc_end_col]
        d['tgt_code'] = self.tgt_code
        return d

class WrongBinOpOp(RewriteOp):

    def __init__(self, loc_range, tgt_code, loc_op):
        super().__init__(loc_range, tgt_code)
        self.loc_op = loc_op

    def tensorize(self, code, tokenizer):
        new_code = self.loc_range.before(code) + self.tgt_code + self.loc_range.after(code)
        extractor = WrongBinOpExtractor(new_code, tokenizer)
        visit(new_code, extractor)
        start = self.loc_range.start + (1 if self.tgt_code[0].isspace() else 0)
        loc_range = CodeRange(CodeSpan(start, len(self.tgt_code.strip())))
        tgt_op = extractor.get_op(loc_range=loc_range, tgt_code=WrongBinOpExtractor.BinOp_To_Str[self.loc_op])
        if tgt_op is None: raise NoOpFoundException
        d = extractor.tensorize(tokenizer)
        loc_correct_mask = ranges_to_mask(d, [loc_range], new_code)
        tgt_correct_mask = [0] * len(WrongBinOpExtractor.BinOp_To_Str)
        if self.loc_op in WrongBinOpExtractor.Arithmetics:
            index = WrongBinOpExtractor.Arithmetics.index(self.loc_op)
        elif self.loc_op in WrongBinOpExtractor.Comparisons:
            index = len(WrongBinOpExtractor.Arithmetics) + WrongBinOpExtractor.Comparisons.index(self.loc_op)
        elif self.loc_op in WrongBinOpExtractor.Booleans:
            index = len(WrongBinOpExtractor.Arithmetics) + len(WrongBinOpExtractor.Comparisons) + WrongBinOpExtractor.Booleans.index(self.loc_op)
        tgt_correct_mask[index] = 1
        d['cls_labels'] = 1
        d['loc_correct_masks'] = loc_correct_mask
        d['tgt_correct_masks'] = tgt_correct_mask
        d['code'] = new_code
        d['op'] = tgt_op.to_json(new_code)
        return d

    def __str__(self):
        return f'WrongBinOp({self.loc_range}, \'{WrongBinOpExtractor.BinOp_To_Str[self.loc_op]}\'->\'{self.tgt_code}\')'

    def to_json(self, code):
        d = dict()
        d['type'] = 'wrong-binary-operator'
        (loc_start_line, loc_start_col), (loc_end_line, loc_end_col) = self.loc_range.to_libcst_code_range(code)
        d['loc_start'] = [loc_start_line, loc_start_col]
        d['loc_end'] = [loc_end_line, loc_end_col]
        d['tgt_code'] = self.tgt_code
        return d

class ArgSwapOp(RewriteOp):

    def __init__(self, func_range, arg1_range, arg1_index, arg2_range, arg2_index, candidates):
        super().__init__(func_range, '')
        self.arg1_range = arg1_range
        self.arg1_index = arg1_index
        self.arg2_range = arg2_range
        self.arg2_index = arg2_index
        self.candidates = candidates

    def tensorize(self, code, tokenizer):
        new_code = ''
        new_code += self.arg1_range.before(code)
        new_code += self.arg2_range.within(code)
        new_code += code[self.arg1_range.end:self.arg2_range.start]
        new_code += self.arg1_range.within(code)
        new_code += self.arg2_range.after(code)

        new_arg1_range = CodeRange(CodeSpan(self.arg1_range.start, len(self.arg2_range)))
        shift = len(self.arg2_range) - len(self.arg1_range)
        new_arg2_range = CodeRange(CodeSpan(self.arg2_range.start+shift, len(self.arg1_range)))

        extractor = ArgSwapExtractor(new_code, tokenizer)
        visit(new_code, extractor)
        tgt_op = extractor.get_op(arg1_range=new_arg1_range, arg2_range=new_arg2_range)
        if tgt_op is None: raise NoOpFoundException

        d = extractor.tensorize(tokenizer)
        loc_correct_mask = ranges_to_mask(d, [new_arg1_range, new_arg2_range], new_code)
        tgt_correct_mask = ranges_to_mask(d, [new_arg1_range, new_arg2_range], new_code)
        d['cls_labels'] = 1
        d['loc_correct_masks'] = loc_correct_mask
        d['tgt_correct_masks'] = tgt_correct_mask
        d['code'] = new_code
        d['op'] = tgt_op.to_json(new_code)
        return d

    def __str__(self):
        return f'ArgSwapOp({self.loc_range}, \'{self.arg1_index}<->{self.arg2_index}\')'

    def to_json(self, code):
        d = dict()
        d['type'] = 'argument-swap'
        (loc_start_line, loc_start_col), (loc_end_line, loc_end_col) = self.loc_range.to_libcst_code_range(code)
        d['loc_start'] = [loc_start_line, loc_start_col]
        d['loc_end'] = [loc_end_line, loc_end_col]
        
        d['arg1_index'] = self.arg1_index
        d['arg2_index'] = self.arg2_index        
        d['arg1_range'] = [self.arg1_range.start, self.arg1_range.end]
        d['arg2_range'] = [self.arg2_range.start, self.arg2_range.end]

        return d

def get_char_end(code, tokenizer):
    d = tokenizer(code)
    res = -1
    for char_end in d['char_ends']:
        res = max(res, char_end)
    return res

class Extractor(cst.CSTVisitor):

    def __init__(self, code, tokenizer):
        self.code = code
        self.max_char_end = get_char_end(code, tokenizer)
        self.ops = dict()

    def add_op(self, op):
        if op.loc_range not in self.ops:
            self.ops[op.loc_range] = list()
        self.ops[op.loc_range].append(op)

    def flatten(self):
        return flatten(self.ops)

    @abc.abstractclassmethod
    def tensorize(self, tokenizer, with_label=False):
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_op(self, **kwargs):
        raise NotImplementedError

# adapted from https://github.com/microsoft/neurips21-self-supervised-bug-detection-and-repair/blob/main/buglab/rewriting/rewriteops.py
class VarMisuseExtractor(Extractor):
    METADATA_DEPENDENCIES = (StrSpanPositionProvider, ParentNodeProvider, ScopeProvider)
    # BLACKLISTED_NAMES = frozenset({"self"})
    BLACKLISTED_NAMES = frozenset({})

    def __init__(self, code, tokenizer):
        super().__init__(code, tokenizer)
        self.enabled = True

    def is_local_symbol(self, scope: Scope, name: str):
        res = any(qn.source == QualifiedNameSource.LOCAL for qn in scope.get_qualified_names_for(name))
        if res: return res
        if isinstance(scope, ComprehensionScope): return self.is_local_symbol(scope.parent, name)
        return False

    def all_possible_var_misuses(self, scope: Scope, node: cst.Name):
        all_names = dict()
        for name in scope._assignments:
            if name in self.BLACKLISTED_NAMES: continue
            if name == node.value: continue
            if not name.isidentifier(): continue
            if not self.has_assign_before_node(scope, node, name): continue
            if not self.is_local_symbol(scope, name): continue
            all_names[name] = set()
            for assign in scope._assignments[name]:
                node_range = CodeRange(self.get_metadata(StrSpanPositionProvider, assign.node))
                if node_range.end > self.max_char_end: continue
                all_names[name].add(node_range)

        if isinstance(scope, ComprehensionScope):
            parent_all_names = self.all_possible_var_misuses(scope.parent, node)
            for name in parent_all_names:
                if name not in all_names:
                    all_names[name] = set()
                all_names[name] |= parent_all_names[name]
        return all_names

    def get_parent_stmt(self, node: cst.CSTNode):
        while not isinstance(node, cst.BaseStatement):
            node = self.get_metadata(ParentNodeProvider, node)
        return node

    def has_assign_before_node(self, scope, node, name):
        stmt = self.get_parent_stmt(node)
        node_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node))
        stmt_range = CodeRange(self.get_metadata(StrSpanPositionProvider, stmt))
        for assign in scope._assignments[name]:
            assign_stmt = self.get_parent_stmt(assign.node)
            assign_node_range = CodeRange(self.get_metadata(StrSpanPositionProvider, assign.node))
            assign_stmt_range = CodeRange(self.get_metadata(StrSpanPositionProvider, assign_stmt))
            if node_range == assign_node_range: continue
            if stmt_range == assign_node_range: continue
            if node_range == assign_stmt_range: continue
            if assign_stmt_range.start <= stmt_range.start: return True

        if isinstance(scope, ComprehensionScope):
            return self.has_assign_before_node(scope.parent, node, name)

        return False

    def visit_Name(self, node: cst.Name):
        if not self.enabled: return
        if node.value in self.BLACKLISTED_NAMES: return
        if not node.value.isidentifier(): return

        try:
            scope: Scope = self.get_metadata(ScopeProvider, node)
        except:
            return

        if not isinstance(scope, FunctionScope) and not isinstance(scope, ComprehensionScope): return
        if not self.is_local_symbol(scope, node.value): return
        is_access = node in scope.accesses
        if not is_access and len(scope.accesses[node]) > 0: return
        if any(isinstance(a.scope, GlobalScope) for a in scope[node.value]): return
        if not self.has_assign_before_node(scope, node, node.value): return
        node_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node))
        if node_range.end > self.max_char_end: return

        candidate_misuses_d = self.all_possible_var_misuses(scope, node)
        for name in candidate_misuses_d:
            replace_op = VarMisuseOp(node_range, name, node.value, candidate_misuses_d)
            self.add_op(replace_op)

    def visit_Parameters(self, node: cst.Parameters):
        self.enabled = False

    def leave_Parameters(self, original_node: cst.Parameters):
        self.enabled = True

    def tensorize_tgt_candidate_mask(self, op, tokenzier_d):
        mask = ranges_to_mask(tokenzier_d, list(sorted(flatten(op.candidates))), self.code)
        return mask

    def tensorize(self, tokenizer, with_label=False):
        d = tokenizer(self.code)
        sorted_loc_ranges = list(sorted(self.ops.keys()))
        loc_candidate_mask = ranges_to_mask(d, sorted_loc_ranges, self.code)
        tgt_candidate_mask = dict()
        j = 0
        for i in range(len(loc_candidate_mask)):
            if loc_candidate_mask[i] == 0: continue
            loc_range = sorted_loc_ranges[j]
            mask = ranges_to_mask(d, list(sorted(flatten(self.ops[loc_range][0].candidates))), self.code)
            tgt_candidate_mask[i] = mask
            j += 1
        d['loc_candidate_masks'] = loc_candidate_mask
        d['tgt_candidate_masks'] = tgt_candidate_mask
        if with_label:
            d['cls_labels'] = 0
            d['loc_correct_masks'] = [0] * config.MAX_SEQ_LEN
            d['tgt_correct_masks'] = [0] * config.MAX_SEQ_LEN
            d['code'] = self.code
            d['op'] = 'NO_BUG'
        return d

    def get_op(self, **kwargs):
        loc_range = kwargs['loc_range']
        tgt_code = kwargs['tgt_code']

        if loc_range not in self.ops: return None
        for op in self.ops[loc_range]:
            if op.tgt_code == tgt_code:
                return op
        else:
            return None

def get_op_pad(loc_op, loc_range, tgt_op, code):
    pad_left, pad_right = '' , ''
    if loc_op not in WrongBinOpExtractor.Comparisons_space and tgt_op in WrongBinOpExtractor.Comparisons_space:
        if not code[loc_range.start-1].isspace():
            pad_left = ' '
        if not code[loc_range.end].isspace():
            pad_right = ' '
    return pad_left, pad_right

class WrongBinOpExtractor(Extractor):
    METADATA_DEPENDENCIES = (StrSpanPositionProvider,)

    Arithmetics = [cst.Add, cst.Multiply, cst.Subtract, cst.Divide, cst.Modulo]
    Comparisons = [cst.Equal, cst.NotEqual, cst.Is, cst.IsNot, cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual, cst.In, cst.NotIn]
    Comparisons_space = [cst.Is, cst.IsNot, cst.In, cst.NotIn]
    Booleans = [cst.And, cst.Or]

    BinOp_To_Str = {
        cst.Add: '+', cst.Multiply: '*', cst.Subtract: '-', cst.Divide: '/', cst.Modulo: '%',
        cst.Equal: '==', cst.NotEqual: '!=', cst.Is: 'is', cst.IsNot: 'is not', cst.LessThan: '<', cst.LessThanEqual: '<=', cst.GreaterThan: '>', cst.GreaterThanEqual: '>=', cst.In: 'in', cst.NotIn: 'not in',
        cst.And: 'and', cst.Or: 'or'
    }

    def visit_BinaryOperation(self, node: cst.BinaryOperation):
        if type(node.operator) not in self.Arithmetics: return
        op_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node.operator))
        if op_range.end > self.max_char_end: return
        for op in self.Arithmetics:
            if isinstance(node.operator, op): continue
            assert(op_range.within(self.code) == self.BinOp_To_Str[type(node.operator)])
            replace_op = WrongBinOpOp(op_range, self.BinOp_To_Str[op], type(node.operator))
            self.add_op(replace_op)

    def visit_ComparisonTarget(self, node: cst.ComparisonTarget):
        op_type = type(node.operator)
        if op_type not in self.Comparisons: return
        op_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node.operator))
        if op_range.end > self.max_char_end: return

        for op in self.Comparisons:
            if isinstance(node.operator, op): continue
            pad_left, pad_right = get_op_pad(op_type, op_range, op, self.code)
            replace_op = WrongBinOpOp(op_range, pad_left+self.BinOp_To_Str[op]+pad_right, op_type)
            self.add_op(replace_op)

    def visit_BooleanOperation(self, node: cst.BooleanOperation):
        if type(node.operator) not in self.Booleans: return
        op_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node.operator))
        if op_range.end > self.max_char_end: return

        for op in self.Booleans:
            if isinstance(node.operator, op): continue
            assert(op_range.within(self.code) == self.BinOp_To_Str[type(node.operator)])
            replace_op = WrongBinOpOp(op_range, self.BinOp_To_Str[op], type(node.operator))
            self.add_op(replace_op)

    def tensorize_tgt_candidate_mask(self, op, tokenzier_d):
        mask = [0] * len(self.BinOp_To_Str)
        if op.loc_op in self.Arithmetics:
            for i in range(5):
                mask[i] = 1
            mask[self.Arithmetics.index(op.loc_op)] = 0
        elif op.loc_op in self.Comparisons:
            for i in range(5, 15):
                mask[i] = 1
            mask[self.Comparisons.index(op.loc_op)+5] = 0
        elif op.loc_op in self.Booleans:
            mask[15] = 1
            mask[16] = 1
            mask[self.Booleans.index(op.loc_op)+15] = 0
        else:
            assert(False)

        return mask

    def tensorize(self, tokenizer, with_label=False):
        d = tokenizer(self.code)
        sorted_loc_ranges = list(sorted(self.ops.keys()))
        loc_candidate_mask = ranges_to_mask(d, sorted_loc_ranges, self.code)
        tgt_candidate_mask = dict()
        j = 0
        for i in range(len(loc_candidate_mask)):
            if loc_candidate_mask[i] == 0: continue
            loc_range = sorted_loc_ranges[j]
            tgt_candidate_mask[i] = self.tensorize_tgt_candidate_mask(self.ops[loc_range][0], d)
            j += 1
        d['loc_candidate_masks'] = loc_candidate_mask
        d['tgt_candidate_masks'] = tgt_candidate_mask
        if with_label:
            d['cls_labels'] = 0
            d['loc_correct_masks'] = [0] * config.MAX_SEQ_LEN
            d['tgt_correct_masks'] = [0] * len(self.BinOp_To_Str)
            d['code'] = self.code
            d['op'] = 'NO_BUG'
        return d

    def get_op(self, **kwargs):
        loc_range = kwargs['loc_range']
        tgt_code = kwargs['tgt_code']

        if loc_range not in self.ops: return None
        for op in self.ops[loc_range]:
            if op.tgt_code == tgt_code:
                return op
        else:
            return None

class ArgSwapExtractor(Extractor):
    METADATA_DEPENDENCIES = (StrSpanPositionProvider,)

    @staticmethod
    def get_candidate_args(node):
        return [i for i, arg in enumerate(node.args) if arg.keyword is None and arg.star == '']

    def visit_Call(self, node: cst.Call):
        candidate_args = ArgSwapExtractor.get_candidate_args(node)
        if len(candidate_args) <= 1:
            return

        func_range = CodeRange(self.get_metadata(StrSpanPositionProvider, node.func))
        arg_ranges = [CodeRange(self.get_metadata(StrSpanPositionProvider, node.args[i])) for i in candidate_args]
        # TODO: we still end up with ops in self.ops (most of which with ranges > MAX_SEQ_LEN),
        # which cannot be resolved by get_op (raising NoOpFoundException)
        for i, j in combinations(candidate_args, 2):
            if arg_ranges[j].end > self.max_char_end: continue
            self.add_op(ArgSwapOp(func_range, arg_ranges[i], i, arg_ranges[j], j, arg_ranges))

    def tensorize(self, tokenizer, with_label=False):
        d = tokenizer(self.code)
        loc_ranges = set()
        loc_range_to_candidates = dict()
        for ops in self.ops.values():
            for op in ops:
                loc_ranges.add(op.arg1_range)
                loc_ranges.add(op.arg2_range)
                loc_range_to_candidates[op.arg1_range] = op.candidates
                loc_range_to_candidates[op.arg2_range] = op.candidates
        sorted_loc_ranges = list(sorted(loc_ranges))
        loc_candidate_mask = ranges_to_mask(d, sorted_loc_ranges, self.code)
        tgt_candidate_mask = dict()
        j = 0
        for i in range(len(loc_candidate_mask)):
            if loc_candidate_mask[i] == 0: continue
            loc_range = sorted_loc_ranges[j]
            candidates = list(loc_range_to_candidates[loc_range])
            candidates.remove(loc_range)
            mask = ranges_to_mask(d, candidates, self.code)
            tgt_candidate_mask[i] = mask
            j += 1
        d['loc_candidate_masks'] = loc_candidate_mask
        d['tgt_candidate_masks'] = tgt_candidate_mask
        if with_label:
            d['cls_labels'] = 0
            d['loc_correct_masks'] = [0] * config.MAX_SEQ_LEN
            d['tgt_correct_masks'] = [0] * config.MAX_SEQ_LEN
            d['code'] = self.code
            d['op'] = 'NO_BUG'
        return d

    def get_op(self, **kwargs):
        arg1_range = kwargs['arg1_range']
        arg2_range = kwargs['arg2_range']

        tgt_op = None
        for ops in self.ops.values():
            for op in ops:
                if op.arg1_range == arg1_range and op.arg2_range == arg2_range:
                    tgt_op = op
                    break
            if tgt_op is not None: break

        return tgt_op

if __name__ == '__main__':
    class Span:
        def __init__(self, start, length):
            self.start = start
            self.length = length

    # code_range = CodeRange(Span(55, 12))
    # code = "@w.event\ndef on_mouse_press(*args): w.has_exit = True\n\nglClearColor(1, 1, 1, 1)\nwhile not w.has_exit:\n    glClear(GL_COLOR_BUFFER_BIT)\n    w.dispatch_events()\n    w.flip()"
    # print(code_range.within(code))
    # print(code_range.to_libcst_code_range(code))

    code_range = CodeRange(Span(38, 4))
    code = "def build_url(self, path):\n    return self"
    print(code_range.within(code))
    print(code_range.to_libcst_code_range(code))
