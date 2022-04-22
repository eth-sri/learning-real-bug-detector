from transformers.utils.dummy_pt_objects import ElectraForMaskedLM
from ..config import MAX_SEQ_LEN
import sys

def sparsify_list(l):
    if len(l) != MAX_SEQ_LEN: return l
    indices = []
    for i,v in enumerate(l):
        if v == 1: indices.append(i)
    return {"_sparse": indices}

def desparsify_list(l):
    if not type(l) is dict or not "_sparse" in l.keys(): return l
    dense_l = [0] * MAX_SEQ_LEN
    for index in l["_sparse"]:
        dense_l[index] = 1
    return dense_l

def remove_padding(value, padding):
    result = list()
    for v in value:
        if v != padding: result.append(v)
        else: break
    return result

NO_EFFECT_KEYS = {
    "user", "repo", "path", "cls_labels", "code", "op", "char_starts", "char_ends",
    "user1", "repo1", "path1", "cls_labels1", "code1", "op1", "char_starts1", "char_ends1",
    "user2", "repo2", "path2", "cls_labels2", "code2", "op2", "char_starts2", "char_ends2",
}

def sparsify_sample(o):
    """
    Applies different sparsification strategies to the different 
    features of a sample to save some memory.

    To undo sparsification use the function `desparsify_sample`.
    """

    result = dict()
    for k in o.keys(): 
        value = o[k]
        if k in ("input_ids", "input_ids1", "input_ids2"):
            result[k] = remove_padding(value, 0)
        elif k in ("token_type_ids", "token_type_ids1", "token_type_ids2"):
            result[k] = remove_padding(value, 0)
        elif k in ("attention_mask", "attention_mask1", "attention_mask2"):
            result[k] = sparsify_list(value)
        elif k in ("loc_candidate_masks", "loc_candidate_masks1", "loc_candidate_masks2"):
            result[k] = sparsify_list(value)
        elif k in ("loc_correct_masks", "loc_correct_masks1", "loc_correct_masks2"):
            result[k] = sparsify_list(value)
        elif k in ("tgt_correct_masks", "tgt_correct_masks1", "tgt_correct_masks2"):
            result[k] = sparsify_list(value)
        elif k in ("tgt_candidate_masks", "tgt_candidate_masks1", "tgt_candidate_masks2"):
            d = value
            res_d = dict()
            for key,value in d.items():
                res_d[key] = sparsify_list(value)
            result[k] = res_d
        elif k in NO_EFFECT_KEYS:
            result[k] = value
        else:
            print("warning: unhandled key during sparsification (sparse_sample.py) {}. No sparsification will be applied.".format(k))
            result[k] = value
    return result

def add_padding(value, padding):
    return value + [padding] * (MAX_SEQ_LEN - len(value))

def desparsify_sample(o):
    result = dict()
    for k in o.keys(): 
        value = o[k]
        if k in ("input_ids", "input_ids1", "input_ids2"):
            result[k] = add_padding(value, 0)
        elif k in ("token_type_ids", "token_type_ids1", "token_type_ids2"):
            result[k] = add_padding(value, 0)
        elif k in ("attention_mask", "attention_mask1", "attention_mask2"):
            result[k] = desparsify_list(value)
        elif k in ("loc_candidate_masks", "loc_candidate_masks1", "loc_candidate_masks2"):
            result[k] = desparsify_list(value)
        elif k in ("loc_correct_masks", "loc_correct_masks1", "loc_correct_masks2"):
            result[k] = desparsify_list(value)
        elif k in ("tgt_correct_masks", "tgt_correct_masks1", "tgt_correct_masks2"):
            result[k] = desparsify_list(value)
        elif k in ("tgt_candidate_masks", "tgt_candidate_masks1", "tgt_candidate_masks2"):
            d = value
            res_d = dict()
            for key,value in d.items():
                res_d[key] = desparsify_list(value)
            result[k] = res_d
        elif k in NO_EFFECT_KEYS:
            result[k] = value
        else:
            print("warning: unhandled key during sparsification (sparse_sample.py) {}. No sparsification will be applied.".format(k))
            result[k] = value
    return result
