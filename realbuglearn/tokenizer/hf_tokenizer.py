# Thanks to DNGRos for this huggingface/transformers compatible version
# https://github.com/google-research/google-research/issues/582
#
import os
from re import sub
from tensorflow.python.ops.gen_math_ops import add
import torch
import collections
from typing import *
from transformers import BertTokenizer, BertTokenizerFast
from .cubert_tokenizer import CuBertTokenizer
from .python_tokenizer import PythonTokenizer
from .. import config
from tensor2tensor.data_generators import text_encoder
from transformers import BatchEncoding

def combine_tokenizer_with_subword(
    initial_tokenizer: CuBertTokenizer,
    subword_tokenizer: text_encoder.SubwordTextEncoder,
) -> Callable[[str], List[str]]:
    # Try to match the functionality at 
    # https://github.com/google-research/google-research/blob/50c6cd94b5/cubert/code_to_subtokenized_sentences.py#L111-L118
    
    def tokenize(string: str) -> List[str]:
        toks = initial_tokenizer.tokenize(string)
        tokens = flatten_list(
            subword_tokenizer.decode_list(
                subword_tokenizer.encode_without_tokenizing(token)
            )
            for token in toks
        )
        return tokens
    return tokenize


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class CuBertHugTokenizer(BertTokenizer):
    """
    A hacky solution that extends the cuBERT tokenizer
    to the BertTokenizer from transformers.

    Args:
        BertTokenizer ([type]): The BertTokenizer class from transformers

    Returns:
        [CuBertHugTokenizer]: A transformers compatible cuBERT tokenizer
    """    
    def __init__(
        self,
        vocab_file: str,
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token="[UNK]_",
            sep_token="[SEP]_",
            pad_token="<pad>_",
            cls_token="[CLS]_",
            mask_token="[MASK]_",
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    vocab_file)
            )
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.first_tokenizer = PythonTokenizer()
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(str(vocab_file))
        self._combined_func = combine_tokenizer_with_subword(
            self.first_tokenizer, self.subword_tokenizer)

    def __call__(self, text, add_special_tokens=True, with_pos=True):
        assert(isinstance(text, str))

        tokens, char_starts, char_ends = self.tokenize_with_pos(text)
        input_ids = self.convert_tokens_to_ids(tokens)

        num_added_tokens = 0
        if add_special_tokens:
            num_added_tokens += 2

        seq_len = config.MAX_SEQ_LEN - num_added_tokens
        if len(input_ids) > seq_len:
            input_ids, char_starts, char_ends = input_ids[:seq_len], char_starts[:seq_len], char_ends[:seq_len]
        token_type_ids, attention_masks = [0] * len(input_ids), [1] * len(input_ids)

        if add_special_tokens:
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
            char_starts = [-1] + char_starts + [-1]
            char_ends = [-1] + char_ends + [-1]
            token_type_ids = [0] + token_type_ids + [0]
            attention_masks = [1] + attention_masks + [1]

        if len(input_ids) < config.MAX_SEQ_LEN:
            diff = config.MAX_SEQ_LEN - len(input_ids)
            input_ids, char_starts, char_ends = input_ids + [0] * diff, char_starts + [-1] * diff, char_ends + [-1] * diff
            token_type_ids, attention_masks = token_type_ids + [0] * diff, attention_masks + [0] * diff

        res = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_masks,
        }
        if with_pos:
            res['char_starts'] = char_starts
            res['char_ends'] = char_ends

        return res

    @property
    def do_lower_case(self):
        return False

    def _tokenize(self, text):
        return self._combined_func(text)

    def tokenize_with_pos(self, text):
        tokens, starts, ends = [], [], []
        multi_tokens = self.first_tokenizer.tokenize_multi_tokens(text)
        for multi_token in multi_tokens:
            for token in multi_token.spellings:
                subwords = self.subword_tokenizer.decode_list(self.subword_tokenizer.encode_without_tokenizing(token))
                for subword in subwords:
                    tokens.append(subword)
                    starts.append(multi_token.metadata.start.char)
                    ends.append(multi_token.metadata.end.char)
        return tokens, starts, ends

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        return self.subword_tokenizer._subtoken_string_to_id[token]

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
