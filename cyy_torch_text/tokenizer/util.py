from typing import Callable

import torch
import transformers

from .spacy import SpacyTokenizer


def get_mask_token(tokenizer: Callable) -> str:
    match tokenizer:
        case SpacyTokenizer():
            return "<mask>"
        case transformers.PreTrainedTokenizerBase():
            return tokenizer.mask_token
        case _:
            raise NotImplementedError(type(tokenizer))


def extract_token_indices(
    sample_input: transformers.BatchEncoding | torch.Tensor,
    tokenizer: Callable,
    strip_special_token: bool = True,
) -> tuple:
    match tokenizer:
        case transformers.PreTrainedTokenizerBase():
            assert isinstance(sample_input, transformers.BatchEncoding)
            input_ids = sample_input["input_ids"].squeeze()
            input_ids = input_ids[input_ids != tokenizer.pad_token_id]
            res = []
            input_ids = input_ids.view(-1).tolist()
            if strip_special_token:
                if input_ids[0] == tokenizer.cls_token_id:
                    input_ids = input_ids[1:]
                if input_ids[-1] == tokenizer.sep_token_id:
                    input_ids = input_ids[:-1]
            res.append(tuple(input_ids))
            return tuple(res)
        case SpacyTokenizer():
            assert isinstance(sample_input, torch.Tensor)
            return tuple((word_idx,) for word_idx in sample_input.tolist())
        case _:
            raise NotImplementedError(type(tokenizer))
