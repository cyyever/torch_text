from typing import Callable

import torch
import transformers
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Executor
from cyy_torch_toolbox.tensor import tensor_to

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


def collect_token_indices(
    executor: Executor,
    phrase: str,
    tokenizer: Callable | None = None,
    strip_special_token: bool = True,
) -> torch.Tensor:
    if tokenizer is None:
        tokenizer = executor.model_evaluator.tokenizer
    match tokenizer:
        case SpacyTokenizer():
            transforms = executor.dataset_collection.get_transforms(
                phase=executor.phase
            )
            word_indices = transforms.transform_input(
                transforms.transform_text(phrase), apply_random=False
            )
            assert len(word_indices) == 1
            assert tokenizer.itos[word_indices[0]] == phrase
            return tensor_to(word_indices, device=executor.device, check_slowdown=False)
        case transformers.PreTrainedTokenizerBase():
            transforms = executor.dataset_collection.get_transforms(
                phase=executor.phase
            )
            sample_input = transforms.transform_inputs([phrase])
            input_ids = extract_token_indices(
                sample_input=sample_input,
                tokenizer=tokenizer,
                strip_special_token=strip_special_token,
            )[0]
            decoded_phrase = tokenizer.decode(input_ids)
            if decoded_phrase.replace(" ", "") != phrase.replace(" ", ""):
                get_logger().error("failed to recover phrase")
                get_logger().error("phrase is: %s", phrase)
                get_logger().error("decoded phrase is: %s", decoded_phrase)
                raise RuntimeError("failed to recover phrase")
            return torch.tensor(input_ids, device=executor.device)
        case _:
            raise NotImplementedError()
