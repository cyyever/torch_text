from typing import TypeAlias

import torch
import transformers
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Executor

from .spacy import SpacyTokenizer

TokenizerType: TypeAlias = SpacyTokenizer | transformers.PreTrainedTokenizerBase
TokenContainerType: TypeAlias = transformers.BatchEncoding | torch.Tensor


def get_mask_token(tokenizer: TokenizerType) -> str:
    match tokenizer:
        case SpacyTokenizer():
            return "<mask>"
        case transformers.PreTrainedTokenizerBase():
            return tokenizer.mask_token
        case _:
            raise NotImplementedError(type(tokenizer))


def extract_token_indices(
    token_container: TokenContainerType,
    tokenizer: TokenizerType | None = None,
    strip_special_token: bool = True,
) -> tuple:
    match token_container:
        case transformers.BatchEncoding():
            assert isinstance(token_container, transformers.BatchEncoding)
            input_ids: torch.Tensor = token_container["input_ids"].squeeze()
            if tokenizer is not None:
                input_ids = input_ids[input_ids != tokenizer.pad_token_id]
            if strip_special_token and tokenizer is not None:
                if input_ids[0] == tokenizer.cls_token_id:
                    input_ids = input_ids[1:]
                if input_ids[-1] == tokenizer.sep_token_id:
                    input_ids = input_ids[:-1]
            return tuple(input_ids.tolist())
        case torch.Tensor():
            return tuple(token_container.tolist())
        case _:
            raise NotImplementedError(type(tokenizer))


def convert_to_token_indices(
    executor: Executor,
    phrase: str,
    tokenizer: TokenizerType | None = None,
    strip_special_token: bool = True,
) -> tuple:
    if tokenizer is None:
        tokenizer = executor.model_evaluator.tokenizer
    transforms = executor.dataset_collection.get_transforms(phase=executor.phase)
    token_container = transforms.transform_input(
        transforms.transform_text(phrase), apply_random=False
    )
    token_indices = extract_token_indices(
        token_container, tokenizer=tokenizer, strip_special_token=strip_special_token
    )
    match tokenizer:
        case SpacyTokenizer():
            assert "".join(
                tokenizer.itos[idx] for idx in token_indices
            ) == phrase.replace(" ", "")
        case transformers.PreTrainedTokenizerBase():
            decoded_phrase = tokenizer.decode(token_indices)
            fdsds
            if decoded_phrase.replace(" ", "") != phrase.replace(" ", ""):
                get_logger().error("failed to recover phrase")
                get_logger().error("phrase is: %s", phrase)
                get_logger().error("decoded phrase is: %s", decoded_phrase)
                raise RuntimeError("failed to recover phrase")
        case _:
            raise NotImplementedError()
    return token_indices
