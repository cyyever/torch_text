from typing import TypeAlias

import torch
import transformers
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Executor

from .spacy import SpacyTokenizer

TokenizerType: TypeAlias = SpacyTokenizer | transformers.PreTrainedTokenizerBase
TokenContainerType: TypeAlias = transformers.BatchEncoding | torch.Tensor
TokenIDType: TypeAlias = tuple[int] | tuple[tuple[int]]


def __extract_token_ids(
    token_container: TokenContainerType,
    tokenizer: TokenizerType,
    strip_special_token: bool,
) -> TokenIDType:
    match token_container:
        case transformers.BatchEncoding():
            assert isinstance(tokenizer, transformers.PreTrainedTokenizerBase)
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


def convert_token_ids_to_phrase(
    token_ids: TokenIDType, tokenizer: TokenizerType
) -> str:
    match tokenizer:
        case SpacyTokenizer():
            return " ".join(tokenizer.itos[token_id] for token_id in token_ids)
        case transformers.PreTrainedTokenizerBase():
            return tokenizer.decode(token_ids)
        case _:
            raise NotImplementedError()


def convert_phase_to_token_ids(
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
    token_ids = __extract_token_ids(
        token_container, tokenizer=tokenizer, strip_special_token=strip_special_token
    )
    decoded_phrase = convert_token_ids_to_phrase(
        token_ids=token_ids, tokenizer=tokenizer
    )
    if decoded_phrase.replace(" ", "") != phrase.replace(" ", ""):
        get_logger().error("failed to recover phrase")
        get_logger().error("phrase is: %s", phrase)
        get_logger().error("decoded phrase is: %s", decoded_phrase)
        raise RuntimeError("failed to recover phrase")
    return token_ids
