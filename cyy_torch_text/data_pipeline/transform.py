import functools
from collections.abc import Sequence

import torch
from cyy_naive_lib.log import log_info
from cyy_preprocessing_pipeline import BatchTransform, Transform
from cyy_torch_toolbox import (
    DatasetCollection,
    DatasetType,
)

from ..model.text_evaluator import TextModelEvaluator
from ..tokenizer.spacy import SpacyTokenizer


def truncate(input_seq: Sequence, max_seq_len: int) -> Sequence:
    """Truncate input sequence or batch

    :param input: Input sequence or batch to be truncated
    :param max_seq_len: Maximum length beyond which input is discarded
    :return: Truncated sequence
    """
    return input_seq[:max_seq_len]


def apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: TextModelEvaluator,
) -> None:
    max_len = dc.dataset_kwargs.get("input_max_len", None)
    if max_len is not None:
        log_info("use input text max_len %s", max_len)
    match model_evaluator.tokenizer:
        case SpacyTokenizer():
            dc.append_named_transform(
                Transform(
                    fun=model_evaluator.tokenizer, component="input", cacheable=True
                )
            )
            if max_len is not None:
                dc.append_named_transform(
                    Transform(
                        fun=functools.partial(truncate, max_seq_len=max_len),
                        component="input",
                        cacheable=True,
                    )
                )
            dc.append_named_transform(
                Transform(
                    fun=torch.LongTensor,
                    component="input",
                    cacheable=True,
                )
            )
            dc.append_named_transform(
                BatchTransform(
                    fun=functools.partial(
                        torch.nn.utils.rnn.pad_sequence,
                        padding_value=model_evaluator.tokenizer.get_token_id("<pad>"),
                    ),
                    component="input",
                )
            )


def add_text_transforms(
    dc: DatasetCollection, model_evaluator: TextModelEvaluator
) -> None:
    assert dc.dataset_type in (DatasetType.Text, DatasetType.CodeText)
    apply_tokenizer_transforms(dc=dc, model_evaluator=model_evaluator)
