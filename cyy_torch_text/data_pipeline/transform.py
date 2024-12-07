import functools
from collections.abc import Sequence

import torch
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    DatasetCollection,
    DatasetType,
)
from cyy_torch_toolbox.data_pipeline.transform import Transform

from ..model.text_evaluator import TextModelEvaluator
from ..tokenizer.spacy import SpacyTokenizer


def truncate(input_seq: Sequence, max_seq_len: int) -> Sequence:
    """Truncate input sequence or batch

    :param input: Input sequence or batch to be truncated
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param max_seq_len: Maximum length beyond which input is discarded
    :type max_seq_len: int
    :return: Truncated sequence
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    return input_seq[:max_seq_len]


def apply_tokenizer_transforms(
    dc: DatasetCollection,
    model_evaluator: TextModelEvaluator,
) -> None:
    # Input && InputBatch
    max_len = dc.dataset_kwargs.get("input_max_len", None)
    if max_len is not None:
        log_info("use input text max_len %s", max_len)
    match model_evaluator.tokenizer:
        case SpacyTokenizer():
            dc.append_named_transform(
                Transform(fun=model_evaluator.tokenizer, component="input")
            )
            if max_len is not None:
                dc.append_named_transform(
                    Transform(
                        fun=functools.partial(truncate, max_seq_len=max_len),
                        component="input",
                    )
                )
            dc.append_named_transform(
                Transform(fun=torch.LongTensor, component="input")
            )
            dc.append_named_transform(
                Transform(
                    fun=functools.partial(
                        torch.nn.utils.rnn.pad_sequence,
                        padding_value=model_evaluator.tokenizer.get_token_id("<pad>"),
                    ),
                    for_batch=True,
                )
            )


def get_label_to_text_mapping(dataset_name: str) -> dict | None:
    match dataset_name.lower():
        case "multi_nli":
            return {0: "entailment", 1: "neutral", 2: "contradiction"}
        case "imdb":
            return {0: "negative", 1: "positive"}
    return None


def add_text_transforms(
    dc: DatasetCollection, model_evaluator: TextModelEvaluator
) -> None:
    assert dc.dataset_type in (DatasetType.Text, DatasetType.CodeText)
    apply_tokenizer_transforms(dc=dc, model_evaluator=model_evaluator)
