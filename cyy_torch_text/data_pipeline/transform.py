import functools
from typing import Sequence

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import (DatasetCollection, DatasetType,
                               MachineLearningPhase, ModelType, TransformType)
from cyy_torch_toolbox.data_pipeline.common import (backup_target,
                                                    int_target_to_text,
                                                    replace_str)

from ..model.text_evaluator import TextModelEvaluator
from ..tokenizer.hugging_face import HuggingFaceTokenizer
from ..tokenizer.spacy import SpacyTokenizer
from .template import get_text_template, interpret_template


def add_text_extraction(dc: DatasetCollection) -> None:
    assert dc.dataset_type == DatasetType.Text
    # ExtractData
    dc.append_transform(backup_target, key=TransformType.ExtractData)
    dataset_name: str = dc.name.lower()
    # InputText
    if dataset_name == "imdb":
        dc.append_transform(
            functools.partial(replace_str, old="<br />", new=""),
            key=TransformType.InputText,
        )


def squeeze_huggingface_input(huggingface_input) -> None:
    for k in ("input_ids", "attention_mask"):
        if k in huggingface_input:
            huggingface_input[k] = huggingface_input[k].squeeze(dim=0)
    return huggingface_input


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
    max_len: int | None,
    for_input: bool,
) -> None:
    if for_input:
        batch_key = TransformType.InputBatch
        key = TransformType.Input
    else:
        batch_key = TransformType.TargetBatch
        key = TransformType.Target
    match model_evaluator.tokenizer:
        case SpacyTokenizer():
            dc.append_transform(model_evaluator.tokenizer, key=key)
            if max_len is not None:
                dc.append_transform(
                    functools.partial(truncate, max_seq_len=max_len),
                    key=key,
                )
            dc.append_transform(torch.LongTensor, key=key)
            dc.append_transform(
                functools.partial(
                    torch.nn.utils.rnn.pad_sequence,
                    padding_value=model_evaluator.tokenizer.get_token_id("<pad>"),
                ),
                key=batch_key,
            )
        case HuggingFaceTokenizer():
            dc.append_transform(
                functools.partial(
                    model_evaluator.tokenizer.tokenizer,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                ),
                key=key,
            )
            dc.append_transform(squeeze_huggingface_input, key=key)
        case _:
            raise NotImplementedError(type(model_evaluator.tokenizer))


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
    dataset_name: str = dc.name.lower()
    # InputText
    assert model_evaluator.model_type is not None
    text_template = get_text_template(
        dataset_name=dataset_name, model_type=model_evaluator.model_type
    )
    if text_template is not None:
        dc.append_transform(
            functools.partial(interpret_template, template=text_template),
            key=TransformType.InputText,
        )

    # Input && InputBatch
    input_max_len = dc.dataset_kwargs.get("input_max_len", None)
    if input_max_len is not None:
        get_logger().info("use input text max_len %s", input_max_len)
    apply_tokenizer_transforms(
        dc=dc, model_evaluator=model_evaluator, max_len=input_max_len, for_input=True
    )

    # Target
    if (
        model_evaluator.model_type == ModelType.TextGeneration
        or model_evaluator.get_underlying_model_type() == ModelType.TextGeneration
    ):
        mapping = get_label_to_text_mapping(dataset_name)
        if mapping is not None:
            dc.append_transform(
                functools.partial(int_target_to_text, mapping=mapping),
                key=TransformType.Target,
            )
        elif isinstance(
            dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
                0
            ),
            int,
        ):
            dc.append_transform(int_target_to_text, key=TransformType.Target)
        max_len = dc.dataset_kwargs.get("output_max_len", None)
        get_logger().info("use output text max len %s", max_len)
        apply_tokenizer_transforms(
            dc=dc, model_evaluator=model_evaluator, max_len=max_len, for_input=False
        )
    # elif model_evaluator.model_type == ModelType.Classification:
    #     if isinstance(
    #         dc.get_dataset_util(phase=MachineLearningPhase.Training).get_sample_label(
    #             0
    #         ),
    #         str,
    #     ):
    #         label_names = dc.get_label_names()
    #         dc.append_transform(
    #             str_target_to_int(label_names), key=TransformType.Target
    #         )
