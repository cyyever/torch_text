import functools

import transformers
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import DatasetCollection, DatasetType
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.model import (create_model,
                                     global_model_evaluator_factory,
                                     global_model_factory)
from cyy_torch_toolbox.model.repositary import get_model_info

from ..tokenizer import get_tokenizer
from .huggingface_evaluator import HuggingFaceModelEvaluator
from .huggingface_model import get_huggingface_model_info
from .text_evaluator import TextModelEvaluator


def get_model_evaluator(model, **kwargs):
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    return TextModelEvaluator(model=model, **kwargs)


global_model_evaluator_factory.register(DatasetType.Text, get_model_evaluator)
global_model_evaluator_factory.register(DatasetType.CodeText, get_model_evaluator)


def get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
    if "hugging_face" in kwargs.get("name", ""):
        tokenizer_kwargs["type"] = "hugging_face"
        tokenizer_kwargs["name"] = model_constructor_info["name"]
    tokenizer = get_tokenizer(dataset_collection, tokenizer_kwargs)
    get_logger().debug("tokenizer is %s", tokenizer)

    if tokenizer is not None and hasattr(tokenizer, "itos"):
        token_num = len(tokenizer.get_vocab())
        for k in ("num_embeddings", "token_num"):
            if k not in kwargs:
                final_model_kwargs[k] = token_num
    input_max_len = dataset_collection.dataset_kwargs.get("input_max_len", None)
    if input_max_len is not None:
        final_model_kwargs["max_len"] = input_max_len
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)

    res = {"model": model}
    if tokenizer is not None:
        res |= {"tokenizer": tokenizer}
        word_vector_name = kwargs.get("word_vector_name", None)
        if word_vector_name is not None:
            from .word_vector import PretrainedWordVector

            PretrainedWordVector(word_vector_name).load_to_model(
                model=model,
                tokenizer=tokenizer,
            )
    return res


model_constructors = (
    get_model_info().get(DatasetType.Text, {}) | get_huggingface_model_info()
)
for name, model_constructor_info in model_constructors.items():
    for dataset_type in (DatasetType.Text, DatasetType.CodeText):
        if dataset_type not in global_model_factory:
            global_model_factory[dataset_type] = Factory()
        global_model_factory[dataset_type].register(
            name, functools.partial(get_model, model_constructor_info)
        )
