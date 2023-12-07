import copy
import functools

import transformers
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import DatasetCollection, DatasetType
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.model import (global_model_evaluator_factory,
                                     global_model_factory)
from cyy_torch_toolbox.model.repositary import get_model_info

from ..tokenizer import get_tokenizer
from .hugging_face_evaluator import HuggingFaceModelEvaluator
from .huggingface_model import get_hugging_face_model_info
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
    final_model_kwargs: dict = {}
    tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
    if "hugging_face" in kwargs.get("name", ""):
        tokenizer_kwargs["type"] = "hugging_face"
        tokenizer_kwargs["name"] = model_constructor_info["name"]
    tokenizer = get_tokenizer(dataset_collection, tokenizer_kwargs)
    get_logger().debug("tokenizer is %s", tokenizer)

    if tokenizer is not None and hasattr(tokenizer, "itos"):
        for k in ("num_embeddings", "token_num"):
            if k not in kwargs:
                final_model_kwargs[k] = len(tokenizer.itos)

    final_model_kwargs |= kwargs

    while True:
        try:
            model = model_constructor_info["constructor"](**final_model_kwargs)
            break
        except TypeError as e:
            retry = False
            for k in copy.copy(final_model_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    final_model_kwargs.pop(k)
                    retry = True
                    break
            if not retry:
                raise e

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
    get_model_info().get(DatasetType.Text, {}) | get_hugging_face_model_info()
)
for name, model_constructor_info in model_constructors.items():
    for dataset_type in (DatasetType.Text, DatasetType.CodeText):
        if dataset_type not in global_model_factory:
            global_model_factory[dataset_type] = Factory()
        global_model_factory[dataset_type].register(
            name, functools.partial(get_model, model_constructor_info)
        )
