import functools
from collections.abc import Callable
from typing import Any

import cyy_huggingface_toolbox.model  # noqa: F401
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory
from cyy_torch_toolbox.model import (
    create_model,
    global_model_evaluator_factory,
    global_model_factory,
)
from cyy_torch_toolbox.model.repositary import get_model_info

from ..tokenizer import get_tokenizer
from .text_evaluator import TextModelEvaluator
from .word_vector import PretrainedWordVector

global_model_evaluator_factory.register(DatasetType.Text, [TextModelEvaluator])
global_model_evaluator_factory.register(DatasetType.CodeText, [TextModelEvaluator])


model_constructors = get_model_info().get(DatasetType.Text, {})


class TextModelFactory(Factory):
    def get(
        self, key: str, case_sensitive: bool = True, default: Any = None
    ) -> Callable | None:
        model_name = self._lower_key(key)
        if model_name in model_constructors:
            return functools.partial(
                self.__create_text_model,
                model_constructor_info=model_constructors[model_name],
            )

        return default

    def __create_text_model(
        self,
        model_constructor_info: dict,
        dataset_collection: DatasetCollection,
        **kwargs: Any,
    ) -> dict:
        final_model_kwargs: dict = kwargs
        tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
        tokenizer = get_tokenizer(dataset_collection, tokenizer_kwargs)
        log_debug("tokenizer is %s", tokenizer)

        if tokenizer is not None and hasattr(tokenizer, "itos"):
            token_num = len(tokenizer.get_vocab())
            for k in ("num_embeddings", "token_num"):
                if k not in kwargs:
                    final_model_kwargs[k] = token_num
        input_max_len = dataset_collection.dataset_kwargs.get("input_max_len", None)
        if input_max_len is not None:
            final_model_kwargs["max_len"] = input_max_len
        model = create_model(
            model_constructor_info["constructor"], **final_model_kwargs
        )

        res: dict[str, Any] = {"model": model}
        if tokenizer is not None:
            res |= {"tokenizer": tokenizer}
            word_vector_name = kwargs.get("word_vector_name")
            if word_vector_name is not None:
                PretrainedWordVector(word_vector_name).load_to_model(
                    model=model,
                    tokenizer=tokenizer,
                )
        return res


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    if dataset_type not in global_model_factory:
        global_model_factory[dataset_type] = []
    global_model_factory[dataset_type].append(TextModelFactory())
