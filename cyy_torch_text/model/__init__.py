import functools
from typing import Any, Callable

import transformers
from cyy_huggingface_toolbox import get_huggingface_constructor
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import DatasetCollection, DatasetType, Factory
from cyy_torch_toolbox.model import (create_model,
                                     global_model_evaluator_factory,
                                     global_model_factory)
from cyy_torch_toolbox.model.repositary import get_model_info

from ..tokenizer import get_tokenizer
from .huggingface_evaluator import HuggingFaceTextModelEvaluator
from .text_evaluator import TextModelEvaluator
from .word_vector import PretrainedWordVector


def get_model_evaluator(
    model, **kwargs: Any
) -> TextModelEvaluator | HuggingFaceTextModelEvaluator:
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceTextModelEvaluator(model=model, **kwargs)
    return TextModelEvaluator(model=model, **kwargs)


global_model_evaluator_factory.register(DatasetType.Text, get_model_evaluator)
global_model_evaluator_factory.register(DatasetType.CodeText, get_model_evaluator)


model_constructors = get_model_info().get(DatasetType.Text, {})


class TextModelFactory(Factory):
    def __init__(self, parent_factory: None | Factory = None) -> None:
        super().__init__()
        self.__parent_factory = parent_factory

    def get(self, key: str, case_sensitive: bool = True) -> Callable | None:
        if self.__parent_factory is not None:
            res = self.__parent_factory.get(key=key, case_sensitive=case_sensitive)
            if res is not None:
                return res
        if not case_sensitive:
            key = self._lower_key(key)
        model_name = key
        if model_name in model_constructors:
            return functools.partial(
                self.create_text_model,
                model_constructor_info=model_constructors[model_name],
                is_hugging_face=False,
            )

        res = get_huggingface_constructor(model_name)
        if res is not None:
            constructor, name = res
            return functools.partial(
                self.create_text_model,
                model_constructor_info={"constructor": constructor, "name": name},
                is_hugging_face=True,
            )
        return None

    def create_text_model(
        self,
        model_constructor_info: dict,
        is_hugging_face: bool,
        dataset_collection: DatasetCollection,
        **kwargs: Any,
    ) -> dict:
        final_model_kwargs: dict = kwargs
        tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
        if is_hugging_face:
            tokenizer_kwargs["type"] = "hugging_face"
            tokenizer_kwargs["name"] = model_constructor_info["name"]
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

        res = {"model": model}
        if tokenizer is not None:
            res |= {"tokenizer": tokenizer}
            word_vector_name = kwargs.get("word_vector_name", None)
            if word_vector_name is not None:
                PretrainedWordVector(word_vector_name).load_to_model(
                    model=model,
                    tokenizer=tokenizer,
                )
        return res


for dataset_type in (DatasetType.Text, DatasetType.CodeText):
    global_model_factory[dataset_type] = TextModelFactory(
        parent_factory=global_model_factory.get(dataset_type, None)
    )
