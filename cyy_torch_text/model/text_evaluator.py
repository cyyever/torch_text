from typing import Any

from cyy_torch_toolbox import ModelEvaluator

from ..tokenizer import Tokenizer


class TextModelEvaluatorMixin:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer

    def get_feature_forward_fun(self) -> str:
        return "forward_input_feature"

    def split_batch_input(self, inputs: Any, **kwargs: Any) -> dict:
        return self.tokenizer.split_batch_input(inputs, **kwargs)


class TextModelEvaluator(ModelEvaluator, TextModelEvaluatorMixin):
    def __init__(self, **kwargs: Any) -> None:
        TextModelEvaluatorMixin.__init__(self, kwargs.pop("tokenizer"))
        ModelEvaluator.__init__(self, **kwargs)
