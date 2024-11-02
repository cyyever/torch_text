from typing import Any

from cyy_huggingface_toolbox import HuggingFaceModelEvaluator

from ..tokenizer import Tokenizer


class HuggingFaceTextModelEvaluator(HuggingFaceModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer: Tokenizer = kwargs.pop("tokenizer", None)
