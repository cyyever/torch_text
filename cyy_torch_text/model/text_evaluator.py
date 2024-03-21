from typing import Any

import torch
from cyy_torch_toolbox import ModelEvaluator, ModelType

from ..tokenizer import Tokenizer


class TextModelEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer: Tokenizer = kwargs.pop("tokenizer", None)

    def get_feature_forward_fun(self) -> str:
        return "forward_input_feature"

    def get_underlying_model_type(self) -> ModelType:
        return self.model_type

    def get_input_embedding(self, inputs) -> torch.Tensor:
        return self.get_input_feature(inputs)

    def split_batch_input(
        self, inputs, targets: Any = None, batch_size: int | None = None
    ) -> dict:
        batch_dim: int = 0
        if isinstance(inputs, torch.Tensor):
            if batch_size is None and targets is not None:
                assert isinstance(targets, torch.Tensor)
                batch_size = targets.shape[0]
            assert batch_size is not None
            if (
                batch_dim == 0
                and inputs.shape[0] != batch_size
                and inputs.shape[1] == batch_size
            ):
                batch_dim = 1
            if batch_dim != 0:
                inputs = inputs.permute(batch_dim, 0)
        return {"inputs": inputs, "batch_dim": batch_dim}
