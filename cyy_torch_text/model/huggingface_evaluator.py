from typing import Any, Callable

import torch
import transformers
from cyy_torch_toolbox import ModelType
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .text_evaluator import TextModelEvaluator


class HuggingFaceModelEvaluator(TextModelEvaluator):
    def __init__(self, model, **kwargs: Any) -> None:
        model_type = kwargs.get("model_type", None)
        if model_type is None:
            kwargs["model_type"] = HuggingFaceModelEvaluator._get_underlying_model_type(
                model
            )
        super().__init__(model=model, **kwargs)

    @classmethod
    def _get_underlying_model_type(cls, model) -> ModelType | None:
        if "ConditionalGeneration" in model.__class__.__name__:
            return ModelType.TextGeneration
        return None

    def split_batch_input(self, inputs, batch_size: int) -> dict:
        batch_dim = 0
        new_inputs = []
        if isinstance(inputs, transformers.BatchEncoding | dict):
            first_value = next(iter(inputs.values()))
            assert isinstance(first_value, torch.Tensor)
            for i in range(first_value.size(dim=0)):
                new_inputs.append({k: v[i].unsqueeze(dim=0) for k, v in inputs.items()})
            return {"inputs": new_inputs, "batch_dim": batch_dim}
        return {"inputs": inputs, "batch_dim": batch_dim}

    def get_input_feature(
        self, inputs: transformers.BatchEncoding
    ) -> transformers.BatchEncoding:
        assert isinstance(inputs, transformers.BatchEncoding | dict)
        if "inputs_embeds" not in inputs:
            input_ids = inputs["input_ids"]
            if hasattr(self.model, "distilbert"):
                if len(list(input_ids.shape)) == 1:
                    input_ids = input_ids.unsqueeze(dim=0)
                embeddings = self.model.distilbert.embeddings(input_ids).detach()
            elif hasattr(self.model, "bert"):
                embeddings = self.model.get_input_embeddings()(input_ids).detach()
            else:
                raise NotImplementedError(self.model)
            inputs["inputs_embeds"] = embeddings
        inputs.pop("input_ids", None)
        return inputs

    def get_input_embedding(self, inputs) -> torch.Tensor:
        return self.get_input_feature(inputs)["inputs_embeds"]

    def _create_input(
        self,
        inputs: dict,
        targets: Any,
        device: torch.device,
        **kwargs: Any,
    ) -> dict:
        if hasattr(targets, "input_ids"):
            targets = targets.input_ids
        inputs["labels"] = targets
        return inputs

    def get_feature_forward_fun(self) -> str:
        return "_forward_model"

    def _forward_model(self, *args: Any, **kwargs: Any) -> dict:
        model_input = self._create_input(*args, **kwargs)
        output = self.model(**model_input)
        # targets = kwargs["targets"]
        # if kwargs.get("reduce_loss", True):
        #     return {
        #         "model_input": model_input,
        #         "model_output": output,
        #         "logits": output.logits,
        #         "loss": output.loss,
        #         "is_averaged_loss": True,
        #         "loss_batch_size": targets.shape[0],
        #     }
        return self._compute_loss(*args, output=output.logits, **kwargs)

    def _choose_loss_function(self) -> Callable:
        match self.model.config.problem_type:
            case "regression":
                return MSELoss()
            case "single_label_classification":
                return CrossEntropyLoss()
            case "multi_label_classification":
                return BCEWithLogitsLoss()
        raise NotImplementedError(self.model.config.problem_type)
