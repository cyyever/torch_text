import transformers
from cyy_torch_toolbox.ml_type import DatasetType
from cyy_torch_toolbox.model_evaluator import global_model_evaluator_factory

from .hugging_face import HuggingFaceModelEvaluator
from .text import TextModelEvaluator


def get_model_evaluator(model, **kwargs):
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingFaceModelEvaluator(model=model, **kwargs)
    return TextModelEvaluator(model=model, **kwargs)


global_model_evaluator_factory.register(DatasetType.Text, get_model_evaluator)
