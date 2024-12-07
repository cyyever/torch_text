import cyy_huggingface_toolbox.data_pipeline
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.data_pipeline import global_data_transform_factory

from ..model.text_evaluator import TextModelEvaluator
from .transform import add_text_transforms


def append_transforms_to_dc(dc, model_evaluator: TextModelEvaluator) -> None:
    add_text_transforms(dc=dc, model_evaluator=model_evaluator)


global_data_transform_factory.register(DatasetType.Text, [append_transforms_to_dc])
