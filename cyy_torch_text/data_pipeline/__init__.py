import cyy_huggingface_toolbox.data_pipeline
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.data_pipeline import global_data_transform_factory

from ..model.text_evaluator import TextModelEvaluator
from .transform import add_text_transforms


global_data_transform_factory.register(DatasetType.Text, [add_text_transforms])
