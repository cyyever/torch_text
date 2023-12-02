from cyy_torch_toolbox.data_transform import global_data_transform_factory
from cyy_torch_toolbox.ml_type import DatasetType

from .transform import add_text_extraction, add_text_transforms


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    if dc.dataset_type == DatasetType.Text:
        if model_evaluator is None:
            add_text_extraction(dc=dc)
        else:
            add_text_transforms(dc=dc, model_evaluator=model_evaluator)
        return


global_data_transform_factory.register(DatasetType.Text, append_transforms_to_dc)
