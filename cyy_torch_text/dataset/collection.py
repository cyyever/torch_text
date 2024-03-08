from typing import Any

from cyy_torch_toolbox import DatasetCollection, DatasetType


class TextDatasetCollection(DatasetCollection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.dataset_type == DatasetType.Text
