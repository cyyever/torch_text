import functools
from typing import Any

from cyy_torch_toolbox.dataset import DatasetFactory
from datasets import load_dataset as load_hugging_face_dataset
from datasets import load_dataset_builder


class HunggingFaceFactory(DatasetFactory):
    def get(self, key: str, case_sensitive: bool = True) -> Any:
        assert case_sensitive
        try:
            load_dataset_builder(path=key)
        except BaseException:
            return None
        return functools.partial(load_hugging_face_dataset, path=key)

    def get_similar_keys(self, key: str) -> list[str]:
        return []
