import functools
import os
from typing import Any

from cyy_torch_toolbox.dataset import DatasetFactory
from datasets import load_dataset as load_hugging_face_dataset
from datasets import load_dataset_builder


class HunggingFaceFactory(DatasetFactory):
    def get(
        self, key: str, case_sensitive: bool = True, cache_dir: str | None = None
    ) -> Any:
        assert case_sensitive
        assert cache_dir is not None
        if not self.__has_dataset(key, cache_dir):
            return None

        return functools.partial(self.__get_dataset, path=key, cache_dir=cache_dir)

    @classmethod
    def __get_dataset(cls, path: str, cache_dir: str, split: Any, **kwargs) -> Any:
        dataset = load_hugging_face_dataset(path=path, split=split, **kwargs)
        with open(cls.__dataset_id_file(cache_dir), "wt", encoding="utf8") as f:
            f.write("1")
        return dataset

    @classmethod
    def __dataset_id_file(cls, cache_dir: str) -> str:
        os.makedirs(os.path.join(cache_dir, ".cache"), exist_ok=True)
        return os.path.join(cache_dir, ".cache", "__from_hg")

    @classmethod
    def __has_dataset(cls, key: Any, cache_dir: str) -> bool:
        if os.path.isfile(cls.__dataset_id_file(cache_dir)):
            return True
        try:
            load_dataset_builder(path=key)
            return True
        except BaseException:
            pass
        return False

    def get_similar_keys(self, key: str) -> list[str]:
        return []
