import functools

import huggingface_hub
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.repository import register_dataset_constructors
from datasets import load_dataset as load_hugging_face_dataset


@functools.cache
def get_hungging_face_datasets() -> dict:
    return {
        dataset.id: functools.partial(load_hugging_face_dataset, path=dataset.id)
        for dataset in huggingface_hub.list_datasets(full=False)
    }


def register_constructors() -> None:
    for name, constructor in get_hungging_face_datasets().items():
        register_dataset_constructors(DatasetType.Text, name, constructor)
        register_dataset_constructors(DatasetType.CodeText, name, constructor)
