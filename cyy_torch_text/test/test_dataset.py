import cyy_torch_text  # noqa: F401
from cyy_torch_toolbox.dataset_collection import create_dataset_collection


def test_dataset() -> None:
    create_dataset_collection(
        "multi_nli",
        dataset_kwargs={"val_split": "validation_matched"},
    )


def test_dataset_labels() -> None:
    for name in ("imdb",):
        dc = create_dataset_collection(name)
        assert dc.get_labels()
