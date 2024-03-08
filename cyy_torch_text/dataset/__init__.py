from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.repository import register_dataset_factory
from cyy_torch_toolbox.dataset.util import global_dataset_util_factor

from .collection import TextDatasetCollection  # noqa: F401
from .repository import HunggingFaceFactory
from .util import TextDatasetUtil

global_dataset_util_factor.register(DatasetType.Text, TextDatasetUtil)
register_dataset_factory(DatasetType.Text, HunggingFaceFactory())
register_dataset_factory(DatasetType.CodeText, HunggingFaceFactory())
