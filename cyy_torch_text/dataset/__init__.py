from cyy_torch_toolbox.dataset.util import global_dataset_util_factor
from cyy_torch_toolbox.ml_type import DatasetType

from .repository import register_constructors
from .util import TextDatasetUtil

global_dataset_util_factor.register(DatasetType.Text, TextDatasetUtil)
register_constructors()
