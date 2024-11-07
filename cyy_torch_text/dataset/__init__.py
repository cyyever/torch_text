import cyy_huggingface_toolbox.dataset  # noqa: F401
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset.util import global_dataset_util_factor

from .collection import TextDatasetCollection  # noqa: F401
from .util import TextDatasetUtil

global_dataset_util_factor.register(DatasetType.Text, TextDatasetUtil)
