import torch
import torch.utils
from cyy_torch_toolbox.dataset.util import DatasetUtil


class TextDatasetUtil(DatasetUtil):
    @torch.no_grad()
    def get_sample_text(self, index: int) -> str:
        return self._get_sample_input(index, apply_transform=False)
