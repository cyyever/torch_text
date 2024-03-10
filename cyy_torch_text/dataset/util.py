import torch
import torch.utils
from cyy_torch_toolbox import DatasetUtil


class TextDatasetUtil(DatasetUtil):
    @torch.no_grad()
    def get_sample_text(self, index: int, apply_transform: bool = False) -> str:
        sample_text = self._get_sample_input(index=index, apply_transform=False)
        if apply_transform:
            assert self._transforms is not None
            sample_text = self._transforms.transform_text(sample_text)
        return sample_text
