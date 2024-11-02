from collections import Counter
from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from cyy_torch_toolbox import DatasetCollection, MachineLearningPhase

from ..dataset import TextDatasetUtil

TokenIDType: TypeAlias = int | tuple[int] | list[int] | torch.Tensor
TokenIDsType: TypeAlias = torch.Tensor


class Tokenizer:
    def get_vocab(self) -> Mapping[str, int]:
        raise NotImplementedError()

    def get_mask_token(self) -> str:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str]:
        raise NotImplementedError()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        raise NotImplementedError()

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        raise NotImplementedError()

    def get_token(self, token_id: TokenIDType) -> str:
        raise NotImplementedError()

    def get_phrase(self, token_ids: TokenIDsType) -> str:
        return " ".join(self.get_token(token_id) for token_id in token_ids)

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        raise NotImplementedError()


def collect_tokens(
    tokenizer: Tokenizer,
    dc: DatasetCollection,
    phase: MachineLearningPhase | None = None,
) -> Counter:
    counter: Counter = Counter()
    if phase is None:
        util_list = [
            dc.get_dataset_util(phase=phase)
            for phase in MachineLearningPhase
            if dc.has_dataset(phase)
        ]
    else:
        util_list = [dc.get_dataset_util(phase=phase)]
    for util in util_list:
        assert isinstance(util, TextDatasetUtil)
        for index in range(len(util)):
            transformed_token_results = util._get_sample_input(
                index, apply_transform=True
            )
            tokens = tokenizer.get_tokens_from_transformed_result(
                transformed_token_results
            )
            counter.update(tokens)
    return counter
