from collections import Counter

from cyy_torch_toolbox import (
    DatasetCollection,
    DatasetUtil,
    MachineLearningPhase,
    Tokenizer,
)


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
        assert isinstance(util, DatasetUtil)
        for index in range(len(util)):
            transformed_token_results = util._get_sample_input(index)
            tokens = tokenizer.get_tokens_from_transformed_result(
                transformed_token_results
            )
            counter.update(tokens)
    return counter
