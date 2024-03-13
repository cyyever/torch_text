from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Executor

from .base import TokenIDsType, Tokenizer


def convert_phrase_to_transformed_result(
    executor: Executor,
    phrase: str,
) -> TokenIDsType:
    transforms = executor.dataset_collection.get_transforms(phase=executor.phase)
    return transforms.transform_input(
        transforms.transform_text(phrase), apply_random=False
    )


def convert_phrase_to_token_ids(
    executor: Executor,
    phrase: str,
    strip_special_token: bool = True,
) -> TokenIDsType:
    tokenizer = executor.model_evaluator.tokenizer
    assert isinstance(tokenizer, Tokenizer)
    transformed_token_results = convert_phrase_to_transformed_result(
        executor=executor, phrase=phrase
    )
    token_ids = tokenizer.get_token_ids_from_transformed_result(
        transformed_token_results
    )
    if strip_special_token:
        token_ids = tokenizer.strip_special_tokens(token_ids)
        decoded_phrase = tokenizer.get_phrase(token_ids)
        if decoded_phrase.replace(" ", "") != phrase.replace(" ", ""):
            get_logger().error("failed to recover phrase")
            get_logger().error("phrase is: %s", phrase)
            get_logger().error("decoded phrase is: %s", decoded_phrase)
            raise RuntimeError("failed to recover phrase")
    return token_ids
