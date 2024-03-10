from functools import cached_property
from typing import Any, Mapping

import transformers

from .base import TokenIDsType, TokenIDType, Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer_config: dict) -> None:
        self.__tokenizer: transformers.PreTrainedTokenizerBase = (
            transformers.AutoTokenizer.from_pretrained(
                tokenizer_config["name"], **tokenizer_config.get("kwargs", {})
            )
        )

    @cached_property
    def special_tokens(self) -> set[str]:
        tokens = set()
        for attr in self.__tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            if attr != "additional_special_tokens" and hasattr(self.__tokenizer, attr):
                tokens.add(getattr(self.__tokenizer, attr))
        return tokens

    @cached_property
    def special_token_ids(self) -> set[TokenIDType]:
        return {self.get_token_id(token) for token in self.special_tokens}

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self.__tokenizer

    def get_vocab(self) -> Mapping[str, int]:
        return self.__tokenizer.get_vocab()

    def get_mask_token(self) -> str:
        return self.__tokenizer.mask_token

    def tokenize(self, phrase: str) -> list[str]:
        encoding = self.__tokenizer(phrase, return_tensors="pt", truncation=False)
        return [
            token for token in encoding.tokens() if token not in self.special_tokens
        ]

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        assert isinstance(transformed_result, transformers.BatchEncoding)
        input_ids: TokenIDsType = transformed_result["input_ids"].squeeze()
        return input_ids

    def get_token_id(self, token: str) -> TokenIDType:
        return self.__tokenizer.convert_tokens_to_ids(token)

    def get_token(self, token_id: TokenIDType) -> str:
        return self.__tokenizer.decode(token_id)

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        for special_token_id in self.special_token_ids:
            token_ids = token_ids[token_ids != special_token_id]
        return token_ids
