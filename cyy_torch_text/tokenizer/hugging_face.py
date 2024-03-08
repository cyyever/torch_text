from functools import cached_property

import transformers

from .base import TokenIDType, Tokenizer


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

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self.__tokenizer

    def get_mask_token(self) -> str:
        return self.__tokenizer.mask_token

    def tokenize(self, phrase: str) -> list[str]:
        encoding = self.__tokenizer(phrase, return_tensors="pt", truncation=False)
        return [
            token for token in encoding.tokens() if token not in self.special_tokens
        ]

    def get_token_id(self, token: str) -> TokenIDType:
        return self.__tokenizer.convert_tokens_to_ids(token)

    def get_token(self, token_id: TokenIDType) -> str:
        return self.__tokenizer.decode(token_id)
