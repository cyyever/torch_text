from typing import TypeAlias

TokenIDType: TypeAlias = int | tuple[int] | list[int]


class Tokenizer:
    def get_mask_token(self) -> str:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str]:
        raise NotImplementedError()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()
