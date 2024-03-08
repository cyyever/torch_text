from typing import Iterable, TypeAlias

TokenIDType: TypeAlias = int | tuple[int] | list[int]


class Tokenizer:
    def get_mask_token(self) -> str:
        raise NotImplementedError()

    def tokenize(self, phrase: str) -> list[str]:
        raise NotImplementedError()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()

    def get_token(self, token_id: TokenIDType) -> str:
        raise NotImplementedError()

    def get_phrase(self, token_ids: Iterable[TokenIDType]) -> str:
        return " ".join(self.get_token(token_id) for token_id in token_ids)
