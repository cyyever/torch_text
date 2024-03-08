import transformers

from .base import TokenIDType, Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer_config: dict) -> None:
        self.__tokenizer: transformers.PreTrainedTokenizerBase = (
            transformers.AutoTokenizer.from_pretrained(
                tokenizer_config["name"], **tokenizer_config.get("kwargs", {})
            )
        )

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self.__tokenizer

    def get_mask_token(self) -> str:
        return self.__tokenizer.mask_token

    def tokenize(self, phrase: str) -> list[str]:
        encoding = self.__tokenizer(phrase, return_tensors="pt", truncation=False)
        return encoding.tokens()

    def get_token_id(self, token: str) -> TokenIDType:
        raise NotImplementedError()
