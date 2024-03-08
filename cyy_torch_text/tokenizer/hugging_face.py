import torch
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
        return self.__tokenizer.convert_tokens_to_ids(self.token)
        # return self.__tokenizer.convert_tokens_to_
        # encoding = self.__tokenizer(phrase, return_tensors="pt", truncation=False)
        # input_ids: torch.Tensor = encoding["input_ids"].squeeze()
        # input_ids = input_ids[input_ids != self.__tokenizer.pad_token_id]
        # if input_ids[0] == tokenizer.cls_token_id:
        #     input_ids = input_ids[1:]
        # if input_ids[-1] == tokenizer.sep_token_id:
        #     input_ids = input_ids[:-1]
