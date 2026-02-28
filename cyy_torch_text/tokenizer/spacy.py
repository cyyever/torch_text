import base64
import functools
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import spacy.language
import spacy.symbols
import torch
import transformers
from cyy_huggingface_toolbox import HuggingFaceTokenizerBase
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import DatasetCollection, TokenIDsType
from cyy_torch_toolbox.tokenizer import collect_tokens
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer as HFTokenizerBase
from tokenizers.models import WordLevel as HFWordLevel


def vocab(
    ordered_dict: OrderedDict[str, int],
    min_freq: int = 1,
    specials: list[str] | None = None,
) -> tuple[dict[str, int], OrderedDict[str, int]]:
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurrence frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.


    """
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    for token in specials:
        tokens.append(token)
    # Save room for special tokens
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    stoi = {token: idx for idx, token in enumerate(tokens)}
    return stoi, ordered_dict


class SpacyTokenizer(HuggingFaceTokenizerBase):
    def __init__(
        self,
        dc: DatasetCollection,
        model_name: str = "spacy/en_core_web_sm",
        special_tokens: None | Iterable[str] | set[str] = None,
        keep_punct: bool = True,
        keep_stop: bool = True,
        min_freq: int = 1,
        max_tokens: None | int = None,
    ) -> None:
        self.__dc = dc
        self.__keep_punct = keep_punct
        self.__keep_stop = keep_stop
        self.__min_freq = min_freq
        self.__model_name = model_name
        repo_id = model_name if "/" in model_name else f"spacy/{model_name}"
        model_path = snapshot_download(repo_id=repo_id)
        self.__spacy: spacy.language.Language = spacy.load(model_path)

        if special_tokens is None:
            self.__special_tokens = set()
        else:
            self.__special_tokens = set(special_tokens)
        for token in ("<pad>", "<unk>", "<mask>", "<cls>", "<sep>"):
            self.__special_tokens.add(token)
        if max_tokens is not None:
            assert len(self.__special_tokens) < max_tokens, (
                "len(special_tokens) >= max_tokens, so the vocab will be entirely special tokens."
            )
            max_tokens = max_tokens - len(self.__special_tokens)
        self.__max_tokens = max_tokens
        self.__freq_dict: OrderedDict[str, int] = OrderedDict()
        self.__hf_tokenizer: transformers.PreTrainedTokenizerFast | None = None
        self.__itos: list[str] | None = None

    @property
    def itos(self) -> list[str]:
        if self.__itos is None:
            self.__collect_tokens()
            v = self.tokenizer.get_vocab()
            self.__itos = [token for token, _ in sorted(v.items(), key=lambda x: x[1])]
        return self.__itos

    @property
    def stoi(self) -> dict[str, int]:
        self.__collect_tokens()
        return self.tokenizer.get_vocab()

    @property
    def freq_dict(self) -> OrderedDict[str, int]:
        self.__collect_tokens()
        return self.__freq_dict

    @property
    def spacy_model(self) -> spacy.language.Language:
        return self.__spacy

    def get_vocab(self) -> Mapping[str, int]:
        return self.stoi

    def __call__(self, phrase: str) -> list[int]:
        return [self.get_token_id(token) for token in self.tokenize(phrase)]

    def get_mask_token(self) -> str:
        return "<mask>"

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        self.__collect_tokens()
        assert self.__hf_tokenizer is not None
        return self.__hf_tokenizer

    def tokenize(self, phrase: str) -> list[str]:
        tokens = self.spacy_model.tokenizer(phrase)
        return [
            t.text
            for t in tokens
            if (self.__keep_punct or not t.is_punct)
            and (self.__keep_stop or not t.is_stop)
        ]

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        if isinstance(transformed_result[0], str):
            return torch.tensor(
                [self.get_token_id(s) for s in transformed_result], dtype=torch.int64
            )
        if isinstance(transformed_result[0], int):
            return torch.tensor(transformed_result, dtype=torch.int64)
        assert isinstance(transformed_result, torch.Tensor)
        return transformed_result

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        if isinstance(transformed_result, str):
            return self.tokenize(transformed_result)

        assert isinstance(transformed_result, torch.Tensor)
        return [
            self.get_token(token_id=token_id)
            for token_id in transformed_result.tolist()
        ]

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        return token_ids[token_ids != self.get_token_id("<pad>")]

    def __collect_tokens(self) -> None:
        if self.__hf_tokenizer is not None:
            return

        for token in self.__special_tokens:
            self.__spacy.tokenizer.add_special_case(
                token,
                [{spacy.symbols.ORTH: token}],
            )
        # First sort by descending frequency, then lexicographically
        filename = base64.b64encode(
            f"spacy_tokens_{self.__model_name}_{self.__keep_punct}_{self.__keep_stop}_{self.__max_tokens}_{'_'.join(sorted(self.__special_tokens))}".encode()
        ).decode()
        counter: Counter = self.__dc.get_cached_data(
            file=f"{filename}.pk",
            computation_fun=functools.partial(
                collect_tokens, tokenizer=self, dc=self.__dc
            ),
        )
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if self.__max_tokens is None:
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            ordered_dict = OrderedDict(sorted_by_freq_tuples[: self.__max_tokens])
        stoi, self.__freq_dict = vocab(
            ordered_dict,
            min_freq=self.__min_freq,
            specials=sorted(self.__special_tokens),
        )

        hf_base = HFTokenizerBase(HFWordLevel(vocab=stoi, unk_token="<unk>"))
        self.__hf_tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=hf_base,
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            cls_token="<cls>",
            sep_token="<sep>",
        )
        log_info("vocab size is %s", len(stoi))

    def split_batch_input(self, inputs: torch.Tensor, batch_size: int) -> dict[str, Any]:
        batch_dim: int = 0
        if isinstance(inputs, torch.Tensor):
            if (
                batch_dim == 0
                and inputs.shape[0] != batch_size
                and inputs.shape[1] == batch_size
            ):
                batch_dim = 1
            if batch_dim != 0:
                inputs = inputs.permute(batch_dim, 0)
        return {"inputs": inputs, "batch_dim": batch_dim}
