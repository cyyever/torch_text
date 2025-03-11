import base64
import functools
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import spacy.cli
import spacy.language
import spacy.symbols
import spacy.util
import torch
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TokenIDsType, TokenIDType, TokenizerMixin
from cyy_torch_toolbox.tokenizer import collect_tokens


def vocab(
    ordered_dict: OrderedDict,
    min_freq: int = 1,
    specials: list[str] | None = None,
) -> tuple[list, dict, OrderedDict]:
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

    itos = tokens
    stoi = {token: idx for idx, token in enumerate(tokens)}
    return itos, stoi, ordered_dict


class SpacyTokenizer(TokenizerMixin):
    def __init__(
        self,
        dc,
        package_name: str = "en_core_web_sm",
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
        self.__package_name = package_name
        if not spacy.util.is_package(package_name):
            spacy.cli.download(package_name)
        self.__spacy: spacy.language.Language = spacy.load(package_name)

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
        self.__itos: list[str] = []
        self.__stoi: dict = {}
        self.__freq_dict: OrderedDict = OrderedDict()
        self.__default_index: int = -1

    @property
    def itos(self) -> list[str]:
        self.__collect_tokens()
        return self.__itos

    @property
    def stoi(self) -> dict[str, int]:
        self.__collect_tokens()
        return self.__stoi

    @property
    def freq_dict(self) -> OrderedDict:
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
    def tokenizer(self):
        return self.spacy_model.tokenizer

    def tokenize(self, phrase: str) -> list[str]:
        tokens = self.spacy_model.tokenizer(phrase)
        return [
            t.text
            for t in tokens
            if (self.__keep_punct or not t.is_punct)
            and (self.__keep_stop or not t.is_stop)
        ]

    def get_token_id(self, token: str) -> int:
        return self.stoi.get(token, self.__default_index)

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

    def get_token(self, token_id: TokenIDType) -> str:
        return self.itos[token_id]

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        return token_ids[token_ids != self.get_token_id("<pad>")]

    def __collect_tokens(self) -> None:
        if self.__itos:
            return

        for token in self.__special_tokens:
            self.__spacy.tokenizer.add_special_case(
                token,
                [{spacy.symbols.ORTH: token}],
            )
        # First sort by descending frequency, then lexicographically
        filename = base64.b64encode(
            f"spacy_tokens_{self.__package_name}_{self.__keep_punct}_{self.__keep_stop}_{self.__max_tokens}_{'_'.join(sorted(self.__special_tokens))}".encode()
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
        self.__itos, self.__stoi, self.__freq_dict = vocab(
            ordered_dict,
            min_freq=self.__min_freq,
            specials=list(self.__special_tokens),
        )

        self.__default_index = self.__stoi["<unk>"]
        log_info("vocab size is %s", len(self.__stoi))

    def split_batch_input(self, inputs: torch.Tensor, batch_size: int) -> dict:
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
