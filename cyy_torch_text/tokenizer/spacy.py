import functools
from collections import Counter, OrderedDict
from typing import Any, Iterable, Mapping

import torch
from cyy_naive_lib.log import get_logger

import spacy
import spacy.symbols

from .base import TokenIDsType, TokenIDType, Tokenizer, collect_tokens


def vocab(
    ordered_dict: OrderedDict,
    min_freq: int = 1,
    specials: list[str] | None = None,
    special_first: bool = True,
) -> tuple[list, dict, OrderedDict]:
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurance frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.

    Returns:
        torchtext.vocab.Vocab: A `Vocab` object

    Examples:
        >>> from torchtext.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> print(v1['a']) #prints 1
        >>> print(v1['out of vocab']) #raise RuntimeError since default index is not set
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> #adding <unk> token and default index
        >>> unk_token = '<unk>'
        >>> default_index = -1
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
        >>> v2.set_default_index(default_index)
        >>> print(v2['<unk>']) #prints 0
        >>> print(v2['out of vocab']) #prints -1
        >>> #make default index same as index of unk_token
        >>> v2.set_default_index(v2[unk_token])
        >>> v2['out of vocab'] is v2[unk_token] #prints True
    """
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    # Save room for special tokens
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if special_first:
        tokens[0:0] = specials
    else:
        tokens.extend(specials)
    itos = tokens
    stoi = {token: idx for idx, token in enumerate(tokens)}
    return itos, stoi, ordered_dict


class SpacyTokenizer(Tokenizer):
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
        self.__keep_punct = keep_punct
        self.__keep_stop = keep_stop
        self.__spacy: spacy.language.Language = spacy.load(package_name)
        self.unusual_words: set = set()

        counter: Counter = dc.get_cached_data(
            file="tokenizer_word_counter.pk",
            computation_fun=functools.partial(collect_tokens, tokenizer=self, dc=dc),
        )

        if special_tokens is None:
            special_tokens = set()
        else:
            special_tokens = set(special_tokens)
        for token in ("<pad>", "<unk>", "<mask>", "<cls>", "<sep>"):
            special_tokens.add(token)
        for token in special_tokens:
            self.__spacy.tokenizer.add_special_case(
                token,
                [{spacy.symbols.ORTH: token}],
            )

        # First sort by descending frequency, then lexicographically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if max_tokens is None:
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            assert (
                len(special_tokens) < max_tokens
            ), "len(special_tokens) >= max_tokens, so the vocab will be entirely special tokens."
            ordered_dict = OrderedDict(
                sorted_by_freq_tuples[: max_tokens - len(special_tokens)]
            )
        self.__itos, self.__stoi, self.__freq_dict = vocab(
            ordered_dict,
            min_freq=min_freq,
            specials=list(special_tokens),
        )

        self.__default_index = self.__stoi["<unk>"]
        get_logger().info("vocab size is %s", len(self.__stoi))

    @property
    def itos(self) -> list:
        return self.__itos

    @property
    def stoi(self) -> dict[str, int]:
        return self.__stoi

    @property
    def freq_dict(self) -> OrderedDict:
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

    def tokenize(self, phrase: str) -> list[str]:
        tokens = self.__spacy.tokenizer(phrase)
        return [
            t.text
            for t in tokens
            if (self.__keep_punct or not t.is_punct)
            and (self.__keep_stop or not t.is_stop)
        ]

    def get_token_id(self, token: str) -> int:
        return self.__stoi.get(token, self.__default_index)

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        assert isinstance(transformed_result, torch.Tensor)
        return transformed_result

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        assert isinstance(transformed_result, torch.Tensor)
        return [
            self.get_token(token_id=token_id)
            for token_id in transformed_result.tolist()
        ]

    def get_token(self, token_id: TokenIDType) -> str:
        return self.itos[token_id]

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        return token_ids[token_ids != self.get_token_id("<pad>")]
