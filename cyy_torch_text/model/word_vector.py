import os
import pickle

import torch
from cyy_naive_lib.log import log_debug, log_info
from cyy_naive_lib.source_code.package_spec import PackageSpecification
from cyy_naive_lib.source_code.tarball_source import TarballSource
from cyy_torch_toolbox import ModelUtil
from torch import nn

from ..tokenizer.spacy import SpacyTokenizer


class PretrainedWordVector:
    __word_vector_root_dir: str = os.path.join(
        os.path.expanduser("~"), "pytorch_word_vector"
    )
    __word_vector_cache: dict[str, dict] = {}

    def __init__(self, name: str) -> None:
        self.__name = name

    @property
    def word_vector_dict(self):
        if self.__name not in PretrainedWordVector.__word_vector_cache:
            PretrainedWordVector.__word_vector_cache[self.__name] = self.__download()
        return PretrainedWordVector.__word_vector_cache[self.__name]

    def load_to_model(self, model, tokenizer) -> None:
        assert isinstance(tokenizer, SpacyTokenizer)
        itos = tokenizer.itos

        def __load_embedding(name, module, module_util) -> None:
            unknown_tokens = set()
            embeddings = module.weight.tolist()
            for idx, token in enumerate(itos):
                word_vector = self.word_vector_dict.get(token, None)
                if word_vector is None:
                    word_vector = self.word_vector_dict.get(token.lower(), None)
                if word_vector is not None:
                    embeddings[idx] = word_vector
                else:
                    unknown_tokens.add(token)
            assert list(module.weight.shape) == [
                len(itos),
                len(next(iter(self.word_vector_dict.values()))),
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            module.weight = nn.Parameter(torch.tensor(embeddings))
            if unknown_tokens:
                log_info(
                    "there are %s unrecognized tokens in word vectors for a total of %s",
                    len(unknown_tokens),
                    len(itos),
                )

        log_debug("load word vector %s", self.__name)
        ModelUtil(model).change_modules(f=__load_embedding, module_type=nn.Embedding)

    @classmethod
    def get_root_dir(cls) -> str:
        return os.getenv("PYTORCH_WORD_VECTOR_ROOT_DIR", cls.__word_vector_root_dir)

    def __download(self) -> dict:
        word_vector_dict: dict = {}
        urls: dict = {
            "glove.6B.300d": (
                "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
                "sha256:617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb",
            ),
            "glove.840B.300d": (
                "https://nlp.stanford.edu/data/glove.840B.300d.zip",
                "sha256:c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f",
            ),
        }
        pickle_file = os.path.join(self.get_root_dir(), self.__name + ".pkl")
        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                log_info("load cached word vectors")
                return pickle.load(f)
        urls["glove.6B.50d"] = urls["glove.6B.300d"]
        urls["glove.6B.100d"] = urls["glove.6B.300d"]
        urls["glove.6B.200d"] = urls["glove.6B.300d"]
        url, checksum = urls.get(self.__name, (None, None))
        if url is None:
            raise RuntimeError(f"unknown word vector {self.__name}")
        tarball = TarballSource(
            spec=PackageSpecification(self.__name.replace(".", "")),
            url=url,
            root_dir=self.get_root_dir(),
            checksum=checksum,
        )
        with tarball:
            if self.__name.startswith("glove"):
                dim = int(self.__name.split(".")[-1].replace("d", ""))
                with open(f"{self.__name}.txt", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip().split()
                        word_vector_dict[" ".join(s[:-dim])] = torch.tensor(
                            [float(i) for i in s[-dim:]]
                        )
        with open(pickle_file, "wb") as f:
            pickle.dump(word_vector_dict, f)
        return word_vector_dict
