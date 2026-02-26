from typing import Any

from cyy_huggingface_toolbox import HuggingFaceTokenizer
from cyy_torch_toolbox import DatasetCollection, Tokenizer

has_spacy: bool = False
try:
    from .spacy import SpacyTokenizer

    has_spacy = True
except:
    pass


def get_tokenizer(dc: DatasetCollection, tokenizer_config: dict[str, Any]) -> Tokenizer | None:
    tokenizer_type: str = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            return HuggingFaceTokenizer(tokenizer_config)
        case "spacy":
            if has_spacy:
                kwargs = tokenizer_config.get("kwargs", {}).copy()
                if "name" in tokenizer_config and "model_name" not in kwargs:
                    kwargs["model_name"] = tokenizer_config["name"]
                return SpacyTokenizer(dc, **kwargs)
            else:
                raise RuntimeError("Spacy is broken")
    return None


__all__ = ["Tokenizer", "get_tokenizer"]
