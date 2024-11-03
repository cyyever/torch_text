from cyy_huggingface_toolbox import HuggingFaceTokenizer

from .base import Tokenizer
from .spacy import SpacyTokenizer


def get_tokenizer(dc, tokenizer_config: dict) -> Tokenizer | None:
    tokenizer_type: str = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            return HuggingFaceTokenizer(tokenizer_config)
        case "spacy":
            return SpacyTokenizer(dc, **tokenizer_config.get("kwargs", {}))
    return None
