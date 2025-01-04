from cyy_huggingface_toolbox import HuggingFaceTokenizer

from .base import Tokenizer

has_spacy: bool = False
try:
    from .spacy import SpacyTokenizer

    has_spacy = True
except:
    pass


def get_tokenizer(dc, tokenizer_config: dict) -> Tokenizer | None:
    tokenizer_type: str = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            return HuggingFaceTokenizer(tokenizer_config)
        case "spacy":
            if has_spacy:
                return SpacyTokenizer(dc, **tokenizer_config.get("kwargs", {}))
            else:
                raise RuntimeError("Spacy is broken")
    return None
