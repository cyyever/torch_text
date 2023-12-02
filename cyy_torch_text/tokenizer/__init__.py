from typing import Any

import transformers

from .spacy import SpacyTokenizer


def get_tokenizer(dc, tokenizer_config: dict) -> Any:
    tokenizer_type: str = tokenizer_config.get("type", "spacy")
    match tokenizer_type:
        case "hugging_face":
            return transformers.AutoTokenizer.from_pretrained(
                tokenizer_config["name"], **tokenizer_config.get("kwargs", {})
            )
        case "spacy":
            return SpacyTokenizer(dc, **tokenizer_config.get("kwargs", {}))
    return None
