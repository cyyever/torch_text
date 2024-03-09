from cyy_torch_text.model import TextModelEvaluator
from cyy_torch_text.tokenizer import (HuggingFaceTokenizer, SpacyTokenizer,
                                      Tokenizer)
from cyy_torch_text.tokenizer.util import convert_phrase_to_token_ids
from cyy_torch_toolbox import Config, Executor


def tokenizer_testcases(executor: Executor, tokenizer: Tokenizer) -> None:
    phrase = "hello world!"
    tokens = tokenizer.tokenize(phrase=phrase)
    assert tokens == ["hello", "world", "!"]
    for token in tokens:
        token_id = tokenizer.get_token_id(token)
        recovered_token = tokenizer.get_token(token_id)
        assert recovered_token == token
    convert_phrase_to_token_ids(executor=executor, phrase=phrase)


def test_spacy_tokenizer() -> None:
    config = Config(dataset_name="imdb", model_name="simplelstm")
    trainer = config.create_trainer()
    assert isinstance(trainer.model_evaluator, TextModelEvaluator)
    tokenizer = trainer.model_evaluator.tokenizer
    assert isinstance(tokenizer, SpacyTokenizer)
    tokenizer_testcases(executor=trainer, tokenizer=tokenizer)


def test_hugging_face_tokenizer() -> None:
    config = Config(
        dataset_name="imdb",
        model_name="hugging_face_sequence_classification_distilbert-base-cased",
    )
    config.dc_config.dataset_kwargs["max_len"] = 300
    trainer = config.create_trainer()
    assert isinstance(trainer.model_evaluator, TextModelEvaluator)
    tokenizer = trainer.model_evaluator.tokenizer
    assert isinstance(tokenizer, HuggingFaceTokenizer)
    tokenizer_testcases(executor=trainer, tokenizer=tokenizer)
