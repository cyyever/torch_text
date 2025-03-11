import os

from cyy_torch_text.model import TextModelEvaluator
from cyy_torch_toolbox import Config

has_spacy: bool = False
try:
    from cyy_torch_text.tokenizer import SpacyTokenizer

    has_spacy = True
except:
    pass

if has_spacy:
    os.environ["USE_THREAD_DATALOADER"] = "1"

    def tokenizer_testcases(tokenizer: SpacyTokenizer) -> None:
        phrase = "hello world!"
        tokens = tokenizer.tokenize(phrase=phrase)
        assert tokens == ["hello", "world", "!"]
        for token in tokens:
            token_id = tokenizer.get_token_id(token)
            recovered_token = tokenizer.get_token(token_id)
            assert recovered_token == token

    def test_spacy_tokenizer() -> None:
        config = Config(dataset_name="imdb", model_name="simplelstm")
        trainer = config.create_trainer()
        assert isinstance(trainer.model_evaluator, TextModelEvaluator)
        tokenizer = trainer.model_evaluator.tokenizer
        assert isinstance(tokenizer, SpacyTokenizer)
        tokenizer_testcases(tokenizer=tokenizer)
