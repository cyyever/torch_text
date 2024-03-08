from cyy_torch_text.tokenizer.util import collect_token_indices  # noqa: F401
from cyy_torch_toolbox import Config, MachineLearningPhase


def test_tokenizer() -> None:
    config = Config(dataset_name="imdb", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    collect_token_indices(executor=inferencer, phrase="Hello world!")
