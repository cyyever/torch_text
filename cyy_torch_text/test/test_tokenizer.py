from cyy_torch_text.tokenizer.util import convert_phase_to_token_ids
from cyy_torch_toolbox import Config, MachineLearningPhase


def test_tokenizer() -> None:
    pass
    # config = Config(dataset_name="imdb", model_name="simplelstm")
    # trainer = config.create_trainer()
    # inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    # convert_phase_to_token_ids(executor=inferencer, phrase="Hello world!")

    # config = Config(dataset_name="imdb", model_name="TransformerClassificationModel")
    # config.model_config.model_kwargs["max_len"] = 300
    # trainer = config.create_trainer()
    # inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    # convert_phase_to_token_ids(executor=inferencer, phrase="Hello world!")
