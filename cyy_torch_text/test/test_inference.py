from cyy_torch_toolbox import Config, MachineLearningPhase


def test_gradient() -> None:
    config = Config(dataset_name="imdb", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    config.dc_config.dataset_kwargs["input_max_len"] = 100
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.get_gradient()
