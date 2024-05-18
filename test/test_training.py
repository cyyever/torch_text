import os
import cyy_torch_text  # noqa: F401
from cyy_torch_toolbox import Config, ExecutorHookPoint, StopExecutingException


os.environ["USE_THREAD_DATALOADER"]="1"
def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_nlp_training() -> None:
    config = Config(dataset_name="imdb", model_name="simplelstm")
    config.trainer_config.hook_config.debug = True
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    config.dc_config.dataset_kwargs["input_max_len"] = 100
    trainer = config.create_trainer()
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()
