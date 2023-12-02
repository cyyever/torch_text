import copy

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import DatasetType, ModelType
from cyy_torch_toolbox.model_factory import globel_model_factory
from cyy_torch_toolbox.model_factory.torch_model import get_torch_model_info

from ..tokenizer import get_tokenizer
from .huggingface_model import get_hugging_face_model_info


def get_model(name: str, dataset_collection, model_kwargs: dict) -> dict:
    model_constructors = (
        get_torch_model_info()[DatasetType.Text] | get_hugging_face_model_info()
    )
    model_constructor_info = model_constructors.get(name.lower(), {})
    if not model_constructor_info:
        raise NotImplementedError(
            f"unsupported model {name}, supported models are "
            + str(model_constructors.keys())
        )

    final_model_kwargs: dict = {}
    tokenizer_kwargs = dataset_collection.dataset_kwargs.get("tokenizer", {})
    if "hugging_face" in model_kwargs.get("name", ""):
        tokenizer_kwargs["type"] = "hugging_face"
        tokenizer_kwargs["name"] = (
            model_kwargs["name"]
            .replace("hugging_face_seq2seq_lm_", "")
            .replace("hugging_face_sequence_classification_", "")
            .replace("hugging_face_", "")
        )
    tokenizer = get_tokenizer(dataset_collection, tokenizer_kwargs)
    get_logger().info("tokenizer is %s", tokenizer)

    if tokenizer is not None and hasattr(tokenizer, "itos"):
        for k in ("num_embeddings", "token_num"):
            if k not in model_kwargs:
                final_model_kwargs[k] = len(tokenizer.itos)

    final_model_kwargs |= model_kwargs
    model_type = ModelType.Classification
    if "rcnn" in name.lower():
        model_type = ModelType.Detection
    if model_type in (ModelType.Classification, ModelType.Detection):
        if "num_classes" not in final_model_kwargs:
            final_model_kwargs["num_classes"] = dataset_collection.label_number  # E:
            get_logger().debug("detect %s classes", final_model_kwargs["num_classes"])
        else:
            assert (
                final_model_kwargs["num_classes"] == dataset_collection.label_number
            )  # E:
    if model_type == ModelType.Detection:
        final_model_kwargs["num_classes"] += 1
    final_model_kwargs["num_labels"] = final_model_kwargs["num_classes"]
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model = model_constructor_info["constructor"](**final_model_kwargs)
            get_logger().debug(
                "use model arguments %s for model %s",
                final_model_kwargs,
                model_constructor_info["name"],
            )
            res = {"model": model}
            if tokenizer is not None:
                res |= {"tokenizer": tokenizer}
            word_vector_name = model_kwargs.get("word_vector_name", None)
            if word_vector_name is not None:
                from .word_vector import PretrainedWordVector

                PretrainedWordVector(word_vector_name).load_to_model(
                    model=model,
                    tokenizer=tokenizer,
                )
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(final_model_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    final_model_kwargs.pop(k)
                    retry = True
                    break
            # if not retry:
            #     if "pretrained" in str(e) and not model_kwargs["pretrained"]:
            #         model_kwargs.pop("pretrained")
            #         retry = True
            if not retry:
                raise e


globel_model_factory.register(DatasetType.Text, get_model)
