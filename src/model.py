import os

from transformers import AutoConfig, AutoModelForSequenceClassification


# Evita a thread de auto-conversão para safetensors no Hub, que pode gerar 403 não fatal.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


MODEL_REGISTRY = {
    "securebert2": "cisco-ai/SecureBERT2.0-base",
    "securebert": "ehsanaghaei/SecureBERT",
    "bert": "bert-base-uncased",
}
DEFAULT_MODEL_NAME = MODEL_REGISTRY["securebert2"]


def resolve_model_name(model_name=None, model_key=None):
    if model_name:
        return model_name
    if model_key:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"model_key '{model_key}' inválido. Opções: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model_key]
    return DEFAULT_MODEL_NAME


def _apply_dropout_config(config, hidden_dropout=0.3, attention_dropout=0.2, classifier_dropout=0.3):
    config_dict = config.to_dict()
    if "hidden_dropout" in config_dict:
        config.hidden_dropout = hidden_dropout
    if "hidden_dropout_prob" in config_dict:
        config.hidden_dropout_prob = hidden_dropout
    if "attention_dropout" in config_dict:
        config.attention_dropout = attention_dropout
    if "attention_probs_dropout_prob" in config_dict:
        config.attention_probs_dropout_prob = attention_dropout
    if "classifier_dropout" in config_dict:
        config.classifier_dropout = classifier_dropout


def get_model(
    num_labels,
    id2label=None,
    label2id=None,
    model_name=None,
    model_key=None,
    hidden_dropout=0.3,
    attention_dropout=0.2,
    classifier_dropout=0.3,
):
    selected_model_name = resolve_model_name(model_name=model_name, model_key=model_key)
    config = AutoConfig.from_pretrained(
        selected_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )
    _apply_dropout_config(
        config=config,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        classifier_dropout=classifier_dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        selected_model_name,
        config=config,
    )
    return model
