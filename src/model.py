from transformers import AutoModelForSequenceClassification

model_name = "cisco-ai/SecureBERT2.0-base"

def get_model(num_labels, id2label=None, label2id=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )
    return model