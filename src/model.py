# --- src/model.py ---
from transformers import AutoModelForSequenceClassification

def get_model(num_labels):
    """Função baseada no Passo 4 para carregar o modelo"""
    model = AutoModelForSequenceClassification.from_pretrained(
        "ehsanaghaei/SecureBERT",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model