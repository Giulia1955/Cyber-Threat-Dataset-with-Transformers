# --- src/trainer.py ---
from transformers import Trainer
from torch import nn

class WeightedTrainer(Trainer):
    """Classe customizada do Passo 4 para lidar com desbalanceamento"""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights deve ser um tensor enviado para a GPU
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # BCEWithLogitsLoss com pos_weight para tratar desbalanceamento
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss