# --- src/trainer.py ---
import torch
import torch.nn.functional as F
from transformers import Trainer
from pathlib import Path


class CustomTrainer(Trainer):
    """Trainer com Focal Loss para Multi-Label Classification.

    A Focal Loss foca nos exemplos difíceis de classificar, reduzindo o peso
    dos exemplos já bem classificados. Isso ajuda em datasets desbalanceados.

    Uso:
        - gamma=0.0 → equivalente ao BCEWithLogitsLoss padrão
        - gamma=2.0 → valor padrão recomendado na literatura
        - pos_weight → pesos de classe para desbalanceamento (opcional)
    """

    def __init__(self, focal_gamma=2.0, use_pos_weights=True, pos_weights_path="../data/processed/pos_weights.pt", **kwargs):
        super().__init__(**kwargs)
        self.focal_gamma = focal_gamma
        self.use_pos_weights = use_pos_weights
        self.pos_weights_path = Path(pos_weights_path)
        self._cached_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")


        outputs = model(**inputs)
        logits = outputs.get("logits")

 
        if self.use_pos_weights:
            if self._cached_weights is None:
                weights = torch.load(
                    self.pos_weights_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self._cached_weights = weights
            weights = self._cached_weights.to(model.device)
            bce = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=weights, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        # Focal modulation: (1 - p_t)^gamma * BCE
        p_t = torch.exp(-bce)
        focal_loss = ((1 - p_t) ** self.focal_gamma) * bce
        loss = focal_loss.mean()

        return (loss, outputs) if return_outputs else loss
