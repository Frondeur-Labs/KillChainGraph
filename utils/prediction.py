# prediction.py

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer


# ==============================
# Embedding utilities
# ==============================
def encode_with_attack_bert(
    text: str, sentence_model: SentenceTransformer
) -> np.ndarray:
    """
    Encode a single input text into ATTACK-BERT embedding.
    Args:
        text (str): input attack text
        sentence_model (SentenceTransformer): pretrained model (basel/ATTACK-BERT)
    Returns:
        np.ndarray: embedding vector (1D)
    """
    emb = sentence_model.encode([text], show_progress_bar=False)
    return emb[0]  # shape: (embedding_dim,)


# ==============================
# Prediction per phase
# ==============================
@torch.no_grad()
def predict_topk_for_phase(
    phase_name: str,
    model: torch.nn.Module,
    label_encoder: LabelEncoder,
    text_vector: np.ndarray,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict top-k techniques for one phase.
    Args:
        phase_name (str): phase key (e.g., "recon", "weapon", etc.)
        model (nn.Module): trained transformer classifier for the phase
        label_encoder (LabelEncoder): fitted encoder for technique labels
        text_vector (np.ndarray): input vector (embedding)
        k (int): top-k results
    Returns:
        List[Tuple[str, float]]: list of (technique_name, probability)
    """
    x_tensor = torch.tensor(text_vector, dtype=torch.float32).unsqueeze(0)
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idxs = probs.argsort()[::-1][:k]

    return [(label_encoder.classes_[i], float(probs[i])) for i in idxs]


# ==============================
# Full kill chain prediction
# ==============================
def run_kill_chain_prediction(
    text: str,
    phase_models: Dict[str, Tuple[torch.nn.Module, LabelEncoder]],
    sentence_model: SentenceTransformer,
    k: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Run technique prediction across all kill chain phases.
    Args:
        text (str): full attack description text
        phase_models (dict): { phase: (model, label_encoder) }
        sentence_model (SentenceTransformer): embedding model (ATTACK-BERT)
        k (int): top-k techniques to return per phase
    Returns:
        Dict[str, List[Tuple[str, float]]]
    """
    # Encode once
    text_vec = encode_with_attack_bert(text, sentence_model)

    predictions = {}
    for phase, (model, le) in phase_models.items():
        preds = predict_topk_for_phase(phase, model, le, text_vec, k=k)
        predictions[phase] = preds

    return predictions


# ==============================
# Debug / test
# ==============================
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from load_models import load_phase_models

    MODEL_ROOT = "../transformer_model_killchain"

    # Load sentence model + phase models
    sentence_model = SentenceTransformer("basel/ATTACK-BERT")
    phase_models = load_phase_models(MODEL_ROOT)

    test_text = "Suspicious phishing email detected with a malicious Word document that executed macros and downloaded a RAT for remote control."

    preds = run_kill_chain_prediction(test_text, phase_models, sentence_model, k=3)

    print("\n=== Top Predictions per Phase ===")
    for phase, results in preds.items():
        print(f"\n{phase}:")
        for tech, score in results:
            print(f"  {tech} (prob={score:.4f})")
