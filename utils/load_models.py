#load_models.py

import os
import joblib
import pickle
import torch
import torch.nn as nn


# -------------------------
# Architecture: SimpleTransformerClassifier
# -------------------------
class SimpleTransformerClassifier(nn.Module):
    """
    Lightweight transformer-on-top-of-linear embedding classifier used for per-phase models.
    This matches the architecture used during training in your notebook.
    """

    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.embedding(x).unsqueeze(1)  # (batch, seq=1, d_model)
        x = self.transformer(x).mean(dim=1)  # (batch, d_model)
        return self.fc(x)  # (batch, num_classes)


# -------------------------
# Utilities for loading models + encoders
# -------------------------
def _infer_input_dim_from_embeddings(
    model_root, emb_filename="technique_embeddings.pkl"
):
    """
    Try to infer the embedding dimensionality by loading a sample vector
    from a saved embeddings pickle stored under model_root.
    Returns integer embedding dim if found, otherwise None.
    """
    emb_path = os.path.join(model_root, emb_filename)
    if not os.path.exists(emb_path):
        return None
    try:
        with open(emb_path, "rb") as f:
            emb_map = pickle.load(f)
        # emb_map keys are (phase, tech) -> numpy array
        for v in emb_map.values():
            try:
                return int(v.shape[-1])
            except Exception:
                continue
    except Exception:
        return None
    return None


def _load_label_encoder(label_encoder_path):
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found at: {label_encoder_path}")
    return joblib.load(label_encoder_path)


def _load_state_dict(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model state dict not found at: {model_path}")
    return torch.load(model_path, map_location="cpu")


def load_phase_model(
    phase_name, model_root, input_dim=None, emb_map_filename="technique_embeddings.pkl"
):
    """
    Load a single phase's model and its label encoder.

    Args:
        phase_name (str): short phase folder name (e.g. "c2", "recon", "weapon", ...)
        model_root (str): root folder that contains phase subfolders that have model + encoder.
            Expected layout:
                model_root/
                  <phase_name>/
                    model.pt
                    label_encoder.pkl
            OR if you used a different structure, pass explicit paths accordingly.
        input_dim (int, optional): embedding input dimension. If None, this function will try to
            infer input_dim from model_root/emb_map_filename (defaults to technique_embeddings.pkl).
        emb_map_filename (str): name of pickle under model_root to infer input dim (optional)

    Returns:
        (model, label_encoder)
            model: torch.nn.Module set to eval() on CPU
            label_encoder: sklearn LabelEncoder (or equivalent) loaded via joblib
    """
    phase_dir = os.path.join(model_root, phase_name)
    if not os.path.isdir(phase_dir):
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

    # locate artifacts
    model_path = os.path.join(phase_dir, "model.pt")
    label_encoder_path = os.path.join(phase_dir, "label_encoder.pkl")

    # load label encoder first to know number of classes
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(
            f"Label encoder file missing for phase '{phase_name}'. Expected at: {label_encoder_path}"
        )
    label_encoder = _load_label_encoder(label_encoder_path)
    num_classes = len(getattr(label_encoder, "classes_", []))

    # infer input_dim if not supplied
    if input_dim is None:
        inferred = _infer_input_dim_from_embeddings(model_root, emb_map_filename)
        if inferred is not None:
            input_dim = inferred
        else:
            raise ValueError(
                "input_dim not provided and could not be inferred from embeddings. "
                "Either provide input_dim argument or place a technique_embeddings.pkl at model_root."
            )

    # Instantiate model and load state
    model = SimpleTransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    state = _load_state_dict(model_path)
    # If state is a dict with keys 'model_state_dict' (wrapped), try common variants
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model, label_encoder


def load_phase_models(
    model_root,
    phase_names=None,
    input_dim=None,
    emb_map_filename="technique_embeddings.pkl",
):
    """
    Load multiple phase models.

    Args:
        model_root (str): root folder that contains phase subfolders.
        phase_names (list[str], optional): list of phase names to load. If None, uses default common set.
        input_dim (int, optional): if provided, used for all phases; otherwise inferred from embeddings.
        emb_map_filename (str): filename to look for under model_root to infer input dim.

    Returns:
        dict: { phase_name: (model, label_encoder) }
    """
    if phase_names is None:
        phase_names = [
            "recon",
            "weapon",
            "delivery",
            "exploit",
            "install",
            "c2",
            "objectives",
        ]

    models = {}
    # if input_dim is None, try to infer once
    inferred_dim = None
    if input_dim is None:
        inferred_dim = _infer_input_dim_from_embeddings(model_root, emb_map_filename)

    for p in phase_names:
        models[p] = load_phase_model(
            p,
            model_root=model_root,
            input_dim=(input_dim if input_dim is not None else inferred_dim),
            emb_map_filename=emb_map_filename,
        )
    return models


# ============================
# Ensemble model loading
# ============================
def load_ensemble_phase_models(
    model_root,
    phase_names=None,
    models_per_phase=2,
    input_dim=None,
    emb_map_filename="technique_embeddings.pkl",
):
    """
    Load multiple model variants per phase for ensemble prediction.
    For demonstration, we load the same model multiple times.
    In practice, load different model architectures (Transformer, LightGBM, etc.).

    Args:
        model_root (str): root folder that contains phase subfolders.
        phase_names (list[str], optional): list of phase names to load.
        models_per_phase (int): number of model variants to load per phase.
        input_dim (int, optional): embedding input dimension.
        emb_map_filename (str): filename for embeddings.

    Returns:
        dict: { phase: [(model1, le), (model2, le), ...] }
    """
    if phase_names is None:
        phase_names = [
            "recon",
            "weapon",
            "delivery",
            "exploit",
            "install",
            "c2",
            "objectives",
        ]

    ensemble_models = {}
    for phase in phase_names:
        models_list = []
        for i in range(models_per_phase):
            # For demo, load the same model multiple times
            # In production, load different model types or trained variants
            try:
                model, label_encoder = load_phase_model(
                    phase,
                    model_root=model_root,
                    input_dim=input_dim,
                    emb_map_filename=emb_map_filename,
                )
                models_list.append((model, label_encoder))
            except Exception as e:
                print(f"Could not load model {i} for phase {phase}: {e}")
                break
        if models_list:
            ensemble_models[phase] = models_list

    return ensemble_models


# -------------------------
# Example / quick test (run as script)
# -------------------------
if __name__ == "__main__":
    # Example usage. Adjust model_root to match your repo layout.
    example_model_root = "../transformer_model_killchain"  # adjust as needed

    print("Attempting to load single phase models from:", example_model_root)
    try:
        models = load_phase_models(example_model_root)
        for phase, (m, le) in models.items():
            print(
                f"- Loaded phase '{phase}': model={type(m).__name__}, classes={len(le.classes_)}"
            )
    except Exception as e:
        print("Error loading single models:", e)
        print(
            "If error is 'input_dim not provided', either provide input_dim to load_phase_models or place technique_embeddings.pkl at model_root."
        )

    print("\nAttempting to load ensemble models (2 per phase):")
    try:
        ensemble_models = load_ensemble_phase_models(example_model_root, models_per_phase=2)
        for phase, models_list in ensemble_models.items():
            print(f"- Phase '{phase}': {len(models_list)} models loaded")
    except Exception as e:
        print("Error loading ensemble models:", e)
