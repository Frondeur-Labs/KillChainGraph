import os
import pickle
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# 1. Technique embeddings loader
# -------------------------
def build_technique_embedding_dict(
    embedding_pkl_path: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Load precomputed ATTACK-BERT embeddings for all techniques.

    Args:
        embedding_pkl_path (str): path to technique_embeddings.pkl

    Returns:
        dict: { (phase, technique_name): embedding_vector }
    """
    if not os.path.exists(embedding_pkl_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_pkl_path}")
    with open(embedding_pkl_path, "rb") as f:
        emb_map = pickle.load(f)
    return emb_map


# -------------------------
# 2. Phase linkages via cosine similarity
# -------------------------

def build_phase_linkages(predicted_dict, emb_map, sim_threshold=0.01, top_k_edges=None):
    """
    Build semantic edges between predicted techniques across phases.
    - Keep edges with cosine similarity >= sim_threshold
    - If top_k_edges is set, also cap to that many strongest edges per phase pair
    """
    edges = []
    phase_order = [
        "recon",
        "weapon",
        "delivery",
        "exploit",
        "install",
        "c2",
        "objectives",
    ]

    for i in range(len(phase_order) - 1):
        src_phase, tgt_phase = phase_order[i], phase_order[i + 1]
        local_edges = []

        for src_tech, _ in predicted_dict[src_phase]:
            for tgt_tech, _ in predicted_dict[tgt_phase]:
                emb_a = emb_map.get((src_phase, src_tech))
                emb_b = emb_map.get((tgt_phase, tgt_tech))
                if emb_a is not None and emb_b is not None:
                    sim = float(cosine_similarity([emb_a], [emb_b])[0][0])
                    if sim >= sim_threshold:
                        local_edges.append(
                            (src_phase, src_tech, tgt_phase, tgt_tech, sim)
                        )

        # keep only top-k if requested
        if top_k_edges is not None:
            local_edges = sorted(local_edges, key=lambda x: x[4], reverse=True)[
                :top_k_edges
            ]

        edges.extend(local_edges)

    return edges, None


# -------------------------
# Debug / test
# -------------------------
if __name__ == "__main__":
    # Example run
    predicted_dict = {
        "recon": [("Social Media", 0.98), ("Link Target", 0.01)],
        "weapon": [("Outlook Forms", 0.99), ("Dynamic Data Exchange", 0.01)],
        "delivery": [("HTML Smuggling", 0.98)],
        "exploit": [("Install Root Certificate", 0.32)],
        "install": [("RC Scripts", 0.61)],
        "c2": [("Mail Protocols", 0.97)],
        "objectives": [("File Transfer Protocols", 0.24)],
    }

    emb_map = build_technique_embedding_dict(
        "../transformer_model_killchain/technique_embeddings.pkl"
    )
    edges, adj = build_phase_linkages(predicted_dict, emb_map)

    print("\n=== Sample Edges ===")
    for e in edges[:10]:
        print(e)
