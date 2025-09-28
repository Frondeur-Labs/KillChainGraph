# 🛡️ KillChainGraph

A phase-aware multi-model ML framework emulating adversarial behavior across the Cyber Kill Chain using MITRE ATT&CK. ATTACK-BERT maps techniques to seven phases, forming phase-specific datasets. We assess LightGBM, a custom Transformer, fine-tuned BERT, and GNN, combining outputs via weighted soft-voting. This repo only contains the transformer model with a Streamlit app that takes a free-form attack plan (text) and predicts likely MITRE ATT&CK techniques across the seven kill-chain phases, computes semantic links between techniques, visualizes the layered kill-chain graph, and outputs the top diverse attack paths.

---

## Introduction

This app converts a natural-language attack description into predicted MITRE ATT&CK techniques separated into seven phases:

**Recon → Weapon → Delivery → Exploit → Install → C2 → Objectives**

Key capabilities:

- Phase-wise predicted techniques (top-k per phase)
- Semantic similarity edges between predicted techniques across phases (cosine similarity on ATTACK-BERT embeddings)
- Interactive layered graph (pyvis) with legend and highlighted best path
- Top-K diverse kill-chain paths (tries to avoid reusing techniques across returned paths)
- Pretty path cards with copy-to-clipboard

---

## Data & Sources

- Technique metadata (Name ↔ ID) and technique descriptions from **MITRE ATT&CK**.
- Kill-chain phase grouping inspired by Lockheed Martin’s Kill Chain; MITRE techniques are semantically mapped into **7 phase datasets** using ATTACK-BERT embeddings.
- Precomputed technique embeddings are stored in `transformer_model_killchain/technique_embeddings.pkl`.

---

## Visual legend (what you see in the graph)

- **Dark-colored node**: top predicted technique (highest probability) in that phase.
- **Red edge**: strongest cosine-similarity edge between two phases (per phase-pair).
- **Pink edge**: edges that belong to the **best predicted path**.

A markdown legend is displayed above the graph in the app.

---

## Requirements

- Python 3.9+ recommended

Create and/or activate a Python environment (venv / conda) before installing.

Example (venv):

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

## Install dependencies

pip install -r requirements.txt

## Setup & Pretrained models

- Ensure per-phase models are placed where `load_phase_models` expects them (or update that function to your model paths).
- Place `technique_embeddings.pkl` in `transformer_model_killchain/` (or the path used by `build_technique_embedding_dict`).
- Ensure `data/attack_techniques.csv` contains the Name↔ID mappings.

## Run the app

streamlit run app.py

## How to use

1. Paste/enter a **full attack plan** (free text) into the input area.

   Example input:

   _"Phishing email with malicious Office template; macro drops a PowerShell payload which installs persistence via login items and communicates over mail protocols."_

2. Click **Run Prediction and Mapping** .

The app will:

- Load models (progress bar shown).
- Run phase-wise predictions (top-k techniques).
- Compute semantic links (cosine similarity) between predicted techniques. You can tune `sim_threshold` in `build_phase_linkages`.
- Visualize the layered kill-chain graph with legend and highlighted best path.
- Build and display top-K diverse kill-chain paths and render cards for easy copying.
- Expand “Semantic Link Details” to inspect pairwise link similarities.
- Expand “Paths” to view chain text and use the copy button on cards.

## Tuning & options

- `k` in `run_kill_chain_prediction(..., k=10)`: candidates per phase.
- `sim_threshold` in `build_phase_linkages`: lower values show more edges (e.g., `0.01`) — but may clutter the graph.
- `top_k` in `build_paths_from_edges` / `build_paths`: number of diverse paths returned.
- Diversity mode: the path-builder can be strict or soft about reuse of techniques across paths — tune the algorithm in `build_paths.py` if needed.

## Troubleshooting

- **“Running load_all()” spinner** : The loader shows an internal progress bar. If you see Streamlit’s default spinner, use `@st.cache_resource(show_spinner=False)` to hide it.
- **Missing embeddings/models** : verify file paths to `technique_embeddings.pkl` and phase model folders.
- **Graph missing edges** : make sure `sim_threshold` is low enough if you want faint/low-similarity connections, and check `top_k_edges` behavior in `build_phase_linkages`.
- **Repeated path cards** : ensure `render_paths_cards` builds cards per path and no accidental trailing commas or malformed f-strings are present.

## License & Credits

- MITRE ATT&CK used for technique names & IDs.
- Embeddings + architecture: ATTACK-BERT / sentence-transformers.
- Graph and visualization: pyvis + networkx.
