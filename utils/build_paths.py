# utils/build_paths.py
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import pandas as pd
import html as _html


def build_paths_from_edges(predicted_dict, edges, technique_df=None, top_k=10):
    """
    Build diverse paths along semantic edges, ensuring each path
    uses different techniques where possible.
    """
    phase_order = [
        "recon",
        "weapon",
        "delivery",
        "exploit",
        "install",
        "c2",
        "objectives",
    ]
    name_to_id = (
        {row["Name"]: row["ID"] for _, row in technique_df.iterrows()}
        if technique_df is not None
        else {}
    )

    # Build adjacency lookup from edges
    adjacency = {}
    for src_phase, src_tech, tgt_phase, tgt_tech, sim in edges:
        adjacency.setdefault((src_phase, src_tech), []).append(
            (tgt_phase, tgt_tech, sim)
        )

    # Global record of used techniques per phase
    used_per_phase = {p: set() for p in phase_order}

    final_paths = []

    # Iterate until we collect top_k unique paths
    for recon_tech, recon_prob in predicted_dict.get("recon", []):
        if len(final_paths) >= top_k:
            break

        stack = [("recon", recon_tech, recon_prob, name_to_id.get(recon_tech, None))]
        score = recon_prob
        current_phase, current_tech = "recon", recon_tech
        valid = True

        for i in range(len(phase_order) - 1):
            src, tgt = phase_order[i], phase_order[i + 1]
            options = adjacency.get((src, current_tech), [])
            if not options:
                valid = False
                break

            # filter out techniques already used in this phase
            options = [o for o in options if o[1] not in used_per_phase[tgt]]

            if not options:
                valid = False
                break

            # pick best among remaining
            tgt_phase, tgt_tech, sim = max(options, key=lambda x: x[2])
            stack.append((tgt_phase, tgt_tech, sim, name_to_id.get(tgt_tech, None)))
            score += sim
            current_phase, current_tech = tgt_phase, tgt_tech

        if valid and stack[-1][0] == "objectives":
            final_paths.append((score, stack))
            # Mark techniques as used
            for phase, tech, _, _ in stack:
                used_per_phase[phase].add(tech)

    return sorted(final_paths, key=lambda x: x[0], reverse=True), len(final_paths)


PHASE_COLORS = {
    "recon": "#1f77b4",
    "weapon": "#ff7f0e",
    "delivery": "#2ca02c",
    "exploit": "#d62728",
    "install": "#9467bd",
    "c2": "#8c564b",
    "objectives": "#e377c2",
}


def render_paths_cards(paths, total_complete, max_cols=2):
    """
    Render paths as inline cards.
    `paths` = [(score, [(phase, tech, _, tech_id), ...]), ...]
    """
    st.markdown(
        f"### 🛤️ Top {len(paths)} Predicted Kill Chain Paths ",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(max_cols)
    for i, (score, path) in enumerate(paths):
        col = cols[i % max_cols]
        with col:
            chain_text = " → ".join(
                [
                    f"{phase}:{tech} ({tid})" if tid else f"{phase}:{tech}"
                    for phase, tech, _, tid in path
                ]
            )

            # Build rows for phases/techniques
            node_rows = []
            for phase, tech, _, tid in path:
                tid_str = (
                    f"<span style='color:#9aa5b1; font-size:12px'>{_html.escape(str(tid))}</span>"
                    if tid
                    else ""
                )
                pcol = PHASE_COLORS.get(phase, "#9ad1ff")

                node_rows.append(
                    f"<div style='margin:4px 0; display:flex; gap:8px; align-items:center;'>"
                    f"<div style='min-width:110px; color:{pcol}; font-weight:600;'>{_html.escape(phase.capitalize())}</div>"
                    f"<div style='flex:1; color:#ffffff;'>{_html.escape(tech)}</div>"
                    f"<div style='min-width:80px; text-align:right;'>{tid_str}</div>"
                    f"</div>"
                )

            nodes_html = "".join(node_rows)
            safe_chain = _html.escape(chain_text).replace('"', "&quot;")

            card_html = (
                f"<div style='background:#111; border-radius:10px; padding:12px; "
                f"margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.4);'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<div style='font-weight:700; font-size:14px; color:#fff;'>Path {i+1}</div>"
                f"<div style='background:#ffa500; padding:4px 10px; border-radius:999px; "
                f"color:#111; font-weight:700; font-size:12px;'>{float(score):.4f}</div>"
                f"</div>"
                f"<div style='margin-top:8px; font-size:13px; color:#e6e6e6;'>{nodes_html}</div>"
                f"<div style='display:flex; justify-content:flex-end; margin-top:8px;'>"
                f"</div>"
                f"</div>"
            )

            col.markdown(card_html, unsafe_allow_html=True)
