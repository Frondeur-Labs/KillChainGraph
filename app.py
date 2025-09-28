import os

os.environ["STREAMLIT_WATCHER_IGNORE_MODULES"] = "torch"

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

# 🔹 Import utils
from utils.load_models import load_phase_models
from utils.prediction import run_kill_chain_prediction
from utils.linkage import (
    build_technique_embedding_dict,
    build_phase_linkages,
)
from utils.graph_viz import visualize_kill_chain_graph
from utils.build_paths import render_paths_cards, build_paths_from_edges


# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="🛡️ Cyber Kill Chain Analyzer")
st.title("🛡️ Cyber Kill Chain Phase Predictor")


# -------------------------
# Cache loading
# -------------------------


@st.cache_resource(show_spinner=False)
def load_all():
    progress = st.progress(0, text="🔄 Initializing...")

    # Step 1 - Load ATTACK-BERT
    progress.progress(20, text="📥 Loading ATTACK-BERT model...")
    sentence_model = SentenceTransformer("basel/ATTACK-BERT")

    # Step 2 - Load phase-specific models
    progress.progress(50, text="⚙️ Loading phase-wise transformer models...")
    phase_models = load_phase_models("transformer_model_killchain")

    # Step 3 - Load technique embeddings
    progress.progress(75, text="🧠 Loading technique embeddings...")
    emb_map = build_technique_embedding_dict(
        "transformer_model_killchain/technique_embeddings.pkl"
    )

    # Step 4 - Load Name ↔ ID mapping
    progress.progress(95, text="📊 Loading MITRE technique mapping...")
    technique_df = pd.read_csv("data/attack_techniques.csv")

    progress.progress(100, text="✅ All models loaded successfully!")

    return sentence_model, phase_models, emb_map, technique_df


# -------------------------
# Input Section
# -------------------------
attack_input = st.text_area("Paste the full attack plan below:", height=300)

if st.button("Run Prediction and Mapping"):
    if not attack_input.strip():
        st.warning("⚠️ Please paste an attack plan before running.")
    else:
        sentence_model, phase_models, technique_embeddings, technique_df = load_all()

        # -------------------------
        # 1. Run predictions
        # -------------------------
        with st.spinner("⏳ Running phase-wise technique predictions..."):
            predicted_techniques = run_kill_chain_prediction(
                attack_input, phase_models, sentence_model, k=10
            )

        # st.success("✅ Technique prediction completed.")
        # st.subheader("📋 Top Predicted Techniques per Phase")

        # name_to_id = {row["Name"]: row["ID"] for _, row in technique_df.iterrows()}

        # for phase, techniques in predicted_techniques.items():
        #     st.markdown(f"### 🔹 {phase.capitalize()}")
        #     for tech_name, score in techniques:
        #         tech_id = name_to_id.get(tech_name, "❓ Unknown ID")
        #         st.markdown(f"- **{tech_name}** (`{tech_id}`): `{score:.4f}`")

        st.success("✅ Technique prediction completed.")
        st.subheader("📋 Top Predicted Techniques per Phase")

        name_to_id = {row["Name"]: row["ID"] for _, row in technique_df.iterrows()}

        # Create 7 columns (one per phase)
        cols = st.columns(7)

        # Define the phase order to display
        phase_order = [
            "recon",
            "weapon",
            "delivery",
            "exploit",
            "install",
            "c2",
            "objectives",
        ]

        for idx, phase in enumerate(phase_order):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background-color:#1e1e1e; padding:15px; border-radius:12px; 
                                box-shadow:0 4px 6px rgba(0,0,0,0.3); text-align:center;">
                        <h4 style="color:#ffa500; margin-bottom:10px;">{phase.capitalize()}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                techniques = predicted_techniques.get(phase, [])
                for tech_name, score in techniques:
                    tech_id = name_to_id.get(tech_name, "❓ Unknown ID")
                    st.markdown(
                        f"""
                        <div style="background-color:#2b2b2b; padding:8px; margin:6px 0; 
                                    border-radius:8px; text-align:left; color:white;">
                            <b>{tech_name}</b><br>
                            <small>ID: {tech_id}</small><br>
                            <small>Score: {score:.4f}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # -------------------------
        # 2. Compute semantic links
        # -------------------------
        with st.spinner("🔗 Computing semantic links between phases..."):
            edges, _ = build_phase_linkages(
                predicted_techniques,
                technique_embeddings,
                sim_threshold=0.3,
                top_k_edges=None,
            )

        st.success("✅ Semantic mapping complete.")

        # -------------------------
        # 3. Visualize graph
        # -------------------------
        # st.subheader("📈 Kill Chain Graph")
        # visualize_kill_chain_graph(predicted_techniques, edges, technique_df)

        paths, total_complete = build_paths_from_edges(
            predicted_techniques, edges, technique_df=technique_df, top_k=5
        )

        best_path = paths[0][1] if paths else None

        st.markdown(
            """
            ### 📊 Graph Legend
            - **Dark-colored nodes** → Best predicted technique (highest probability) for that phase.  
            - **Orange edges** → Strongest similarity edge between two phases.  
            - **Pink edges** → Edges belonging to the best predicted path.  
            """
        )

        st.subheader("📈 Kill Chain Graph")
        visualize_kill_chain_graph(
            predicted_techniques, edges, technique_df, best_path=best_path
        )

        # -------------------------
        # 4. Show link details
        # -------------------------
        # with st.expander("🧠 Semantic Link Details"):
        #     for edge in edges:
        #         src_phase, src_tech, tgt_phase, tgt_tech, sim = edge
        #         st.markdown(
        #             f"- **{src_phase} → {tgt_phase}**: `{src_tech}` → `{tgt_tech}` | Cosine Sim: `{sim:.3f}`"
        #         )

        # -------------------------
        # 5. Build kill chain paths
        # -------------------------

        # st.subheader(f"🛤️ Top {len(paths)} Predicted Kill Chain Paths ")

        # with st.expander("🛤️ Paths"):
        #     for idx, (score, path) in enumerate(paths, 1):
        #         chain = " → ".join(
        #             [
        #                 f"{phase}:{tech} ({tech_id})" if tech_id else f"{phase}:{tech}"
        #                 for phase, tech, _, tech_id in path
        #             ]
        #         )
        #         st.markdown(f"**Path {idx}** | Score={score:.4f}  \n{chain}")

        with st.expander("🛤️ Paths", expanded=True):
            render_paths_cards(paths, total_complete, max_cols=2)
