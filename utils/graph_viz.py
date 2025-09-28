# utils/graph_viz.py
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Phase color palette (dark base colors)
PHASE_COLORS = {
    "recon": "#1f77b4",
    "weapon": "#ff7f0e",
    "delivery": "#2ca02c",
    "exploit": "#d62728",
    "install": "#9467bd",
    "c2": "#8c564b",
    "objectives": "#e377c2",
}


def _phase_color(phase):
    return PHASE_COLORS.get(phase, "#7f7f7f")


def _lighten_color(hex_color, factor=0.6):
    """Lighten the given hex color by blending toward white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def visualize_kill_chain_graph(
    predicted_dict, edges, technique_df=None, best_path=None
):
    """
    Visualize kill chain graph.
    - Normal edges = grey
    - Best similarity edges (per phase) = orange
    - Best path edges (Path 1) = pink/red
    """
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#000000",  # black background
        font_color="white",
    )

    # Phase ordering for layout
    phase_positions = {
        "recon": 0,
        "weapon": 1,
        "delivery": 2,
        "exploit": 3,
        "install": 4,
        "c2": 5,
        "objectives": 6,
    }
    x_spacing, y_spacing = 300, 120

    # Map Name → ID for hover tooltips
    name_to_id = {}
    if technique_df is not None:
        if "Name" in technique_df.columns and "ID" in technique_df.columns:
            name_to_id = {row["Name"]: row["ID"] for _, row in technique_df.iterrows()}

    # --- Add technique nodes
    phase_counts = {}
    phase_top_nodes = {}
    for phase, techniques in predicted_dict.items():
        if not techniques:
            continue

        x = phase_positions.get(phase, len(phase_positions)) * x_spacing
        phase_counts[phase] = 0

        # Find the max-prob technique in this phase
        top_tech, top_prob = max(techniques, key=lambda t: t[1])
        base_color = _phase_color(phase)
        light_color = _lighten_color(base_color, factor=0.6)

        for tech_name, score in techniques:
            y = phase_counts[phase] * y_spacing
            node_id = f"{phase}:{tech_name}"
            tech_id = name_to_id.get(tech_name, "N/A")

            title_text = (
                f"Technique: {tech_name} | Phase: {phase} | "
                f"Prob: {score:.4f} | ID: {tech_id}"
            )

            node_color = base_color if tech_name == top_tech else light_color

            net.add_node(
                node_id,
                label=tech_name if len(tech_name) <= 28 else tech_name[:25] + "...",
                title=title_text,
                color=node_color,
                x=x,
                y=y,
                size=25,
                physics=False,
                font={"size": 14, "face": "Arial"},
            )

            if phase_counts[phase] == 0:
                phase_top_nodes[phase] = node_id

            phase_counts[phase] += 1

    # --- Add header nodes
    for phase, col in phase_positions.items():
        x = col * x_spacing
        y = -200
        header_id = f"header:{phase}"
        net.add_node(
            header_id,
            label=phase.capitalize(),
            shape="box",
            color="#372041",
            borderWidth=3,
            font={"size": 32, "face": "Arial", "bold": True, "color": "white"},
            x=x,
            y=y,
            physics=False,
            fixed=True,
            widthConstraint={"minimum": 180, "maximum": 220},
            heightConstraint={"minimum": 60},
        )
        if phase in phase_top_nodes:
            net.add_edge(
                header_id,
                phase_top_nodes[phase],
                color="#000000",
                width=0.1,
                hidden=True,
            )

    # --- Collect best path edges
    best_path_edges = set()
    if best_path:
        for i in range(len(best_path) - 1):
            src_phase, src_tech, _, _ = best_path[i]
            tgt_phase, tgt_tech, _, _ = best_path[i + 1]
            best_path_edges.add((src_phase, src_tech, tgt_phase, tgt_tech))

    # --- Identify best edges per phase pair
    best_edges = {}
    for src_phase, src_tech, tgt_phase, tgt_tech, sim in edges:
        key = (src_phase, tgt_phase)
        if key not in best_edges or sim > best_edges[key][2]:
            best_edges[key] = (src_tech, tgt_tech, sim)

    # --- Add edges
    for src_phase, src_tech, tgt_phase, tgt_tech, sim in edges:
        src_id = f"{src_phase}:{src_tech}"
        tgt_id = f"{tgt_phase}:{tgt_tech}"

        if src_id in net.node_ids and tgt_id in net.node_ids:
            if (src_phase, src_tech, tgt_phase, tgt_tech) in best_path_edges:
                color = "#ff69b4"  # pink for best path edges
                width = 4
            elif (
                (src_phase, tgt_phase) in best_edges
                and best_edges[(src_phase, tgt_phase)][0] == src_tech
                and best_edges[(src_phase, tgt_phase)][1] == tgt_tech
            ):
                color = "#ffa500"  # orange for best similarity edge
                width = 3
            else:
                color = "#bdbdbd"  # grey for others
                width = 2

            net.add_edge(
                src_id,
                tgt_id,
                title=f"Cosine Sim: {sim:.3f}",
                color=color,
                width=width,
                smooth={"type": "continuous"},
            )

    # --- Styling
    net.set_options(
        """
        var options = {
          "nodes": {
            "font": {"size": 14, "face": "Arial", "color": "#ffffff"},
            "borderWidth": 2,
            "borderWidthSelected": 3
          },
          "edges": {
            "smooth": {"type": "continuous", "roundness": 0.2},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {"enabled": false}
        }
        """
    )

    net.save_graph("kill_chain_graph.html")
    components.html(
        open("kill_chain_graph.html", "r", encoding="utf-8").read(),
        height=800,
        scrolling=True,
    )
