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


def _build_name_to_id(technique_df):
    """Build Name → ID mapping from technique dataframe."""
    if technique_df is None:
        return {}
    if "Name" in technique_df.columns and "ID" in technique_df.columns:
        return {row["Name"]: row["ID"] for _, row in technique_df.iterrows()}
    return {}


def _add_technique_nodes(net, predicted_dict, phase_positions, name_to_id, x_spacing, y_spacing):
    """Add technique nodes to the network graph. Returns phase_top_nodes mapping."""
    phase_top_nodes = {}
    for phase, techniques in predicted_dict.items():
        if not techniques:
            continue

        x = phase_positions.get(phase, len(phase_positions)) * x_spacing
        top_tech = max(techniques, key=lambda t: t[1])[0]
        base_color = _phase_color(phase)
        light_color = _lighten_color(base_color, factor=0.6)

        for idx, (tech_name, score) in enumerate(techniques):
            node_id = f"{phase}:{tech_name}"
            tech_id = name_to_id.get(tech_name, "N/A")
            net.add_node(
                node_id,
                label=tech_name if len(tech_name) <= 28 else tech_name[:25] + "...",
                title=f"Technique: {tech_name} | Phase: {phase} | Prob: {score:.4f} | ID: {tech_id}",
                color=base_color if tech_name == top_tech else light_color,
                x=x, y=idx * y_spacing, size=25, physics=False,
                font={"size": 14, "face": "Arial"},
            )
            if idx == 0:
                phase_top_nodes[phase] = node_id

    return phase_top_nodes


def _add_header_nodes(net, phase_positions, phase_top_nodes, x_spacing):
    """Add phase header nodes and invisible edges to top technique nodes."""
    for phase, col in phase_positions.items():
        header_id = f"header:{phase}"
        net.add_node(
            header_id, label=phase.capitalize(), shape="box", color="#372041",
            borderWidth=3,
            font={"size": 32, "face": "Arial", "bold": True, "color": "white"},
            x=col * x_spacing, y=-200, physics=False, fixed=True,
            widthConstraint={"minimum": 180, "maximum": 220},
            heightConstraint={"minimum": 60},
        )
        if phase in phase_top_nodes:
            net.add_edge(header_id, phase_top_nodes[phase], color="#000000", width=0.1, hidden=True)


def _collect_best_path_edges(best_path):
    """Collect set of (src_phase, src_tech, tgt_phase, tgt_tech) tuples from best path."""
    result = set()
    if not best_path:
        return result
    for i in range(len(best_path) - 1):
        src_phase, src_tech, _, _ = best_path[i]
        tgt_phase, tgt_tech, _, _ = best_path[i + 1]
        result.add((src_phase, src_tech, tgt_phase, tgt_tech))
    return result


def _find_best_edges_per_pair(edges):
    """Find the highest-similarity edge for each phase pair."""
    best = {}
    for src_phase, src_tech, tgt_phase, tgt_tech, sim in edges:
        key = (src_phase, tgt_phase)
        if key not in best or sim > best[key][2]:
            best[key] = (src_tech, tgt_tech, sim)
    return best


def _resolve_edge_style(src_phase, src_tech, tgt_phase, tgt_tech, best_path_edges, best_edges):
    """Determine edge color and width based on path/similarity status."""
    if (src_phase, src_tech, tgt_phase, tgt_tech) in best_path_edges:
        return "#ff69b4", 4  # pink for best path
    pair = best_edges.get((src_phase, tgt_phase))
    if pair and pair[0] == src_tech and pair[1] == tgt_tech:
        return "#ffa500", 3  # orange for best similarity
    return "#bdbdbd", 2  # grey for others


def _add_edges(net, edges, best_path_edges, best_edges):
    """Add all similarity edges to the network with appropriate styling."""
    for src_phase, src_tech, tgt_phase, tgt_tech, sim in edges:
        src_id = f"{src_phase}:{src_tech}"
        tgt_id = f"{tgt_phase}:{tgt_tech}"
        if src_id not in net.node_ids or tgt_id not in net.node_ids:
            continue
        color, width = _resolve_edge_style(src_phase, src_tech, tgt_phase, tgt_tech, best_path_edges, best_edges)
        net.add_edge(src_id, tgt_id, title=f"Cosine Sim: {sim:.3f}", color=color, width=width, smooth={"type": "continuous"})


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
        bgcolor="#000000",
        font_color="white",
    )

    phase_positions = {
        "recon": 0, "weapon": 1, "delivery": 2, "exploit": 3,
        "install": 4, "c2": 5, "objectives": 6,
    }
    x_spacing, y_spacing = 300, 120

    name_to_id = _build_name_to_id(technique_df)
    phase_top_nodes = _add_technique_nodes(net, predicted_dict, phase_positions, name_to_id, x_spacing, y_spacing)
    _add_header_nodes(net, phase_positions, phase_top_nodes, x_spacing)

    best_path_edges = _collect_best_path_edges(best_path)
    best_edges = _find_best_edges_per_pair(edges)
    _add_edges(net, edges, best_path_edges, best_edges)

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
