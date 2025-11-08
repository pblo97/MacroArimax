"""
visualization.py
Interactive visualization for liquidity graphs.
"""

import plotly.graph_objects as go
import networkx as nx
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple

if TYPE_CHECKING:
    from macro_plumbing.graph.graph_builder import LiquidityGraph
    from macro_plumbing.graph.enhanced_graph_builder import EnhancedLiquidityGraph, EnhancedGraphMetrics


def create_interactive_graph_plotly(graph: "LiquidityGraph"):
    """
    Create interactive Plotly visualization of liquidity graph.

    Parameters
    ----------
    graph : LiquidityGraph
        The liquidity graph to visualize

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive graph figure
    """
    # Get node positions using spring layout
    pos = nx.spring_layout(graph.G, k=2, iterations=50, seed=42)

    # Extract node and edge data
    edge_traces = []

    # Create edge traces
    for edge in graph.G.edges(data=True):
        source, target, attrs = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        flow = attrs.get('flow', 0)
        is_drain = attrs.get('is_drain', False)
        driver = attrs.get('driver', '')
        z_score = attrs.get('z_score', 0)

        # Color and width based on flow
        edge_color = 'red' if is_drain else 'green'
        edge_width = min(abs(flow) / 50, 10)  # Scale width, cap at 10

        # Arrow annotation
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_width,
                color=edge_color,
            ),
            hoverinfo='text',
            hovertext=f"{source} → {target}<br>Flow: {flow:.1f}B<br>Z-score: {z_score:.2f}<br>{driver}",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    # Color map for node types
    color_map = {
        'fed': 'lightblue',
        'treasury': 'orange',
        'bank': 'lightgreen',
        'mmf': 'pink',
    }

    for node in graph.G.nodes(data=True):
        name, attrs = node
        x, y = pos[name]
        node_x.append(x)
        node_y.append(y)

        # Node info for hover
        balance = attrs.get('balance', 0)
        delta_1d = attrs.get('delta_1d', 0)
        delta_5d = attrs.get('delta_5d', 0)
        z_score = attrs.get('z_score', 0)
        percentile = attrs.get('percentile', 0)
        node_type = attrs.get('type', 'unknown')

        hover_text = (
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"Balance: ${balance:.1f}B<br>"
            f"Δ1D: ${delta_1d:.1f}B<br>"
            f"Δ5D: ${delta_5d:.1f}B<br>"
            f"Z-score: {z_score:.2f}<br>"
            f"Percentile: {percentile:.1%}"
        )
        node_text.append(hover_text)

        # Size based on balance
        node_sizes.append(max(abs(balance) / 50, 10))  # Scale size
        node_colors.append(color_map.get(node_type, 'gray'))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[node[0] for node in graph.G.nodes(data=True)],
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='black'),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title={
            'text': "Grafo de Flujos de Liquidez<br><sub>Rojo = Drenaje | Verde = Inyección | Tamaño = Balance</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600,
    )

    return fig


def create_enhanced_graph_plotly(
    graph: "EnhancedLiquidityGraph",
    metrics: "EnhancedGraphMetrics"
):
    """
    Create enhanced interactive visualization with all 4 phases.

    Features:
    - Node colors by SIM score (systemically important)
    - Vulnerable nodes highlighted with red border
    - NBFI nodes clearly differentiated
    - Edge colors by type (margin calls, NBFI flows, traditional)
    - Enhanced hover text with SIM, LCR, NBFI stress
    - Interactive legend

    Parameters
    ----------
    graph : EnhancedLiquidityGraph
        Enhanced liquidity graph
    metrics : EnhancedGraphMetrics
        Enhanced metrics from all 4 phases

    Returns
    -------
    plotly.graph_objects.Figure
        Enhanced interactive graph
    """
    # Get node positions using hierarchical layout for better NBFI visualization
    # Put core nodes (Fed, Treasury, Banks) in center, NBFI on periphery
    pos = _compute_enhanced_layout(graph.G)

    # Extract vulnerable nodes for highlighting
    vulnerable_set = {node for node, _, _, _ in metrics.vulnerable_nodes}

    # Create edge traces with type-based coloring
    edge_traces = []
    edge_types_seen = set()

    for edge in graph.G.edges(data=True):
        source, target, attrs = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        flow = attrs.get('flow', 0)
        is_drain = attrs.get('is_drain', False)
        driver = attrs.get('driver', '')
        z_score = attrs.get('z_score', 0)
        edge_family = attrs.get('family', 'unknown')

        # Determine edge type and color
        if 'Margin' in driver or 'IM' in driver or 'VM' in driver:
            edge_color = 'purple'
            edge_type = 'Margin Call'
        elif any(nbfi in source or nbfi in target for nbfi in ['Hedge_Funds', 'Asset_Managers', 'Insurance_Pensions']):
            edge_color = 'dodgerblue'
            edge_type = 'NBFI Flow'
        elif is_drain:
            edge_color = 'red'
            edge_type = 'Drenaje'
        else:
            edge_color = 'green'
            edge_type = 'Inyección'

        edge_types_seen.add(edge_type)

        # Width based on flow magnitude
        edge_width = min(abs(flow) / 50 + 2, 12)

        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_width,
                color=edge_color,
            ),
            hoverinfo='text',
            hovertext=(
                f"<b>{source} → {target}</b><br>"
                f"Type: {edge_type}<br>"
                f"Flow: ${flow:.1f}B<br>"
                f"Z-score: {z_score:.2f}<br>"
                f"Driver: {driver}<br>"
                f"Family: {edge_family}"
            ),
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node traces with SIM-based coloring
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_border_colors = []
    node_border_widths = []

    # Node type categories for hover
    nbfi_nodes = {'Hedge_Funds', 'Asset_Managers', 'Insurance_Pensions'}

    for node in graph.G.nodes(data=True):
        name, attrs = node
        x, y = pos[name]
        node_x.append(x)
        node_y.append(y)

        # Get node metrics
        balance = attrs.get('balance', 0)
        delta_1d = attrs.get('delta_1d', 0)
        delta_5d = attrs.get('delta_5d', 0)
        z_score = attrs.get('z_score', 0)
        percentile = attrs.get('percentile', 0)
        node_type = attrs.get('type', 'unknown')
        stress_prob = attrs.get('stress_prob', 0.5)

        # Get SIM and LCR from metrics
        sim_score = metrics.sim_scores.get(name, 0.0)
        lcr = metrics.lcr_scores.get(name, float('inf'))

        # Determine node color based on SIM score
        if sim_score > 0.5:
            # Systemically important - red shades
            node_color = f'rgb({int(255 * sim_score)}, 0, 0)'
            importance = '🔴 SIFI'
        elif sim_score > 0.3:
            # Moderately important - orange
            node_color = f'rgb(255, {int(165 * (1 - sim_score))}, 0)'
            importance = '🟡 Moderately Important'
        else:
            # Normal - green shades
            node_color = f'rgb(0, {int(200 * (1 - sim_score))}, 0)'
            importance = '🟢 Normal'

        # Special handling for NBFI nodes
        if name in nbfi_nodes:
            if name == 'Hedge_Funds':
                nbfi_info = f"<br><b>NBFI: Hedge Funds</b><br>Stress: {metrics.hedge_fund_stress:.1%}"
            elif name == 'Asset_Managers':
                nbfi_info = f"<br><b>NBFI: Asset Managers</b><br>Stress: {metrics.asset_manager_stress:.1%}"
            elif name == 'Insurance_Pensions':
                nbfi_info = f"<br><b>NBFI: Insurance/Pensions</b><br>Stress: {metrics.insurance_stress:.1%}"
            else:
                nbfi_info = ""
        else:
            nbfi_info = ""

        # Build hover text
        lcr_display = f"{lcr:.2f}" if lcr < 100 else "∞"
        hover_text = (
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"Balance: ${balance:.1f}B<br>"
            f"Δ1D: ${delta_1d:.1f}B<br>"
            f"Δ5D: ${delta_5d:.1f}B<br>"
            f"Z-score: {z_score:.2f}<br>"
            f"Percentile: {percentile:.1%}<br>"
            f"<br><b>Phase 4 Metrics:</b><br>"
            f"SIM Score: {sim_score:.3f} ({importance})<br>"
            f"Network LCR: {lcr_display}"
            f"{nbfi_info}"
        )
        node_text.append(hover_text)

        # Size based on balance
        node_sizes.append(max(abs(balance) / 30 + 15, 20))

        # Border: Red and thick if vulnerable
        if name in vulnerable_set:
            node_border_colors.append('darkred')
            node_border_widths.append(6)
        else:
            node_border_colors.append('black')
            node_border_widths.append(2)

        node_colors.append(node_color)

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[node[0] for node in graph.G.nodes(data=True)],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(
                width=node_border_widths,
                color=node_border_colors,
            ),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Build legend text
    legend_parts = [
        "<b>Enhanced Liquidity Network (4 Phases)</b>",
        "",
        "<b>Node Colors (SIM Score):</b>",
        "🔴 Red: Systemically Important (SIM > 0.5)",
        "🟡 Orange: Moderately Important (0.3 < SIM < 0.5)",
        "🟢 Green: Normal (SIM < 0.3)",
        "",
        "<b>Node Borders:</b>",
        "⚠️ Thick Red: Vulnerable (SIFI + Low LCR)",
        "",
        "<b>Edge Colors:</b>",
        "🟣 Purple: Margin Calls (Phase 1)",
        "🔵 Blue: NBFI Flows (Phase 2)",
        "🔴 Red: Draining",
        "🟢 Green: Injecting",
        "",
        "<b>Metrics:</b>",
        f"Margin Stress: {metrics.margin_stress_index:.2f}",
        f"NBFI Systemic: {metrics.nbfi_systemic_score:.2f}",
        f"Network Density: {metrics.density:.1%}",
        f"Contagion Index: {metrics.contagion_index:.1f}",
        f"Resilience: {metrics.network_resilience:.1%}",
    ]

    if vulnerable_set:
        legend_parts.extend([
            "",
            f"⚠️ {len(vulnerable_set)} Vulnerable Nodes Detected"
        ])

    legend_text = "<br>".join(legend_parts)

    fig.update_layout(
        title={
            'text': "Enhanced Liquidity Network Graph<br><sub>Comprehensive Analysis (4 Phases) - Hover for Details</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=700,
        annotations=[
            dict(
                text=legend_text,
                xref="paper", yref="paper",
                x=1.02, y=0.5,
                xanchor="left", yanchor="middle",
                showarrow=False,
                font=dict(size=9, family="monospace"),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=10,
            )
        ]
    )

    return fig


def _compute_enhanced_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Compute hierarchical layout for enhanced graph.

    Core nodes (Fed, Treasury, Banks) in center.
    NBFI nodes on periphery.
    Other nodes distributed around.

    Parameters
    ----------
    G : nx.DiGraph
        Graph to layout

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Node positions {node_name: (x, y)}
    """
    import math

    # Define node categories
    core_nodes = {'Fed', 'Treasury', 'Banks'}
    nbfi_nodes = {'Hedge_Funds', 'Asset_Managers', 'Insurance_Pensions'}

    # Manual positions for better visualization
    pos = {}

    # Core nodes in center triangle
    if 'Fed' in G.nodes():
        pos['Fed'] = (0, 1)
    if 'Treasury' in G.nodes():
        pos['Treasury'] = (-0.5, 0)
    if 'Banks' in G.nodes():
        pos['Banks'] = (0.5, 0)

    # NBFI nodes on outer circle
    nbfi_present = [n for n in nbfi_nodes if n in G.nodes()]
    nbfi_count = len(nbfi_present)
    for i, node in enumerate(nbfi_present):
        angle = 2 * math.pi * i / nbfi_count + math.pi / 2
        pos[node] = (2 * math.cos(angle), 2 * math.sin(angle))

    # Other nodes on middle circle
    other_nodes = [n for n in G.nodes() if n not in core_nodes and n not in nbfi_nodes]
    other_count = len(other_nodes)
    for i, node in enumerate(other_nodes):
        angle = 2 * math.pi * i / max(other_count, 1)
        pos[node] = (1.3 * math.cos(angle), 1.3 * math.sin(angle))

    return pos
