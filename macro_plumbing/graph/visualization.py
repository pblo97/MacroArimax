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
    # Get node positions using spring layout with more spacing
    pos = nx.spring_layout(graph.G, k=3, iterations=100, seed=42)

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
        edge_width = min(abs(flow) / 100 + 1, 5)  # Scale width, cap at 5

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

        # Size based on balance (use log scale to prevent giant nodes)
        if abs(balance) > 0:
            import math
            size = 10 + 5 * math.log10(abs(balance) + 1)
            node_sizes.append(min(size, 30))  # Cap at 30 pixels
        else:
            node_sizes.append(12)  # Default size
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
        margin=dict(b=40, l=40, r=40, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        height=700,
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

    # Create edge traces with type-based coloring and contagion risk
    edge_traces = []
    edge_types_seen = set()

    # Calculate contagion contribution for each edge (Phase 4)
    edge_contagion_scores = {}
    for edge in graph.G.edges(data=True):
        source, target, attrs = edge
        source_stress = graph.G.nodes.get(source, {}).get('stress_prob', 0.5)
        flow = abs(attrs.get('flow', 0))
        # Simple contagion proxy: stress * flow
        edge_contagion_scores[(source, target)] = source_stress * flow

    max_contagion = max(edge_contagion_scores.values()) if edge_contagion_scores else 1

    for edge in graph.G.edges(data=True):
        source, target, attrs = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        flow = attrs.get('flow', 0)
        is_drain = attrs.get('is_drain', False)
        driver = attrs.get('driver', '')
        z_score = attrs.get('z_score', 0)
        edge_family = attrs.get('family', 'unknown')

        # Get contagion risk for this edge (Phase 4 metric)
        contagion_score = edge_contagion_scores.get((source, target), 0)
        contagion_pct = (contagion_score / max_contagion * 100) if max_contagion > 0 else 0

        # Determine edge type and color
        if 'Margin' in driver or 'IM' in driver or 'VM' in driver:
            edge_color = 'purple'
            edge_type = '🟣 Margin Call'
        elif any(nbfi in source or nbfi in target for nbfi in ['Hedge_Funds', 'Asset_Managers', 'Insurance_Pensions']):
            edge_color = 'dodgerblue'
            edge_type = '🔵 NBFI Flow'
        elif is_drain:
            edge_color = 'red'
            edge_type = '🔴 Drenaje'
        else:
            edge_color = 'green'
            edge_type = '🟢 Inyección'

        edge_types_seen.add(edge_type)

        # Width based on flow magnitude (reduced for cleaner look)
        base_width = min(abs(flow) / 100 + 1, 5)

        # Increase width for high-contagion edges (Phase 4 visual)
        if contagion_pct > 70:
            edge_width = base_width * 1.5
            edge_dash = 'solid'
        elif contagion_pct > 40:
            edge_width = base_width * 1.2
            edge_dash = 'solid'
        else:
            edge_width = base_width
            edge_dash = 'solid'

        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_width,
                color=edge_color,
                dash=edge_dash,
            ),
            hoverinfo='text',
            hovertext=(
                f"<b>{source} → {target}</b><br>"
                f"Type: {edge_type}<br>"
                f"Flow: ${flow:.1f}B<br>"
                f"Z-score: {z_score:.2f}<br>"
                f"Driver: {driver}<br>"
                f"Family: {edge_family}<br>"
                f"<br><b>🔴 PHASE 4:</b><br>"
                f"Contagion Risk: {contagion_pct:.1f}%"
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

        # Determine node color based on SIM score with enhanced visibility
        if sim_score > 0.7:
            # Critical SIFI - Dark Red
            node_color = 'rgb(200, 0, 0)'
            importance = '🔴 CRITICAL SIFI'
        elif sim_score > 0.5:
            # High SIFI - Red
            node_color = 'rgb(255, 50, 50)'
            importance = '🔴 HIGH SIFI'
        elif sim_score > 0.3:
            # Moderately important - Orange
            node_color = 'rgb(255, 150, 0)'
            importance = '🟡 MODERATE'
        elif sim_score > 0.15:
            # Low importance - Yellow-Green
            node_color = 'rgb(200, 200, 0)'
            importance = '🟡 LOW'
        else:
            # Normal - Green
            node_color = 'rgb(50, 200, 50)'
            importance = '🟢 NORMAL'

        # For NBFI nodes, show real AUM (balance is scaled for visualization)
        NBFI_SCALE_FACTOR = 0.1
        if name in nbfi_nodes:
            real_aum = balance / NBFI_SCALE_FACTOR
            balance_display = f"${real_aum:.1f}B (AUM)"

            if name == 'Hedge_Funds':
                nbfi_info = f"<br><b>NBFI: Hedge Funds</b><br>Stress: {metrics.hedge_fund_stress:.1%}"
            elif name == 'Asset_Managers':
                nbfi_info = f"<br><b>NBFI: Asset Managers</b><br>Stress: {metrics.asset_manager_stress:.1%}"
            elif name == 'Insurance_Pensions':
                nbfi_info = f"<br><b>NBFI: Insurance/Pensions</b><br>Stress: {metrics.insurance_stress:.1%}"
            else:
                nbfi_info = ""
        else:
            balance_display = f"${balance:.1f}B"
            nbfi_info = ""

        # Build hover text
        lcr_display = f"{lcr:.2f}" if lcr < 100 else "∞"
        hover_text = (
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"Balance: {balance_display}<br>"
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

        # Size based on balance (reduced to prevent overlap)
        # Use logarithmic scale for very large balances
        if abs(balance) > 0:
            import math
            # Log scale for large values to prevent giant nodes
            # More aggressive scaling to prevent overlap
            size = 8 + 3 * math.log10(abs(balance) + 1)
            node_sizes.append(min(size, 25))  # Cap at 25 pixels (reduced from 35)
        else:
            node_sizes.append(12)  # Default size for zero balance

        # Border: Red and thick if vulnerable
        if name in vulnerable_set:
            node_border_colors.append('darkred')
            node_border_widths.append(6)
        else:
            node_border_colors.append('black')
            node_border_widths.append(2)

        node_colors.append(node_color)

    # Prepare node labels with vulnerability warnings
    node_labels = []
    for node_name in graph.G.nodes():
        label = node_name
        # Add warning symbol for vulnerable nodes
        if node_name in vulnerable_set:
            label = f"⚠️ {node_name}"
        node_labels.append(label)

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color='black', family='Arial', weight='bold'),
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

    # Build legend with enhanced Phase 4 section
    legend_parts = [
        "<b>Enhanced Liquidity Network</b>",
        "<b>(4-Phase Analysis)</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "<b>🔴 PHASE 4: ADVANCED METRICS</b>",
        "",
        "<b>Systemic Risk Scores:</b>",
        f"  • Contagion Index: {metrics.contagion_index:.2f}",
        f"  • Network Resilience: {metrics.network_resilience:.1%}",
        f"  • NBFI Systemic: {metrics.nbfi_systemic_score:+.2f}σ",
        "",
        "<b>Top SIFIs (by SIM):</b>",
    ]

    # Add top 3 SIFIs
    sorted_sim = sorted(metrics.sim_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for node, sim in sorted_sim:
        legend_parts.append(f"  🔴 {node}: {sim:.3f}")

    legend_parts.extend([
        "",
        "<b>Vulnerable Nodes:</b>",
    ])

    if vulnerable_set:
        legend_parts.append(f"  ⚠️ {len(vulnerable_set)} nodes at risk")
        for node, reason, lcr, sim in metrics.vulnerable_nodes[:2]:  # Top 2
            legend_parts.append(f"  • {node}: LCR={lcr:.2f}")
    else:
        legend_parts.append("  ✅ No vulnerable nodes")

    legend_parts.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>📊 SIM-BASED COLORS:</b>",
        "🔴 Dark Red: Critical (>0.7)",
        "🔴 Red: High SIFI (>0.5)",
        "🟡 Orange: Moderate (>0.3)",
        "🟡 Yellow: Low (>0.15)",
        "🟢 Green: Normal (<0.15)",
        "⚠️ + Red Border: Vulnerable",
        "",
        "<b>🔗 EDGE TYPES:</b>",
        "🟣 Margin Calls (Phase 1)",
        "🔵 NBFI Flows (Phase 2)",
        "🔴 Draining | 🟢 Injecting",
        "💥 Thicker = Higher Contagion",
        "",
        "<b>📈 NETWORK METRICS:</b>",
        f"Margin Stress: {metrics.margin_stress_index:+.2f}σ",
        f"Network Density: {metrics.density:.1%}",
    ])

    legend_text = "<br>".join(legend_parts)

    # Prepare annotations list
    annotations = [
        # Main legend
        dict(
            text=legend_text,
            xref="paper", yref="paper",
            x=1.02, y=0.5,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font=dict(size=9, family="monospace"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="red" if vulnerable_set else "black",
            borderwidth=2,
            borderpad=10,
        )
    ]

    # Add Phase 4 alert banner if there are vulnerable nodes
    if vulnerable_set:
        annotations.append(
            dict(
                text=f"<b>⚠️ PHASE 4 ALERT: {len(vulnerable_set)} VULNERABLE NODE(S)</b>",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                xanchor="center", yanchor="bottom",
                showarrow=False,
                font=dict(size=12, family="Arial", color="red"),
                bgcolor="rgba(255, 255, 0, 0.3)",
                bordercolor="red",
                borderwidth=2,
                borderpad=5,
            )
        )

    # Add contagion risk indicator
    contagion_color = "red" if metrics.contagion_index > 100 else "orange" if metrics.contagion_index > 50 else "green"
    contagion_status = "HIGH" if metrics.contagion_index > 100 else "MEDIUM" if metrics.contagion_index > 50 else "LOW"

    annotations.append(
        dict(
            text=f"<b>Contagion Risk: {contagion_status}</b><br>Index: {metrics.contagion_index:.1f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=10, family="monospace", color=contagion_color),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=contagion_color,
            borderwidth=2,
            borderpad=5,
        )
    )

    fig.update_layout(
        title={
            'text': "Enhanced Liquidity Network - PHASE 4 ANALYSIS<br><sub>🔴 SIM-based Risk | ⚠️ Vulnerable Nodes | 🟣 Margin Stress | 🔵 NBFI Flows</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16, family='Arial Black')
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40, l=40, r=240, t=120),  # More margin for legend and title
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        plot_bgcolor='rgba(250, 250, 250, 1)',  # Light grey background
        height=850,  # Taller graph
        annotations=annotations
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

    # Core nodes in center triangle (wider spacing)
    if 'Fed' in G.nodes():
        pos['Fed'] = (0, 1.5)
    if 'Treasury' in G.nodes():
        pos['Treasury'] = (-1.2, 0)
    if 'Banks' in G.nodes():
        pos['Banks'] = (1.2, 0)

    # NBFI nodes on outer circle (larger radius to avoid overlap)
    nbfi_present = [n for n in nbfi_nodes if n in G.nodes()]
    nbfi_count = len(nbfi_present)
    for i, node in enumerate(nbfi_present):
        angle = 2 * math.pi * i / nbfi_count + math.pi / 2
        pos[node] = (4 * math.cos(angle), 4 * math.sin(angle))

    # Other nodes on middle circle (larger radius)
    other_nodes = [n for n in G.nodes() if n not in core_nodes and n not in nbfi_nodes]
    other_count = len(other_nodes)
    for i, node in enumerate(other_nodes):
        angle = 2 * math.pi * i / max(other_count, 1)
        pos[node] = (2.5 * math.cos(angle), 2.5 * math.sin(angle))

    return pos
