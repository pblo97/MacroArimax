"""
visualization.py
Interactive visualization for liquidity graphs.
"""

import plotly.graph_objects as go
import networkx as nx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macro_plumbing.graph.graph_builder import LiquidityGraph


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
