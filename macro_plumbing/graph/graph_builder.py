"""
graph_builder.py
Build dynamic liquidity flow graph (who drains/injects to whom).
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Export public functions
__all__ = ['LiquidityGraph', 'GraphNode', 'build_liquidity_graph']


@dataclass
class GraphNode:
    """Liquidity graph node."""

    name: str
    type: str  # 'fed', 'treasury', 'bank', 'mmf', 'dealer', etc.
    balance: float
    delta_1d: float
    delta_5d: float
    z_score: float
    percentile: float


class LiquidityGraph:
    """Dynamic graph of liquidity flows."""

    def __init__(self):
        """Initialize graph."""
        self.G = nx.DiGraph()
        self.node_data = {}

    def add_node(
        self,
        name: str,
        node_type: str,
        balance: float = 0.0,
        delta_1d: float = 0.0,
        delta_5d: float = 0.0,
        z_score: float = 0.0,
        percentile: float = 0.5,
    ):
        """Add node to graph."""
        self.G.add_node(
            name,
            type=node_type,
            balance=balance,
            delta_1d=delta_1d,
            delta_5d=delta_5d,
            z_score=z_score,
            percentile=percentile,
        )
        self.node_data[name] = GraphNode(
            name=name,
            type=node_type,
            balance=balance,
            delta_1d=delta_1d,
            delta_5d=delta_5d,
            z_score=z_score,
            percentile=percentile,
        )

    def add_edge(
        self,
        source: str,
        target: str,
        flow: float,
        z_score: float = 0.0,
        driver: str = "",
    ):
        """
        Add directed edge (flow).

        Parameters
        ----------
        source : str
            Source node (who is drained or injecting)
        target : str
            Target node (who receives or drains)
        flow : float
            Flow amount (positive = injection, negative = drain)
        z_score : float
            Z-score of flow
        driver : str
            Description of what's driving the flow
        """
        self.G.add_edge(
            source,
            target,
            flow=flow,
            z_score=z_score,
            driver=driver,
            abs_flow=abs(flow),
            is_drain=flow < 0,
        )

    def get_node_attributes(self, attr: str) -> Dict:
        """Get attribute dict for all nodes."""
        return nx.get_node_attributes(self.G, attr)

    def get_edge_attributes(self, attr: str) -> Dict:
        """Get attribute dict for all edges."""
        return nx.get_edge_attributes(self.G, attr)

    def get_sinks(self, top_n: int = 5) -> List[str]:
        """Get top liquidity sinks (nodes with most inward negative flow)."""
        in_flows = {}
        for node in self.G.nodes():
            in_flow = sum(
                [self.G[u][node].get("flow", 0) for u in self.G.predecessors(node)]
            )
            in_flows[node] = in_flow

        sorted_sinks = sorted(in_flows.items(), key=lambda x: x[1])
        return [node for node, _ in sorted_sinks[:top_n]]

    def get_sources(self, top_n: int = 5) -> List[str]:
        """Get top liquidity sources (nodes with most outward positive flow)."""
        out_flows = {}
        for node in self.G.nodes():
            out_flow = sum(
                [self.G[node][v].get("flow", 0) for v in self.G.successors(node)]
            )
            out_flows[node] = out_flow

        sorted_sources = sorted(out_flows.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_sources[:top_n]]

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export to DataFrames.

        Returns
        -------
        tuple
            (nodes_df, edges_df)
        """
        # Nodes
        nodes = []
        for node, attrs in self.G.nodes(data=True):
            nodes.append({"node": node, **attrs})
        nodes_df = pd.DataFrame(nodes)

        # Edges
        edges = []
        for u, v, attrs in self.G.edges(data=True):
            edges.append({"source": u, "target": v, **attrs})
        edges_df = pd.DataFrame(edges)

        return nodes_df, edges_df


def build_liquidity_graph(df: pd.DataFrame) -> LiquidityGraph:
    """
    Build graph from data.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: RESERVES, TGA, RRP, etc.

    Returns
    -------
    LiquidityGraph
    """
    graph = LiquidityGraph()

    # Get latest values
    latest = df.iloc[-1]

    # Calculate z-scores (simple version)
    z_window = min(252, len(df))
    z_scores = (df - df.rolling(z_window).mean()) / df.rolling(z_window).std()
    latest_z = z_scores.iloc[-1]

    # Percentiles
    percentiles = df.rank(pct=True).iloc[-1]

    # Add nodes
    node_configs = {
        "Fed": ("RESERVES", "fed"),
        "Treasury (TGA)": ("TGA", "treasury"),
        "ON RRP": ("RRP", "fed"),
        "Banks": ("RESERVES", "bank"),
        "MMFs": ("RRP", "mmf"),
    }

    for node_name, (col, node_type) in node_configs.items():
        if col in df.columns:
            graph.add_node(
                name=node_name,
                node_type=node_type,
                balance=latest[col],
                delta_1d=df[col].diff().iloc[-1],
                delta_5d=df[col].diff(5).iloc[-1],
                z_score=latest_z.get(col, 0),
                percentile=percentiles.get(col, 0.5),
            )

    # Add edges (simplified flow logic)
    if "RESERVES" in df.columns and "TGA" in df.columns:
        # TGA drain/injection to system
        tga_delta = df["TGA"].diff().iloc[-1]
        z_tga = latest_z.get("TGA", 0)

        if tga_delta > 0:
            # TGA rising = drain from system
            graph.add_edge(
                "Banks",
                "Treasury (TGA)",
                flow=-tga_delta,
                z_score=z_tga,
                driver="TGA accumulation drains reserves",
            )
        else:
            # TGA falling = injection to system
            graph.add_edge(
                "Treasury (TGA)",
                "Banks",
                flow=-tga_delta,
                z_score=z_tga,
                driver="TGA spend injects reserves",
            )

    if "RRP" in df.columns:
        # RRP usage
        rrp_delta = df["RRP"].diff().iloc[-1]
        z_rrp = latest_z.get("RRP", 0)

        if rrp_delta > 0:
            # RRP rising = drain from system
            graph.add_edge(
                "MMFs",
                "Fed",
                flow=rrp_delta,
                z_score=z_rrp,
                driver="MMFs parking cash at Fed RRP",
            )
        else:
            # RRP falling = injection to system
            graph.add_edge(
                "Fed",
                "MMFs",
                flow=-rrp_delta,
                z_score=z_rrp,
                driver="RRP runoff injects liquidity",
            )

    return graph


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    df = pd.DataFrame(
        {
            "RESERVES": 3000 + np.cumsum(np.random.randn(len(dates)) * 10),
            "TGA": 500 + np.cumsum(np.random.randn(len(dates)) * 5),
            "RRP": 1000 + np.cumsum(np.random.randn(len(dates)) * 8),
        },
        index=dates,
    )

    # Build graph
    graph = build_liquidity_graph(df)

    print(f"Nodes: {list(graph.G.nodes())}")
    print(f"Edges: {list(graph.G.edges())}")
    print(f"\nTop sinks: {graph.get_sinks(3)}")
    print(f"Top sources: {graph.get_sources(3)}")

    # Export
    nodes_df, edges_df = graph.to_dataframe()
    print("\nNodes:")
    print(nodes_df)
    print("\nEdges:")
    print(edges_df)
