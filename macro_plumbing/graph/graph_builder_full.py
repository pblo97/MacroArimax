"""
graph_builder_full.py
Complete liquidity flow graph with all plumbing channels.

Nodes: Fed, Treasury (TGA), ON RRP, Banks, MMFs, Dealers/PDs, FHLB, UST Market, Credit (HY)
Edges: Reserves, RRP, TGA, Repo-GC, Unsecured, FHLB, UST liquidity, Credit conditions
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


def zscore_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=30).mean()
    std = series.rolling(window, min_periods=30).std()
    return (series - mean) / std.replace(0, np.nan)


def detect_quarter_end(dates: pd.DatetimeIndex, window: int = 3) -> pd.Series:
    """Detect quarter-end periods (±window business days)."""
    is_qtr_end = pd.Series(False, index=dates)

    for date in dates:
        # Check if within window of quarter end
        quarter_ends = pd.date_range(
            start=date - pd.Timedelta(days=window*2),
            end=date + pd.Timedelta(days=window*2),
            freq='Q'
        )
        if len(quarter_ends) > 0:
            closest = min(quarter_ends, key=lambda x: abs((x - date).days))
            if abs((closest - date).days) <= window:
                is_qtr_end[date] = True

    return is_qtr_end


@dataclass
class GraphNode:
    """Extended liquidity graph node."""
    name: str
    type: str
    balance: float
    delta_1d: float
    delta_5d: float
    z_score: float
    percentile: float
    stress_prob: float = 0.5  # From Markov dynamics


@dataclass
class GraphEdge:
    """Extended liquidity graph edge."""
    source: str
    target: str
    flow: float
    driver: str
    z_score: float
    is_drain: bool
    weight: float


class CompleteLiquidityGraph:
    """Complete liquidity flow graph with all channels."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.nodes_data = {}
        self.edges_data = []
        self.reserve_identity = {}
        self.stress_flow_index = 0.0
        self.hotspots = []

    def add_node_data(self, node: GraphNode):
        """Add node with full data."""
        self.nodes_data[node.name] = node
        self.G.add_node(
            node.name,
            type=node.type,
            balance=node.balance,
            delta_1d=node.delta_1d,
            delta_5d=node.delta_5d,
            z_score=node.z_score,
            percentile=node.percentile,
            stress_prob=node.stress_prob
        )

    def add_edge_data(self, edge: GraphEdge):
        """Add edge with full data."""
        self.edges_data.append(edge)
        self.G.add_edge(
            edge.source,
            edge.target,
            flow=edge.flow,
            driver=edge.driver,
            z_score=edge.z_score,
            is_drain=edge.is_drain,
            weight=edge.weight,
            abs_flow=abs(edge.flow)
        )

    def validate_reserve_identity(self, data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Validate reserve identity:
        ΔReserves ≈ -ΔTGA - ΔONRRP + ΔAssetsFedNet + error
        """
        df = pd.DataFrame({
            'delta_reserves': data.get('delta_reserves', 0),
            'delta_tga': data.get('delta_tga', 0),
            'delta_rrp': data.get('delta_rrp', 0),
            'delta_walcl': data.get('delta_walcl', 0)
        })

        # Simplified identity (missing some components)
        df['predicted_delta_reserves'] = -df['delta_tga'] - df['delta_rrp']
        df['residual'] = df['delta_reserves'] - df['predicted_delta_reserves']
        df['residual_pct'] = df['residual'] / df['delta_reserves'].abs().replace(0, np.nan)

        return df

    def compute_stress_flow_index(self) -> float:
        """
        Compute Stress Flow Index:
        Sum of (red edges - green edges) weighted by magnitude.
        """
        stress_flow = 0.0
        for edge in self.edges_data:
            contribution = edge.weight * (-1 if edge.is_drain else 1)
            stress_flow += contribution

        return stress_flow

    def identify_hotspots(self, threshold: float = 1.5) -> List[Tuple[str, str]]:
        """
        Identify hotspots: red edges with |z| > threshold.
        """
        hotspots = []
        for edge in self.edges_data:
            if edge.is_drain and abs(edge.z_score) > threshold:
                hotspots.append((edge.source, edge.target))

        return hotspots

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to DataFrames."""
        # Nodes
        nodes_list = []
        for name, node in self.nodes_data.items():
            nodes_list.append({
                'node': name,
                'type': node.type,
                'balance': node.balance,
                'delta_1d': node.delta_1d,
                'delta_5d': node.delta_5d,
                'z_score': node.z_score,
                'percentile': node.percentile,
                'stress_prob': node.stress_prob
            })
        nodes_df = pd.DataFrame(nodes_list)

        # Edges
        edges_list = []
        for edge in self.edges_data:
            edges_list.append({
                'source': edge.source,
                'target': edge.target,
                'flow': edge.flow,
                'driver': edge.driver,
                'z_score': edge.z_score,
                'is_drain': edge.is_drain,
                'weight': edge.weight
            })
        edges_df = pd.DataFrame(edges_list)

        return nodes_df, edges_df

    def get_sinks(self, top_n: int = 3) -> List[str]:
        """Get top draining nodes."""
        in_flow = {}
        for edge in self.edges_data:
            in_flow[edge.target] = in_flow.get(edge.target, 0) + edge.flow

        sorted_nodes = sorted(in_flow.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_nodes[:top_n]]

    def get_sources(self, top_n: int = 3) -> List[str]:
        """Get top injecting nodes."""
        out_flow = {}
        for edge in self.edges_data:
            out_flow[edge.source] = out_flow.get(edge.source, 0) + edge.flow

        sorted_nodes = sorted(out_flow.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_nodes[:top_n]]


def build_complete_liquidity_graph(df: pd.DataFrame, quarter_end_relax: bool = True) -> CompleteLiquidityGraph:
    """
    Build complete liquidity graph with all channels.

    Parameters
    ----------
    df : pd.DataFrame
        Data with all required series
    quarter_end_relax : bool
        Relax thresholds near quarter-end

    Returns
    -------
    CompleteLiquidityGraph
        Complete graph instance
    """
    graph = CompleteLiquidityGraph()

    # Detect quarter-end
    is_qtr_end = detect_quarter_end(df.index)
    qtr_multiplier = 0.7 if quarter_end_relax and is_qtr_end.iloc[-1] else 1.0

    # Latest date
    latest = df.index[-1]

    # ====================
    # NODES
    # ====================

    # 1. Fed
    reserves = df.get('RESERVES', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='Fed',
        type='fed',
        balance=reserves.iloc[-1] if len(reserves) > 0 else 0,
        delta_1d=reserves.diff().iloc[-1] if len(reserves) > 1 else 0,
        delta_5d=reserves.diff(5).iloc[-1] if len(reserves) > 5 else 0,
        z_score=zscore_rolling(reserves).iloc[-1] if len(reserves) > 30 else 0,
        percentile=reserves.rank(pct=True).iloc[-1] if len(reserves) > 0 else 0.5
    ))

    # 2. Treasury (TGA)
    tga = df.get('TGA', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='Treasury',
        type='treasury',
        balance=tga.iloc[-1] if len(tga) > 0 else 0,
        delta_1d=tga.diff().iloc[-1] if len(tga) > 1 else 0,
        delta_5d=tga.diff(5).iloc[-1] if len(tga) > 5 else 0,
        z_score=zscore_rolling(tga).iloc[-1] if len(tga) > 30 else 0,
        percentile=tga.rank(pct=True).iloc[-1] if len(tga) > 0 else 0.5
    ))

    # 3. ON RRP
    rrp = df.get('RRP', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='ON_RRP',
        type='facility',
        balance=rrp.iloc[-1] if len(rrp) > 0 else 0,
        delta_1d=rrp.diff().iloc[-1] if len(rrp) > 1 else 0,
        delta_5d=rrp.diff(5).iloc[-1] if len(rrp) > 5 else 0,
        z_score=zscore_rolling(rrp).iloc[-1] if len(rrp) > 30 else 0,
        percentile=rrp.rank(pct=True).iloc[-1] if len(rrp) > 0 else 0.5
    ))

    # 4. Banks
    bank_reserves = df.get('BANK_RESERVES_WEEKLY', df.get('RESERVES', pd.Series(0))).fillna(0)
    graph.add_node_data(GraphNode(
        name='Banks',
        type='bank',
        balance=bank_reserves.iloc[-1] if len(bank_reserves) > 0 else 0,
        delta_1d=bank_reserves.diff().iloc[-1] if len(bank_reserves) > 1 else 0,
        delta_5d=bank_reserves.diff(5).iloc[-1] if len(bank_reserves) > 5 else 0,
        z_score=zscore_rolling(bank_reserves).iloc[-1] if len(bank_reserves) > 30 else 0,
        percentile=bank_reserves.rank(pct=True).iloc[-1] if len(bank_reserves) > 0 else 0.5
    ))

    # 5. MMFs
    # Proxy: use RRP as MMFs are major users
    graph.add_node_data(GraphNode(
        name='MMFs',
        type='mmf',
        balance=rrp.iloc[-1] * 0.8 if len(rrp) > 0 else 0,  # ~80% of RRP from MMFs
        delta_1d=rrp.diff().iloc[-1] * 0.8 if len(rrp) > 1 else 0,
        delta_5d=rrp.diff(5).iloc[-1] * 0.8 if len(rrp) > 5 else 0,
        z_score=zscore_rolling(rrp).iloc[-1] if len(rrp) > 30 else 0,
        percentile=rrp.rank(pct=True).iloc[-1] if len(rrp) > 0 else 0.5
    ))

    # 6. Dealers/PDs
    # Proxy: use repo rate stress
    tgcr = df.get('TGCR', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='Dealers',
        type='dealer',
        balance=100,  # Normalized
        delta_1d=tgcr.diff().iloc[-1] if len(tgcr) > 1 else 0,
        delta_5d=tgcr.diff(5).iloc[-1] if len(tgcr) > 5 else 0,
        z_score=zscore_rolling(tgcr).iloc[-1] if len(tgcr) > 30 else 0,
        percentile=tgcr.rank(pct=True).iloc[-1] if len(tgcr) > 0 else 0.5
    ))

    # 7. FHLB
    fhlb = df.get('FHLB_ADVANCES', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='FHLB',
        type='gse',
        balance=fhlb.iloc[-1] if len(fhlb) > 0 else 0,
        delta_1d=fhlb.diff().iloc[-1] if len(fhlb) > 1 else 0,
        delta_5d=fhlb.diff(5).iloc[-1] if len(fhlb) > 5 else 0,
        z_score=zscore_rolling(fhlb).iloc[-1] if len(fhlb) > 30 else 0,
        percentile=fhlb.rank(pct=True).iloc[-1] if len(fhlb) > 0 else 0.5
    ))

    # 8. UST Market
    # Proxy: use MOVE index or bid-ask
    move = df.get('MOVE', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='UST_Market',
        type='market',
        balance=100,  # Normalized
        delta_1d=move.diff().iloc[-1] if len(move) > 1 else 0,
        delta_5d=move.diff(5).iloc[-1] if len(move) > 5 else 0,
        z_score=zscore_rolling(move).iloc[-1] if len(move) > 30 else 0,
        percentile=move.rank(pct=True).iloc[-1] if len(move) > 0 else 0.5
    ))

    # 9. Credit (HY)
    hy_oas = df.get('HY_OAS', pd.Series(0)).fillna(0)
    graph.add_node_data(GraphNode(
        name='Credit_HY',
        type='credit',
        balance=hy_oas.iloc[-1] if len(hy_oas) > 0 else 0,
        delta_1d=hy_oas.diff().iloc[-1] if len(hy_oas) > 1 else 0,
        delta_5d=hy_oas.diff(5).iloc[-1] if len(hy_oas) > 5 else 0,
        z_score=zscore_rolling(hy_oas).iloc[-1] if len(hy_oas) > 30 else 0,
        percentile=hy_oas.rank(pct=True).iloc[-1] if len(hy_oas) > 0 else 0.5
    ))

    # ====================
    # EDGES (with proper sign conventions)
    # ====================

    # 1. MMFs → ON RRP (driver: +ΔRRP = drain)
    delta_rrp = rrp.diff().fillna(0)
    w_rrp = -zscore_rolling(delta_rrp, 252).iloc[-1] if len(delta_rrp) > 30 else 0
    w_rrp = np.clip(w_rrp * qtr_multiplier, -5, 5)  # Clip to prevent explosions
    graph.add_edge_data(GraphEdge(
        source='MMFs',
        target='ON_RRP',
        flow=delta_rrp.iloc[-1] if len(delta_rrp) > 0 else 0,
        driver=f"ΔRRP={delta_rrp.iloc[-1]:.0f}B",
        z_score=w_rrp,
        is_drain=w_rrp < 0,
        weight=abs(w_rrp)
    ))

    # 2. Fed → Banks (driver: +ΔReserves = injection)
    delta_res = reserves.diff().fillna(0)
    w_res = zscore_rolling(delta_res, 252).iloc[-1] if len(delta_res) > 30 else 0
    w_res = np.clip(w_res * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Fed',
        target='Banks',
        flow=delta_res.iloc[-1] if len(delta_res) > 0 else 0,
        driver=f"ΔReserves={delta_res.iloc[-1]:.0f}B",
        z_score=w_res,
        is_drain=w_res < 0,
        weight=abs(w_res)
    ))

    # 3. Treasury → Banks (driver: +ΔTGA = drain from banks)
    delta_tga = tga.diff().fillna(0)
    w_tga = -zscore_rolling(delta_tga, 252).iloc[-1] if len(delta_tga) > 30 else 0
    w_tga = np.clip(w_tga * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Treasury',
        target='Banks',
        flow=delta_tga.iloc[-1] if len(delta_tga) > 0 else 0,
        driver=f"ΔTGA={delta_tga.iloc[-1]:.0f}B",
        z_score=w_tga,
        is_drain=w_tga < 0,
        weight=abs(w_tga)
    ))

    # 4. Banks ↔ Dealers (Repo GC, driver: TGCR)
    delta_tgcr = tgcr.diff().fillna(0)
    w_tgcr = -zscore_rolling(delta_tgcr, 252).iloc[-1] if len(delta_tgcr) > 30 else 0
    w_tgcr = np.clip(w_tgcr * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Banks',
        target='Dealers',
        flow=delta_tgcr.iloc[-1] if len(delta_tgcr) > 0 else 0,
        driver=f"ΔTGCR={delta_tgcr.iloc[-1]:.2f}bp",
        z_score=w_tgcr,
        is_drain=w_tgcr < 0,
        weight=abs(w_tgcr)
    ))

    # 5. Unsecured (SOFR-EFFR spread)
    sofr = df.get('SOFR', pd.Series(0)).fillna(0)
    effr = df.get('EFFR', pd.Series(0)).fillna(0)
    sofr_effr = (sofr - effr).fillna(0)
    w_sofr_effr = -zscore_rolling(sofr_effr, 252).iloc[-1] if len(sofr_effr) > 30 else 0
    w_sofr_effr = np.clip(w_sofr_effr * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Banks',
        target='MMFs',
        flow=sofr_effr.iloc[-1] if len(sofr_effr) > 0 else 0,
        driver=f"SOFR-EFFR={sofr_effr.iloc[-1]:.2f}bp",
        z_score=w_sofr_effr,
        is_drain=w_sofr_effr < 0,
        weight=abs(w_sofr_effr)
    ))

    # 6. OBFR-SOFR spread
    obfr = df.get('OBFR', pd.Series(0)).fillna(0)
    obfr_sofr = (obfr - sofr).fillna(0)
    w_obfr_sofr = -zscore_rolling(obfr_sofr, 252).iloc[-1] if len(obfr_sofr) > 30 else 0
    w_obfr_sofr = np.clip(w_obfr_sofr * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Banks',
        target='Banks',  # Internal stress
        flow=obfr_sofr.iloc[-1] if len(obfr_sofr) > 0 else 0,
        driver=f"OBFR-SOFR={obfr_sofr.iloc[-1]:.2f}bp",
        z_score=w_obfr_sofr,
        is_drain=w_obfr_sofr < 0,
        weight=abs(w_obfr_sofr)
    ))

    # 7. FHLB → Banks (driver: +ΔFHLB = stress/dependency)
    delta_fhlb = fhlb.diff().fillna(0)
    w_fhlb = -zscore_rolling(delta_fhlb, 252).iloc[-1] if len(delta_fhlb) > 30 else 0
    w_fhlb = np.clip(w_fhlb * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='FHLB',
        target='Banks',
        flow=delta_fhlb.iloc[-1] if len(delta_fhlb) > 0 else 0,
        driver=f"ΔFHLB={delta_fhlb.iloc[-1]:.0f}B",
        z_score=w_fhlb,
        is_drain=w_fhlb < 0,
        weight=abs(w_fhlb)
    ))

    # 8. Dealers → UST Market (driver: UST illiquidity)
    w_ust = -zscore_rolling(move, 252).iloc[-1] if len(move) > 30 else 0
    w_ust = np.clip(w_ust * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Dealers',
        target='UST_Market',
        flow=move.iloc[-1] if len(move) > 0 else 0,
        driver=f"MOVE={move.iloc[-1]:.0f}",
        z_score=w_ust,
        is_drain=w_ust < 0,
        weight=abs(w_ust)
    ))

    # 9. Banks → Credit (HY) (driver: +ΔHY OAS = tightening)
    delta_hy = hy_oas.diff().fillna(0)
    w_hy = -zscore_rolling(delta_hy, 252).iloc[-1] if len(delta_hy) > 30 else 0
    w_hy = np.clip(w_hy * qtr_multiplier, -5, 5)
    graph.add_edge_data(GraphEdge(
        source='Banks',
        target='Credit_HY',
        flow=delta_hy.iloc[-1] if len(delta_hy) > 0 else 0,
        driver=f"ΔHY_OAS={delta_hy.iloc[-1]:.0f}bp",
        z_score=w_hy,
        is_drain=w_hy < 0,
        weight=abs(w_hy)
    ))

    # Compute derived metrics
    graph.stress_flow_index = graph.compute_stress_flow_index()
    graph.hotspots = graph.identify_hotspots(threshold=1.5)

    # Validate reserve identity
    reserve_data = {
        'delta_reserves': delta_res,
        'delta_tga': delta_tga,
        'delta_rrp': delta_rrp,
        'delta_walcl': df.get('WALCL', pd.Series(0)).diff().fillna(0)
    }
    graph.reserve_identity = graph.validate_reserve_identity(reserve_data)

    return graph
