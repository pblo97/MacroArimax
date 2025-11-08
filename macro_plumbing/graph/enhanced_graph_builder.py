"""
enhanced_graph_builder.py
Enhanced liquidity graph builder with all 4 phases integrated.

Combines:
- Phase 1: Liquidity spirals + Margin calls
- Phase 2: NBFI nodes (Hedge Funds, Asset Managers, Insurance)
- Phase 3: Dynamic network metrics
- Phase 4: Advanced metrics (SIM, CoI, Network LCR)

All using FREE FRED data.

Usage:
    from macro_plumbing.graph.enhanced_graph_builder import build_enhanced_graph

    graph, metrics = build_enhanced_graph(df)
    print(graph.summary())
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import our new modules
try:
    from .margin_calls import (
        estimate_initial_margin_change,
        estimate_variation_margin,
        estimate_repo_margin_haircut,
        compute_margin_stress_index
    )
    from .nbfi_nodes import (
        build_nbfi_nodes,
        compute_nbfi_systemic_score
    )
    from .advanced_metrics import (
        compute_systemic_importance,
        compute_contagion_index,
        compute_network_lcr,
        identify_vulnerable_nodes,
        compute_network_resilience_score
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from margin_calls import (
        estimate_initial_margin_change,
        estimate_variation_margin,
        estimate_repo_margin_haircut,
        compute_margin_stress_index
    )
    from nbfi_nodes import (
        build_nbfi_nodes,
        compute_nbfi_systemic_score
    )
    from advanced_metrics import (
        compute_systemic_importance,
        compute_contagion_index,
        compute_network_lcr,
        identify_vulnerable_nodes,
        compute_network_resilience_score
    )


@dataclass
class EnhancedGraphMetrics:
    """Metrics from enhanced graph."""
    # Network structure
    density: float
    centralization: float
    largest_component_pct: float

    # Margin/collateral
    margin_stress_index: float
    current_haircut: float
    delta_im: float
    vm: float

    # NBFI
    nbfi_systemic_score: float
    hedge_fund_stress: float
    asset_manager_stress: float
    insurance_stress: float

    # Advanced metrics
    contagion_index: float
    network_resilience: float
    vulnerable_nodes: List[Tuple[str, str, float, float]]

    # SIM scores
    sim_scores: Dict[str, float]

    # Network LCR
    lcr_scores: Dict[str, float]


class EnhancedLiquidityGraph:
    """Enhanced liquidity graph with all phases."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize enhanced graph.

        Parameters
        ----------
        df : pd.DataFrame
            FRED data
        """
        self.df = df
        self.G = nx.DiGraph()
        self.metrics = None

        # Build graph
        self._build_graph()

    def _build_graph(self):
        """Build graph with all enhancements."""
        # Get latest data
        latest = self.df.index[-1]

        # === PHASE 1: Core nodes + Margin calls ===

        # 1. Fed
        reserves = self.df.get('RESERVES', pd.Series(0)).fillna(0)
        self.G.add_node(
            'Fed',
            type='central_bank',
            balance=reserves.iloc[-1] if len(reserves) > 0 else 0,
            stress_prob=0.01  # Very low
        )

        # 2. Treasury
        tga = self.df.get('TGA', pd.Series(0)).fillna(0)
        self.G.add_node(
            'Treasury',
            type='government',
            balance=tga.iloc[-1] if len(tga) > 0 else 0,
            stress_prob=0.05
        )

        # 3. Banks
        bank_reserves = self.df.get('RESERVES', reserves).fillna(0)
        self.G.add_node(
            'Banks',
            type='bank',
            balance=bank_reserves.iloc[-1] if len(bank_reserves) > 0 else 0,
            stress_prob=0.30  # Moderate
        )

        # 4. MMFs
        rrp = self.df.get('RRP', pd.Series(0)).fillna(0)
        self.G.add_node(
            'MMFs',
            type='mmf',
            balance=rrp.iloc[-1] * 0.8 if len(rrp) > 0 else 0,
            stress_prob=0.20
        )

        # 5. Dealers
        self.G.add_node(
            'Dealers',
            type='dealer',
            balance=1000,  # Normalized
            stress_prob=0.50
        )

        # === PHASE 2: NBFI Nodes ===

        nbfi_nodes = build_nbfi_nodes(self.df)

        for name, nbfi_node in nbfi_nodes.items():
            self.G.add_node(
                nbfi_node.name,
                type=nbfi_node.type,
                balance=nbfi_node.aum_estimate,
                stress_prob=max(0, min(1, (nbfi_node.stress_score + 2) / 4))  # Convert z-score to prob
            )

        # === EDGES ===

        # Core edges
        self._add_core_edges()

        # Margin call edges (Phase 1)
        self._add_margin_call_edges()

        # NBFI edges (Phase 2)
        self._add_nbfi_edges()

    def _add_core_edges(self):
        """Add core liquidity flow edges."""
        # Fed → Banks (reserves)
        reserves = self.df.get('RESERVES', pd.Series(0)).fillna(0)
        delta_res = reserves.diff().iloc[-1] if len(reserves) > 1 else 0

        self.G.add_edge(
            'Fed',
            'Banks',
            flow=delta_res,
            driver='ΔReserves',
            is_drain=delta_res < 0,
            abs_flow=abs(delta_res)
        )

        # Treasury → Banks (TGA)
        tga = self.df.get('TGA', pd.Series(0)).fillna(0)
        delta_tga = tga.diff().iloc[-1] if len(tga) > 1 else 0

        self.G.add_edge(
            'Treasury',
            'Banks',
            flow=delta_tga,
            driver='ΔTGA',
            is_drain=delta_tga > 0,  # Positive TGA = drain
            abs_flow=abs(delta_tga)
        )

        # MMFs → Fed (RRP)
        rrp = self.df.get('RRP', pd.Series(0)).fillna(0)
        delta_rrp = rrp.diff().iloc[-1] if len(rrp) > 1 else 0

        self.G.add_edge(
            'MMFs',
            'Fed',
            flow=delta_rrp,
            driver='ΔRRP',
            is_drain=delta_rrp > 0,  # Positive RRP = drain
            abs_flow=abs(delta_rrp)
        )

        # Banks ↔ Dealers (repo)
        tgcr = self.df.get('TGCR', pd.Series(0)).fillna(0)
        delta_tgcr = tgcr.diff().iloc[-1] if len(tgcr) > 1 else 0

        self.G.add_edge(
            'Banks',
            'Dealers',
            flow=delta_tgcr,
            driver='ΔTGCR',
            is_drain=delta_tgcr > 0,
            abs_flow=abs(delta_tgcr)
        )

    def _add_margin_call_edges(self):
        """Add margin call edges (Phase 1)."""
        # Estimate margin flows
        delta_im = estimate_initial_margin_change(self.df)
        vm = estimate_variation_margin(self.df)

        if len(delta_im) > 0:
            im_latest = delta_im.iloc[-1]
        else:
            im_latest = 0

        if len(vm) > 0:
            vm_latest = vm.iloc[-1]
        else:
            vm_latest = 0

        # Banks → Dealers (Initial Margin)
        self.G.add_edge(
            'Banks',
            'Dealers',
            flow=im_latest / 1000,  # Convert $M to $B
            driver='ΔIM',
            is_drain=im_latest > 0,
            abs_flow=abs(im_latest) / 1000
        )

        # Banks ↔ Dealers (Variation Margin)
        self.G.add_edge(
            'Dealers',
            'Banks',
            flow=vm_latest / 1000,
            driver='VM',
            is_drain=vm_latest < 0,  # Negative VM = Banks pay Dealers
            abs_flow=abs(vm_latest) / 1000
        )

    def _add_nbfi_edges(self):
        """Add NBFI edges (Phase 2)."""
        # Hedge Funds → Dealers (margin/collateral)
        if 'Hedge_Funds' in self.G.nodes():
            hf_stress = self.G.nodes['Hedge_Funds']['stress_prob']

            # Higher stress → more margin calls
            margin_flow = -50 * hf_stress  # $50B max

            self.G.add_edge(
                'Hedge_Funds',
                'Dealers',
                flow=margin_flow,
                driver='HF_Margin',
                is_drain=margin_flow < 0,
                abs_flow=abs(margin_flow)
            )

        # Asset Managers → Markets (redemption pressure)
        if 'Asset_Managers' in self.G.nodes():
            am_stress = self.G.nodes['Asset_Managers']['stress_prob']

            # Higher stress → more redemptions → selling
            redemption_flow = -100 * am_stress  # $100B max

            self.G.add_edge(
                'Asset_Managers',
                'Dealers',
                flow=redemption_flow,
                driver='AM_Redemptions',
                is_drain=redemption_flow < 0,
                abs_flow=abs(redemption_flow)
            )

        # Insurance/Pensions → Banks (duration mismatch, low rates)
        if 'Insurance_Pensions' in self.G.nodes():
            ins_stress = self.G.nodes['Insurance_Pensions']['stress_prob']

            # Duration mismatch creates funding need
            funding_flow = -30 * ins_stress  # $30B max

            self.G.add_edge(
                'Insurance_Pensions',
                'Banks',
                flow=funding_flow,
                driver='Ins_Funding',
                is_drain=funding_flow < 0,
                abs_flow=abs(funding_flow)
            )

    def compute_all_metrics(self) -> EnhancedGraphMetrics:
        """Compute all enhanced metrics."""
        # Network structure
        density = nx.density(self.G)
        centralization = self._compute_centralization()
        largest_comp = self._compute_largest_component()

        # Margin/collateral
        margin_stress = compute_margin_stress_index(self.df)
        margin_stress_latest = margin_stress.iloc[-1] if len(margin_stress) > 0 else 0

        haircut = estimate_repo_margin_haircut(self.df)
        haircut_latest = haircut.iloc[-1] if len(haircut) > 0 else 0

        delta_im = estimate_initial_margin_change(self.df)
        im_latest = delta_im.iloc[-1] if len(delta_im) > 0 else 0

        vm = estimate_variation_margin(self.df)
        vm_latest = vm.iloc[-1] if len(vm) > 0 else 0

        # NBFI - Compute systemic score from stress probs
        hf_stress = self.G.nodes.get('Hedge_Funds', {}).get('stress_prob', 0)
        am_stress = self.G.nodes.get('Asset_Managers', {}).get('stress_prob', 0)
        ins_stress = self.G.nodes.get('Insurance_Pensions', {}).get('stress_prob', 0)

        # Weighted average by AUM
        hf_aum = self.G.nodes.get('Hedge_Funds', {}).get('balance', 0)
        am_aum = self.G.nodes.get('Asset_Managers', {}).get('balance', 0)
        ins_aum = self.G.nodes.get('Insurance_Pensions', {}).get('balance', 0)

        total_aum = hf_aum + am_aum + ins_aum

        if total_aum > 0:
            # Convert stress probs back to z-scores and weight
            hf_z = (hf_stress - 0.5) * 4  # Rough conversion
            am_z = (am_stress - 0.5) * 4
            ins_z = (ins_stress - 0.5) * 4

            nbfi_systemic = (
                (hf_aum * hf_z + am_aum * am_z + ins_aum * ins_z) / total_aum
            )
        else:
            nbfi_systemic = 0

        # Advanced metrics
        coi, _ = compute_contagion_index(self.G)
        resilience = compute_network_resilience_score(self.G)
        vulnerable = identify_vulnerable_nodes(self.G)

        # SIM scores
        sim_scores = {}
        for node in self.G.nodes():
            sim_scores[node] = compute_systemic_importance(self.G, node)

        # Network LCR
        lcr_scores = {}
        for node in self.G.nodes():
            lcr_scores[node] = compute_network_lcr(self.G, node)

        metrics = EnhancedGraphMetrics(
            density=density,
            centralization=centralization,
            largest_component_pct=largest_comp,
            margin_stress_index=margin_stress_latest,
            current_haircut=haircut_latest,
            delta_im=im_latest,
            vm=vm_latest,
            nbfi_systemic_score=nbfi_systemic,
            hedge_fund_stress=hf_stress,
            asset_manager_stress=am_stress,
            insurance_stress=ins_stress,
            contagion_index=coi,
            network_resilience=resilience,
            vulnerable_nodes=vulnerable,
            sim_scores=sim_scores,
            lcr_scores=lcr_scores
        )

        self.metrics = metrics
        return metrics

    def _compute_centralization(self) -> float:
        """Compute network centralization."""
        if self.G.number_of_nodes() <= 1:
            return 0

        out_degrees = dict(self.G.out_degree())
        degrees = list(out_degrees.values())

        if len(degrees) == 0:
            return 0

        max_degree = max(degrees)
        n = self.G.number_of_nodes()

        numerator = sum(max_degree - d for d in degrees)
        denominator = (n - 1) * (n - 2)

        if denominator == 0:
            return 0

        return numerator / denominator

    def _compute_largest_component(self) -> float:
        """Compute largest component percentage."""
        if self.G.number_of_nodes() == 0:
            return 0

        components = list(nx.weakly_connected_components(self.G))

        if len(components) == 0:
            return 0

        largest = max(components, key=len)
        return len(largest) / self.G.number_of_nodes()

    def summary(self) -> str:
        """Generate summary report."""
        if self.metrics is None:
            self.compute_all_metrics()

        report = []
        report.append("="*70)
        report.append("ENHANCED LIQUIDITY NETWORK ANALYSIS")
        report.append("="*70)

        # Network structure
        report.append("\n📊 NETWORK STRUCTURE")
        report.append(f"  Nodes: {self.G.number_of_nodes()}")
        report.append(f"  Edges: {self.G.number_of_edges()}")
        report.append(f"  Density: {self.metrics.density:.3f}")
        report.append(f"  Centralization: {self.metrics.centralization:.3f}")
        report.append(f"  Largest component: {self.metrics.largest_component_pct:.1%}")

        # Margin/Collateral
        report.append("\n💰 MARGIN & COLLATERAL")
        report.append(f"  Margin stress index: {self.metrics.margin_stress_index:+.2f} σ")
        report.append(f"  Current haircut: {self.metrics.current_haircut:.2%}")
        report.append(f"  ΔIM (daily): ${self.metrics.delta_im:+,.0f}M")
        report.append(f"  VM (daily): ${self.metrics.vm:+,.0f}M")

        # NBFI
        report.append("\n🏦 NBFI SECTOR")
        report.append(f"  NBFI systemic score: {self.metrics.nbfi_systemic_score:+.2f} σ")
        report.append(f"  Hedge Funds: {self.metrics.hedge_fund_stress:.2f} stress")
        report.append(f"  Asset Managers: {self.metrics.asset_manager_stress:.2f} stress")
        report.append(f"  Insurance/Pensions: {self.metrics.insurance_stress:.2f} stress")

        # Advanced metrics
        report.append("\n⚠️  SYSTEMIC RISK")
        report.append(f"  Contagion Index: {self.metrics.contagion_index:.1f}")
        report.append(f"  Network Resilience: {self.metrics.network_resilience:.3f}")
        report.append(f"  Vulnerable nodes: {len(self.metrics.vulnerable_nodes)}")

        # Top SIMs
        report.append("\n🎯 TOP SYSTEMICALLY IMPORTANT NODES")
        sorted_sim = sorted(self.metrics.sim_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, sim in sorted_sim:
            report.append(f"  {node:20s} {sim:.3f}")

        # Vulnerable nodes
        if len(self.metrics.vulnerable_nodes) > 0:
            report.append("\n🔴 VULNERABLE NODES")
            for node, reason, lcr, sim in self.metrics.vulnerable_nodes[:5]:
                report.append(f"  {node:20s} (LCR={lcr:.2f}, SIM={sim:.3f})")
                report.append(f"    → {reason}")

        report.append("\n" + "="*70)

        return "\n".join(report)


def build_enhanced_graph(df: pd.DataFrame) -> Tuple[EnhancedLiquidityGraph, EnhancedGraphMetrics]:
    """
    Build enhanced liquidity graph with all phases.

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    tuple
        (graph, metrics)
    """
    graph = EnhancedLiquidityGraph(df)
    metrics = graph.compute_all_metrics()

    return graph, metrics


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n = len(dates)

    df = pd.DataFrame({
        'RESERVES': 3000 + 500 * np.random.randn(n).cumsum() * 0.1,
        'TGA': 500 + 100 * np.random.randn(n).cumsum() * 0.1,
        'RRP': 2000 + 500 * np.random.randn(n).cumsum() * 0.1,
        'TGCR': 5.0 + 0.5 * np.random.randn(n).cumsum() * 0.1,
        'VIX': 15 + 10 * np.random.randn(n).cumsum() * 0.1,
        'MOVE': 80 + 20 * np.random.randn(n).cumsum() * 0.1,
        'HY_OAS': 4 + 2 * np.random.randn(n).cumsum() * 0.1,
        'DGS10': 3.0 + 0.5 * np.random.randn(n).cumsum() * 0.1,
        'bbb_aaa_spread': 1.5 + 0.3 * np.random.randn(n).cumsum() * 0.1,
        'cp_tbill_spread': 0.1 + 0.05 * np.random.randn(n).cumsum() * 0.05,
        'SP500': 3000 * np.exp(0.0001 * np.arange(n) + 0.01 * np.random.randn(n).cumsum()),
    }, index=dates)

    # Clip to realistic ranges
    df['RESERVES'] = df['RESERVES'].clip(1000, 5000)
    df['TGA'] = df['TGA'].clip(100, 1000)
    df['RRP'] = df['RRP'].clip(0, 3000)
    df['VIX'] = df['VIX'].clip(10, 80)
    df['MOVE'] = df['MOVE'].clip(50, 200)
    df['HY_OAS'] = df['HY_OAS'].clip(2, 15)
    df['DGS10'] = df['DGS10'].clip(0.5, 6)

    print("Building enhanced liquidity graph...")
    graph, metrics = build_enhanced_graph(df)

    print("\n" + graph.summary())
