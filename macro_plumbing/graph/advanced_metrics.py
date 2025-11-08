"""
advanced_metrics.py
Advanced network metrics for systemic risk.

Implements:
1. Systemic Importance Measure (SIM) - Basel III SIFI framework
2. Contagion Index (CoI) - Expected systemic losses
3. Network LCR - Liquidity coverage with contagion
4. Vulnerable Node Index - Nodes at risk

Based on Basel III, Cont et al. (2013), IMF frameworks.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional


def compute_systemic_importance(
    G: nx.DiGraph,
    node: str,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute Systemic Importance Measure (SIM) for a node.

    Based on Basel III SIFI framework:
    SIM = weighted average of:
    - Size (node balance/total)
    - Interconnectedness (degree centrality)
    - Substitutability (inverse betweenness - low = important)
    - Complexity (number of connections)

    Parameters
    ----------
    G : nx.DiGraph
        Network graph
    node : str
        Node name
    weights : dict, optional
        Custom weights for each component (default: equal 0.25)

    Returns
    -------
    float
        Systemic importance score [0, 1]
    """
    if weights is None:
        weights = {
            'size': 0.25,
            'interconnectedness': 0.25,
            'substitutability': 0.25,
            'complexity': 0.25,
        }

    if node not in G.nodes():
        return 0

    # 1. Size component (balance relative to total)
    node_balance = G.nodes[node].get('balance', 0)
    total_balance = sum(G.nodes[n].get('balance', 0) for n in G.nodes())

    if total_balance > 0:
        size_score = node_balance / total_balance
    else:
        size_score = 0

    # 2. Interconnectedness (degree centrality)
    degree_centrality = nx.degree_centrality(G)
    interconnect_score = degree_centrality.get(node, 0)

    # 3. Substitutability (inverse betweenness)
    # Low betweenness = high substitutability (node can be bypassed)
    # High betweenness = low substitutability (critical intermediary)
    try:
        betweenness = nx.betweenness_centrality(G, weight='abs_flow')
        # Invert: low betweenness → high substitutability → low risk
        # high betweenness → low substitutability → high risk
        substit_score = betweenness.get(node, 0)
    except:
        substit_score = 0

    # 4. Complexity (normalized degree)
    degree = G.degree(node)
    max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
    complexity_score = degree / max_degree if max_degree > 0 else 0

    # Weighted average
    sim = (
        weights['size'] * size_score +
        weights['interconnectedness'] * interconnect_score +
        weights['substitutability'] * substit_score +
        weights['complexity'] * complexity_score
    )

    return sim


def compute_contagion_index(
    G: nx.DiGraph,
    failure_probs: Dict[str, float] = None,
    default_prob: float = 0.01
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Contagion Index (CoI).

    CoI = Expected systemic losses from node failures

    CoI = Σ_i (prob_failure_i * spillover_i)

    where spillover_i = impact on network if node i fails

    Based on Cont et al. (2013)

    Parameters
    ----------
    G : nx.DiGraph
        Network graph
    failure_probs : dict, optional
        {node: prob_failure}
    default_prob : float
        Default failure probability if not specified

    Returns
    -------
    tuple
        (total_coi, node_contributions)
    """
    if failure_probs is None:
        # Estimate from node stress (if available)
        failure_probs = {}
        for node in G.nodes():
            stress = G.nodes[node].get('stress_prob', 0.5)
            # Convert stress [0,1] to failure prob (sigmoid)
            failure_probs[node] = 1 / (1 + np.exp(-5 * (stress - 0.5)))

    node_contributions = {}
    total_coi = 0

    for node in G.nodes():
        prob_fail = failure_probs.get(node, default_prob)

        # Estimate spillover: remove node and measure impact
        # Proxy: sum of outgoing flows (liquidity drained if node fails)
        spillover = 0

        for successor in G.successors(node):
            edge_data = G.edges[node, successor]
            flow = abs(edge_data.get('flow', 0))
            spillover += flow

        # Contribution to CoI
        contribution = prob_fail * spillover
        node_contributions[node] = contribution
        total_coi += contribution

    return total_coi, node_contributions


def compute_network_lcr(
    G: nx.DiGraph,
    node: str,
    hqla: float = None,
    horizon_days: int = 30,
    contagion_factor: float = 0.3
) -> float:
    """
    Compute Network-aware Liquidity Coverage Ratio (LCR).

    Standard LCR = HQLA / Net Cash Outflows (30-day)

    Network LCR adds contagion component:
    Net Outflows = Direct outflows + Contagion-induced outflows

    Contagion outflows = Σ_neighbors (prob_fail_neighbor * exposure_to_neighbor)

    Parameters
    ----------
    G : nx.DiGraph
        Network graph
    node : str
        Node name
    hqla : float, optional
        High-Quality Liquid Assets (defaults to node balance)
    horizon_days : int
        LCR horizon (default 30 days)
    contagion_factor : float
        Weight on contagion component (0-1)

    Returns
    -------
    float
        Network LCR ratio
    """
    if node not in G.nodes():
        return 0

    # HQLA (proxy: node balance)
    if hqla is None:
        hqla = G.nodes[node].get('balance', 0)

    # Direct outflows (proxy: sum of outgoing drain edges)
    direct_outflows = 0

    for successor in G.successors(node):
        edge_data = G.edges[node, successor]
        if edge_data.get('is_drain', False):
            flow = abs(edge_data.get('flow', 0))
            # Annualize from daily to 30-day
            direct_outflows += flow * horizon_days

    # Contagion outflows
    contagion_outflows = 0

    for predecessor in G.predecessors(node):
        # If predecessor fails, we lose incoming flow
        neighbor_stress = G.nodes[predecessor].get('stress_prob', 0.5)
        prob_fail = 1 / (1 + np.exp(-5 * (neighbor_stress - 0.5)))

        # Exposure
        edge_data = G.edges[predecessor, node]
        exposure = abs(edge_data.get('flow', 0)) * horizon_days

        contagion_outflows += prob_fail * exposure

    # Total outflows
    total_outflows = direct_outflows + contagion_factor * contagion_outflows

    # LCR
    if total_outflows > 0:
        network_lcr = hqla / total_outflows
    else:
        network_lcr = float('inf')

    return network_lcr


def identify_vulnerable_nodes(
    G: nx.DiGraph,
    lcr_threshold: float = 1.0,
    sim_threshold: float = 0.5
) -> List[Tuple[str, str, float, float]]:
    """
    Identify vulnerable nodes.

    Vulnerable if:
    - Low Network LCR (< threshold, e.g., 1.0)
    - High Systemic Importance (> threshold)

    These are "systemically important but vulnerable" nodes.

    Parameters
    ----------
    G : nx.DiGraph
        Network graph
    lcr_threshold : float
        LCR threshold (default 1.0 = 100%)
    sim_threshold : float
        SIM threshold (default 0.5)

    Returns
    -------
    list
        List of (node, vulnerability_reason, lcr, sim) tuples
    """
    vulnerable = []

    for node in G.nodes():
        # Compute metrics
        sim = compute_systemic_importance(G, node)
        lcr = compute_network_lcr(G, node)

        # Check vulnerability
        is_vulnerable = False
        reason = ""

        if lcr < lcr_threshold and sim > sim_threshold:
            is_vulnerable = True
            reason = "Systemically important but low liquidity"
        elif lcr < lcr_threshold:
            is_vulnerable = True
            reason = "Low liquidity coverage"
        elif sim > sim_threshold:
            # High SIM but adequate LCR - monitor
            is_vulnerable = True
            reason = "High systemic importance (monitor)"

        if is_vulnerable:
            vulnerable.append((node, reason, lcr, sim))

    # Sort by SIM (most important first)
    vulnerable.sort(key=lambda x: x[3], reverse=True)

    return vulnerable


def compute_network_resilience_score(G: nx.DiGraph) -> float:
    """
    Compute overall network resilience score.

    Combines:
    - Average LCR (higher = better)
    - Network density (higher = more connected)
    - Inverse contagion index (lower CoI = better)

    Returns score [0, 1] where 1 = most resilient

    Parameters
    ----------
    G : nx.DiGraph
        Network graph

    Returns
    -------
    float
        Resilience score [0, 1]
    """
    # 1. Average Network LCR
    lcrs = []
    for node in G.nodes():
        lcr = compute_network_lcr(G, node)
        if np.isfinite(lcr):
            lcrs.append(min(lcr, 5.0))  # Cap at 5 to avoid inf

    avg_lcr = np.mean(lcrs) if len(lcrs) > 0 else 0
    # Normalize to [0,1] assuming 2.0 is "very good"
    lcr_score = min(avg_lcr / 2.0, 1.0)

    # 2. Network density
    density = nx.density(G)

    # 3. Inverse contagion index
    coi, _ = compute_contagion_index(G)
    # Normalize (assuming CoI < 1000 is good)
    coi_score = max(0, 1 - coi / 1000)

    # Composite (equal weight)
    resilience = (lcr_score + density + coi_score) / 3

    return resilience


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Create sample graph
    G = nx.DiGraph()

    # Add nodes
    nodes_data = [
        ('Fed', {'balance': 5000, 'stress_prob': 0.1}),
        ('Treasury', {'balance': 500, 'stress_prob': 0.3}),
        ('Banks', {'balance': 3000, 'stress_prob': 0.5}),
        ('MMFs', {'balance': 4000, 'stress_prob': 0.4}),
        ('Dealers', {'balance': 1000, 'stress_prob': 0.6}),
    ]

    for node, data in nodes_data:
        G.add_node(node, **data)

    # Add edges
    edges_data = [
        ('Fed', 'Banks', {'flow': 100, 'is_drain': False, 'abs_flow': 100}),
        ('Treasury', 'Banks', {'flow': -50, 'is_drain': True, 'abs_flow': 50}),
        ('MMFs', 'Banks', {'flow': -80, 'is_drain': True, 'abs_flow': 80}),
        ('Banks', 'Dealers', {'flow': 30, 'is_drain': False, 'abs_flow': 30}),
        ('Dealers', 'MMFs', {'flow': 20, 'is_drain': False, 'abs_flow': 20}),
    ]

    for source, target, data in edges_data:
        G.add_edge(source, target, **data)

    print("="*70)
    print("ADVANCED NETWORK METRICS")
    print("="*70)

    # 1. Systemic Importance Measure (SIM)
    print(f"\n{'='*70}")
    print("SYSTEMIC IMPORTANCE MEASURE (SIM)")
    print(f"{'='*70}")

    sim_scores = {}
    for node in G.nodes():
        sim = compute_systemic_importance(G, node)
        sim_scores[node] = sim

    # Sort by SIM
    sorted_sim = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)

    for node, sim in sorted_sim:
        print(f"{node:15s} {sim:.3f}", end="")
        if sim > 0.5:
            print(" 🔴 SIFI (Systemically Important)")
        elif sim > 0.3:
            print(" 🟡 Moderate importance")
        else:
            print(" ✅ Lower importance")

    # 2. Contagion Index (CoI)
    print(f"\n{'='*70}")
    print("CONTAGION INDEX (CoI)")
    print(f"{'='*70}")

    coi, node_contrib = compute_contagion_index(G)

    print(f"\nTotal CoI: {coi:.1f}")

    print(f"\nNode contributions (top 5):")
    sorted_contrib = sorted(node_contrib.items(), key=lambda x: x[1], reverse=True)
    for node, contrib in sorted_contrib[:5]:
        pct = contrib / coi * 100 if coi > 0 else 0
        print(f"  {node:15s} {contrib:>8.1f} ({pct:>5.1f}%)")

    # 3. Network LCR
    print(f"\n{'='*70}")
    print("NETWORK LIQUIDITY COVERAGE RATIO (LCR)")
    print(f"{'='*70}")

    print(f"\nNode LCRs:")
    for node in G.nodes():
        lcr = compute_network_lcr(G, node)
        print(f"{node:15s} {lcr:>6.2f}", end="")

        if lcr < 1.0:
            print(" 🔴 Below regulatory minimum (1.0)")
        elif lcr < 1.5:
            print(" 🟡 Adequate but low buffer")
        else:
            print(" ✅ Strong liquidity")

    # 4. Vulnerable Nodes
    print(f"\n{'='*70}")
    print("VULNERABLE NODES")
    print(f"{'='*70}")

    vulnerable = identify_vulnerable_nodes(G, lcr_threshold=1.5, sim_threshold=0.3)

    if len(vulnerable) > 0:
        print(f"\nIdentified {len(vulnerable)} vulnerable nodes:\n")
        for node, reason, lcr, sim in vulnerable:
            print(f"{node:15s}")
            print(f"  Reason: {reason}")
            print(f"  LCR:    {lcr:.2f}")
            print(f"  SIM:    {sim:.3f}")
            print()
    else:
        print("\n✅ No vulnerable nodes identified")

    # 5. Network Resilience Score
    print(f"{'='*70}")
    print("NETWORK RESILIENCE SCORE")
    print(f"{'='*70}")

    resilience = compute_network_resilience_score(G)

    print(f"\nOverall resilience: {resilience:.3f}")

    if resilience > 0.7:
        print("✅ HIGH RESILIENCE - Network robust")
    elif resilience > 0.5:
        print("🟡 MODERATE RESILIENCE")
    else:
        print("🔴 LOW RESILIENCE - Network vulnerable")

    print(f"\n{'='*70}")
