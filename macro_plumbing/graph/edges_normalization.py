"""
Edge Normalization & Family Classification
===========================================

Separate edges by family (stock $ vs. spread bp) and compute robust SFI.
Prevents SFI from being dominated by unit differences.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from enum import Enum


class EdgeFamily(Enum):
    """Edge type classification."""
    STOCK_FLOW = "stock_$"  # Measured in $ billions (reserves, TGA, RRP, etc.)
    SPREAD_PRESSURE = "spread_bp"  # Measured in basis points (SOFR-EFFR, TGCR, etc.)
    PERCENTAGE = "percentage_%"  # Measured in % (stress probabilities, etc.)
    UNKNOWN = "unknown"


# Edge family mappings
EDGE_FAMILY_MAP = {
    # Stock/Flow edges ($ billions)
    ('Fed', 'Banks'): EdgeFamily.STOCK_FLOW,
    ('Treasury', 'Fed'): EdgeFamily.STOCK_FLOW,
    ('Fed', 'ON_RRP'): EdgeFamily.STOCK_FLOW,
    ('MMFs', 'ON_RRP'): EdgeFamily.STOCK_FLOW,
    ('FHLB', 'Banks'): EdgeFamily.STOCK_FLOW,
    ('Banks', 'Dealers'): EdgeFamily.STOCK_FLOW,
    ('Fed', 'Dealers'): EdgeFamily.STOCK_FLOW,

    # Spread/Pressure edges (basis points)
    ('Banks', 'MMFs'): EdgeFamily.SPREAD_PRESSURE,  # SOFR-EFFR
    ('Dealers', 'MMFs'): EdgeFamily.SPREAD_PRESSURE,  # TGCR, repo rates
    ('UST_Market', 'Dealers'): EdgeFamily.SPREAD_PRESSURE,  # Depth, bid-ask
    ('Credit_HY', 'Dealers'): EdgeFamily.SPREAD_PRESSURE,  # HY OAS spread
}


def classify_edge_family(source: str, target: str, driver: str = "") -> EdgeFamily:
    """
    Classify edge into family based on source, target, and driver.

    Parameters
    ----------
    source : str
        Source node
    target : str
        Target node
    driver : str
        Driver name (e.g., 'delta_reserves', 'sofr_effr_spread')

    Returns
    -------
    EdgeFamily
        Classification
    """
    # Check explicit mapping
    edge_key = (source, target)
    if edge_key in EDGE_FAMILY_MAP:
        return EDGE_FAMILY_MAP[edge_key]

    # Infer from driver name
    driver_lower = driver.lower()

    # Stock indicators
    stock_keywords = ['delta', 'reserves', 'tga', 'rrp', 'walcl', 'fhlb', 'advance']
    if any(kw in driver_lower for kw in stock_keywords):
        return EdgeFamily.STOCK_FLOW

    # Spread indicators
    spread_keywords = ['spread', 'sofr', 'effr', 'tgcr', 'oas', 'bp', 'basis']
    if any(kw in driver_lower for kw in spread_keywords):
        return EdgeFamily.SPREAD_PRESSURE

    # Percentage indicators
    pct_keywords = ['prob', 'stress', 'percentile', 'pct']
    if any(kw in driver_lower for kw in pct_keywords):
        return EdgeFamily.PERCENTAGE

    return EdgeFamily.UNKNOWN


def normalize_edge_weights_by_family(
    graph: nx.DiGraph,
    z_score_attr: str = 'z_score'
) -> Dict[str, float]:
    """
    Normalize edge z-scores within families to make them comparable.

    Strategy:
    - Group edges by family
    - Within each family, z-scores are already normalized
    - Across families, use family-specific scaling factors

    Parameters
    ----------
    graph : nx.DiGraph
        Graph with edge attributes
    z_score_attr : str
        Attribute name for z-scores

    Returns
    -------
    Dict[str, float]
        Family-level statistics
    """
    # Group edges by family
    families = {family: [] for family in EdgeFamily}

    for source, target, data in graph.edges(data=True):
        driver = data.get('driver', '')
        family = classify_edge_family(source, target, driver)
        z_score = data.get(z_score_attr, 0)
        families[family].append({
            'edge': (source, target),
            'z_score': z_score,
            'data': data
        })

    # Compute family statistics
    family_stats = {}
    for family, edges in families.items():
        if not edges:
            continue

        z_scores = [e['z_score'] for e in edges if not np.isnan(e['z_score'])]
        if z_scores:
            family_stats[family.value] = {
                'count': len(z_scores),
                'mean_abs_z': np.mean(np.abs(z_scores)),
                'std_z': np.std(z_scores),
                'max_abs_z': np.max(np.abs(z_scores))
            }

    return family_stats


def compute_robust_sfi(
    graph: nx.DiGraph,
    method: str = 'family_normalized'
) -> Tuple[float, Dict[str, float]]:
    """
    Compute robust Stress Flow Index with family normalization.

    SFI = sum of (z_score * family_weight) for draining edges
        - sum of (z_score * family_weight) for injecting edges

    Parameters
    ----------
    graph : nx.DiGraph
        Graph with edge attributes
    method : str
        'simple': original SFI (sum of z * is_drain - z * is_inject)
        'family_normalized': normalize within families first

    Returns
    -------
    sfi : float
        Stress Flow Index
    breakdown : Dict[str, float]
        SFI contribution by family
    """
    if method == 'simple':
        # Original method
        sfi = 0
        for _, _, data in graph.edges(data=True):
            z = data.get('z_score', 0)
            is_drain = data.get('is_drain', False)
            weight = data.get('weight', 1.0)

            if is_drain:
                sfi += z * weight
            else:
                sfi -= z * weight

        return sfi, {}

    # Family-normalized method
    families = {family: [] for family in EdgeFamily}

    for source, target, data in graph.edges(data=True):
        driver = data.get('driver', '')
        family = classify_edge_family(source, target, driver)
        families[family].append({
            'edge': (source, target),
            'z_score': data.get('z_score', 0),
            'is_drain': data.get('is_drain', False),
            'weight': data.get('weight', 1.0),
            'data': data
        })

    # Compute SFI by family
    sfi_total = 0
    sfi_breakdown = {}

    for family, edges in families.items():
        if not edges:
            continue

        # Compute family SFI
        family_sfi = 0
        for edge in edges:
            z = edge['z_score']
            if np.isnan(z):
                continue

            is_drain = edge['is_drain']
            weight = edge['weight']

            if is_drain:
                family_sfi += z * weight
            else:
                family_sfi -= z * weight

        # Normalize by number of edges in family (prevent single-edge families from dominating)
        n_edges = len([e for e in edges if not np.isnan(e['z_score'])])
        if n_edges > 0:
            family_sfi_normalized = family_sfi / np.sqrt(n_edges)
        else:
            family_sfi_normalized = 0

        sfi_breakdown[family.value] = family_sfi_normalized
        sfi_total += family_sfi_normalized

    return sfi_total, sfi_breakdown


def add_edge_family_attributes(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Add family classification to each edge as an attribute.

    Parameters
    ----------
    graph : nx.DiGraph
        Input graph

    Returns
    -------
    nx.DiGraph
        Graph with 'family' attribute added to edges
    """
    for source, target, data in graph.edges(data=True):
        driver = data.get('driver', '')
        family = classify_edge_family(source, target, driver)
        data['family'] = family.value
        data['family_enum'] = family

    return graph


def get_family_summary_table(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Create summary table of edges by family.

    Returns
    -------
    pd.DataFrame
        Columns: Family, Count, Avg_Z_Score, Draining_Edges, Injecting_Edges
    """
    families = {family: [] for family in EdgeFamily}

    for source, target, data in graph.edges(data=True):
        driver = data.get('driver', '')
        family = classify_edge_family(source, target, driver)
        families[family].append({
            'edge': f"{source}→{target}",
            'z_score': data.get('z_score', 0),
            'is_drain': data.get('is_drain', False),
        })

    rows = []
    for family, edges in families.items():
        if not edges:
            continue

        z_scores = [e['z_score'] for e in edges if not np.isnan(e['z_score'])]
        draining = sum(1 for e in edges if e['is_drain'])
        injecting = len(edges) - draining

        rows.append({
            'Family': family.value,
            'Count': len(edges),
            'Avg_Abs_Z': np.mean(np.abs(z_scores)) if z_scores else 0,
            'Draining': draining,
            'Injecting': injecting
        })

    return pd.DataFrame(rows)


def visualize_edge_units(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Create table showing edge, driver, unit, and z-score.

    Helps debug unit mismatches in SFI calculation.

    Returns
    -------
    pd.DataFrame
        Columns: Edge, Driver, Family, Z_Score, Is_Drain, Flow/Value
    """
    rows = []

    for source, target, data in graph.edges(data=True):
        driver = data.get('driver', '')
        family = classify_edge_family(source, target, driver)
        z_score = data.get('z_score', 0)
        is_drain = data.get('is_drain', False)
        flow = data.get('flow', 0)

        rows.append({
            'Edge': f"{source}→{target}",
            'Driver': driver,
            'Family': family.value,
            'Z_Score': f"{z_score:.2f}",
            'Is_Drain': '🔴' if is_drain else '🟢',
            'Flow/Value': f"{flow:.1f}"
        })

    return pd.DataFrame(rows)
