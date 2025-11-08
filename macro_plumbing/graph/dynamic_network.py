"""
dynamic_network.py
Time-varying network structure analysis.

Based on Fed (2021) "Liquidity Networks, Interconnectedness"

Implements:
1. Rolling network metrics (connectivity, centralization, fragmentation)
2. Structural break detection
3. Regime-conditional analysis
4. Crisis-induced network changes

All using FRED data.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_network_density(G: nx.DiGraph) -> float:
    """
    Compute network density.

    Density = actual_edges / possible_edges

    High density = well-connected
    Low density = fragmented

    Parameters
    ----------
    G : nx.DiGraph
        Network graph

    Returns
    -------
    float
        Density [0, 1]
    """
    return nx.density(G)


def compute_network_centralization(G: nx.DiGraph) -> float:
    """
    Compute network centralization (concentration).

    Measures how much the network is dominated by a few central nodes.

    Based on Freeman centralization:
    C = Σ(max_degree - degree_i) / max_possible

    High centralization = hub-and-spoke (fragile to hub failure)
    Low centralization = distributed (more robust)

    Parameters
    ----------
    G : nx.DiGraph
        Network graph

    Returns
    -------
    float
        Centralization [0, 1]
    """
    if G.number_of_nodes() == 0:
        return 0

    # Out-degree centrality
    out_degrees = dict(G.out_degree())
    degrees = list(out_degrees.values())

    if len(degrees) == 0:
        return 0

    max_degree = max(degrees)
    n = G.number_of_nodes()

    if n <= 1:
        return 0

    # Freeman centralization
    numerator = sum(max_degree - d for d in degrees)
    denominator = (n - 1) * (n - 2)  # Max possible for directed graph

    if denominator == 0:
        return 0

    centralization = numerator / denominator

    return min(centralization, 1.0)


def compute_edge_correlation(df: pd.DataFrame, edge_drivers: List[str]) -> float:
    """
    Compute average correlation among edge weights.

    High correlation → contagion risk (all edges move together)
    Low correlation → diversified stress

    Parameters
    ----------
    df : pd.DataFrame
        FRED data
    edge_drivers : list
        List of column names for edge drivers

    Returns
    -------
    float
        Average pairwise correlation
    """
    available_drivers = [d for d in edge_drivers if d in df.columns]

    if len(available_drivers) < 2:
        return 0

    corr_matrix = df[available_drivers].corr()

    # Get upper triangle (avoid diagonal and duplicates)
    n = len(corr_matrix)
    upper_tri = []
    for i in range(n):
        for j in range(i+1, n):
            upper_tri.append(corr_matrix.iloc[i, j])

    if len(upper_tri) == 0:
        return 0

    avg_corr = np.mean(upper_tri)

    return avg_corr


def compute_largest_component_size(G: nx.DiGraph) -> int:
    """
    Compute size of largest weakly connected component.

    Fragmentation indicator:
    - Size = n_nodes → fully connected
    - Size << n_nodes → fragmented

    Parameters
    ----------
    G : nx.DiGraph
        Network graph

    Returns
    -------
    int
        Number of nodes in largest component
    """
    if G.number_of_nodes() == 0:
        return 0

    components = list(nx.weakly_connected_components(G))

    if len(components) == 0:
        return 0

    largest = max(components, key=len)

    return len(largest)


class DynamicNetworkAnalyzer:
    """Analyze time-varying network structure."""

    def __init__(self, window: int = 63):
        """
        Initialize analyzer.

        Parameters
        ----------
        window : int
            Rolling window (days) for computing metrics
        """
        self.window = window
        self.metrics_history = []

    def compute_rolling_metrics(
        self,
        graphs: Dict[pd.Timestamp, nx.DiGraph],
        df: pd.DataFrame,
        edge_drivers: List[str]
    ) -> pd.DataFrame:
        """
        Compute rolling network metrics.

        Parameters
        ----------
        graphs : dict
            {timestamp: graph}
        df : pd.DataFrame
            FRED data
        edge_drivers : list
            Edge driver column names

        Returns
        -------
        pd.DataFrame
            Rolling metrics with columns:
            - density
            - centralization
            - edge_correlation
            - largest_component_pct
            - avg_degree
        """
        metrics = []

        # Sort by date
        sorted_dates = sorted(graphs.keys())

        for date in sorted_dates:
            G = graphs[date]

            # Compute metrics
            density = compute_network_density(G)
            centralization = compute_network_centralization(G)
            largest_comp_size = compute_largest_component_size(G)
            largest_comp_pct = largest_comp_size / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

            # Average degree
            if G.number_of_nodes() > 0:
                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            else:
                avg_degree = 0

            # Edge correlation (from recent data window)
            window_end = date
            window_start = date - pd.Timedelta(days=self.window)
            window_df = df.loc[window_start:window_end]

            if len(window_df) > 10:
                edge_corr = compute_edge_correlation(window_df, edge_drivers)
            else:
                edge_corr = 0

            metrics.append({
                'date': date,
                'density': density,
                'centralization': centralization,
                'edge_correlation': edge_corr,
                'largest_component_pct': largest_comp_pct,
                'avg_degree': avg_degree,
            })

        self.metrics_history = pd.DataFrame(metrics)

        return self.metrics_history

    def detect_structural_breaks(
        self,
        metric_name: str = 'density',
        min_segment_length: int = 30,
        penalty: float = 10.0
    ) -> List[pd.Timestamp]:
        """
        Detect structural breaks in network metric.

        Uses CUSUM (Cumulative Sum) approach.

        Parameters
        ----------
        metric_name : str
            Metric to analyze ('density', 'centralization', etc.)
        min_segment_length : int
            Minimum days between breakpoints
        penalty : float
            Penalty for adding breakpoints (higher = fewer breaks)

        Returns
        -------
        list
            List of breakpoint dates
        """
        if len(self.metrics_history) == 0:
            return []

        if metric_name not in self.metrics_history.columns:
            return []

        signal = self.metrics_history[metric_name].values

        # Simple CUSUM detector
        # (would use ruptures library for production, but implementing basic version)
        breakpoints = []

        # Compute cumulative sum of deviations
        mean_signal = np.mean(signal)
        cusum_pos = np.zeros(len(signal))
        cusum_neg = np.zeros(len(signal))

        for i in range(1, len(signal)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (signal[i] - mean_signal))
            cusum_neg[i] = max(0, cusum_neg[i-1] - (signal[i] - mean_signal))

        # Detect breaks where CUSUM exceeds threshold
        threshold = penalty * np.std(signal)

        for i in range(min_segment_length, len(signal) - min_segment_length):
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                # Check if far enough from previous break
                if len(breakpoints) == 0 or i - breakpoints[-1] > min_segment_length:
                    breakpoints.append(i)

        # Convert indices to dates
        breakpoint_dates = [self.metrics_history.iloc[i]['date'] for i in breakpoints]

        return breakpoint_dates

    def identify_crisis_regime(
        self,
        date: pd.Timestamp,
        crisis_threshold: Dict[str, float] = None
    ) -> str:
        """
        Identify network regime at given date.

        Regimes:
        - NORMAL: Business as usual
        - STRESS: Elevated but not crisis
        - CRISIS: Fragmentation, high correlation, centralization

        Parameters
        ----------
        date : pd.Timestamp
            Date to evaluate
        crisis_threshold : dict, optional
            Custom thresholds for each metric

        Returns
        -------
        str
            Regime label
        """
        if len(self.metrics_history) == 0:
            return "UNKNOWN"

        # Default thresholds (from literature)
        if crisis_threshold is None:
            crisis_threshold = {
                'density': 0.3,  # Low density = fragmentation
                'centralization': 0.7,  # High centralization = fragile
                'edge_correlation': 0.7,  # High correlation = contagion
            }

        # Find row for this date
        row = self.metrics_history[self.metrics_history['date'] == date]

        if len(row) == 0:
            return "UNKNOWN"

        row = row.iloc[0]

        # Count stress signals
        stress_signals = 0

        if row['density'] < crisis_threshold['density']:
            stress_signals += 1

        if row['centralization'] > crisis_threshold['centralization']:
            stress_signals += 1

        if row['edge_correlation'] > crisis_threshold['edge_correlation']:
            stress_signals += 1

        # Classify
        if stress_signals >= 2:
            return "CRISIS"
        elif stress_signals == 1:
            return "STRESS"
        else:
            return "NORMAL"

    def compute_fragmentation_index(self) -> pd.Series:
        """
        Compute composite fragmentation index.

        Combines:
        - Low density (fragmentation)
        - High centralization (hub dependency)
        - Low largest component (disconnection)

        Returns
        -------
        pd.Series
            Fragmentation index (0-1, higher = more fragmented)
        """
        if len(self.metrics_history) == 0:
            return pd.Series()

        # Normalize metrics [0, 1]
        density_norm = 1 - self.metrics_history['density']  # Invert (low density = fragmentation)
        central_norm = self.metrics_history['centralization']  # High = fragmentation
        component_norm = 1 - self.metrics_history['largest_component_pct']  # Invert

        # Composite (equal weight)
        fragmentation = (density_norm + central_norm + component_norm) / 3

        return pd.Series(fragmentation.values, index=self.metrics_history['date'])


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n = len(dates)

    df = pd.DataFrame({
        'VIX': 15 + 10 * np.random.randn(n).cumsum() * 0.1,
        'HY_OAS': 4 + 2 * np.random.randn(n).cumsum() * 0.1,
        'MOVE': 80 + 20 * np.random.randn(n).cumsum() * 0.1,
    }, index=dates)

    # Create sample graphs (simplified)
    graphs = {}

    for i, date in enumerate(dates[::10]):  # Every 10 days
        G = nx.DiGraph()

        # Add nodes
        nodes = ['Fed', 'Treasury', 'Banks', 'MMFs', 'Dealers']
        G.add_nodes_from(nodes)

        # Add edges (with some time variation)
        base_edges = [
            ('Fed', 'Banks'),
            ('Treasury', 'Banks'),
            ('MMFs', 'Banks'),
            ('Banks', 'Dealers'),
            ('Dealers', 'MMFs'),
        ]

        # Randomly drop some edges during "crisis" periods
        vix_current = df.loc[date, 'VIX']
        if vix_current > 25:  # Stress period
            # Drop some edges (fragmentation)
            edges_to_add = np.random.choice(base_edges, size=3, replace=False)
        else:
            edges_to_add = base_edges

        G.add_edges_from(edges_to_add)

        graphs[date] = G

    print("="*70)
    print("DYNAMIC NETWORK ANALYSIS")
    print("="*70)

    # Initialize analyzer
    analyzer = DynamicNetworkAnalyzer(window=63)

    # Compute rolling metrics
    edge_drivers = ['VIX', 'HY_OAS', 'MOVE']
    metrics_df = analyzer.compute_rolling_metrics(graphs, df, edge_drivers)

    print(f"\nComputed metrics for {len(metrics_df)} timepoints")
    print(f"\nLatest metrics ({metrics_df.iloc[-1]['date'].strftime('%Y-%m-%d')}):")
    print(f"  Density:            {metrics_df.iloc[-1]['density']:.3f}")
    print(f"  Centralization:     {metrics_df.iloc[-1]['centralization']:.3f}")
    print(f"  Edge correlation:   {metrics_df.iloc[-1]['edge_correlation']:+.3f}")
    print(f"  Largest comp (%):   {metrics_df.iloc[-1]['largest_component_pct']:.1%}")
    print(f"  Avg degree:         {metrics_df.iloc[-1]['avg_degree']:.2f}")

    # Detect structural breaks
    print(f"\n{'='*70}")
    print("STRUCTURAL BREAKS (density)")
    print(f"{'='*70}")

    breaks = analyzer.detect_structural_breaks('density', penalty=15.0)

    if len(breaks) > 0:
        print(f"\nDetected {len(breaks)} structural breaks:")
        for brk in breaks:
            print(f"  {brk.strftime('%Y-%m-%d')}")
    else:
        print("\nNo significant structural breaks detected")

    # Regime identification
    print(f"\n{'='*70}")
    print("REGIME CLASSIFICATION")
    print(f"{'='*70}")

    regimes = []
    for date in metrics_df['date']:
        regime = analyzer.identify_crisis_regime(date)
        regimes.append(regime)

    regime_counts = pd.Series(regimes).value_counts()
    print(f"\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime:10s}: {count:3d} days ({pct:5.1f}%)")

    # Current regime
    current_regime = regimes[-1]
    print(f"\nCurrent regime: {current_regime}")

    # Fragmentation index
    print(f"\n{'='*70}")
    print("FRAGMENTATION INDEX")
    print(f"{'='*70}")

    frag_index = analyzer.compute_fragmentation_index()

    print(f"\nCurrent fragmentation: {frag_index.iloc[-1]:.3f}")
    print(f"Mean fragmentation:    {frag_index.mean():.3f}")
    print(f"Max fragmentation:     {frag_index.max():.3f}")
    print(f"Min fragmentation:     {frag_index.min():.3f}")

    if frag_index.iloc[-1] > 0.7:
        print("\n🔴 HIGH FRAGMENTATION - Network vulnerable")
    elif frag_index.iloc[-1] > 0.5:
        print("\n🟡 MODERATE FRAGMENTATION")
    else:
        print("\n✅ LOW FRAGMENTATION - Network resilient")

    print(f"\n{'='*70}")
