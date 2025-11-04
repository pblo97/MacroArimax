"""
graph_analysis.py
Network analysis for liquidity graphs: bottlenecks, centrality, fragility.

Implements:
1. Min-cut analysis (critical edges/nodes)
2. Centrality metrics (PageRank, betweenness, closeness)
3. Systemic importance scores
4. Network fragility measures
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set


class LiquidityNetworkAnalysis:
    """Analyze structural properties of liquidity network."""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize network analysis.

        Parameters
        ----------
        graph : nx.DiGraph
            Liquidity flow graph
        """
        # Store original graph reference
        self._original_graph = graph
        # Create a working copy to avoid modifying original
        self.graph = graph.copy()
        self.nodes = list(self.graph.nodes())
        self.n_nodes = len(self.nodes)

    def compute_min_cut(
        self,
        source: str,
        target: str,
        capacity_attr: str = 'abs_flow',
    ) -> Tuple[float, List[Tuple[str, str]]]:
        """
        Compute minimum cut between two nodes.

        Identifies critical edges whose removal would disconnect source from target.

        Parameters
        ----------
        source : str
            Source node
        target : str
            Target node
        capacity_attr : str
            Edge attribute to use as capacity

        Returns
        -------
        tuple
            (cut_value, cut_edges)
        """
        # Get capacities - convert to list first
        edges_data = list(self.graph.edges(data=True))
        capacities = {}
        for u, v, data in edges_data:
            capacity = data.get(capacity_attr, 1.0)
            capacities[(u, v)] = abs(capacity)

        # Set capacities
        nx.set_edge_attributes(self.graph, capacities, 'capacity')

        # Compute min cut
        try:
            cut_value, partition = nx.minimum_cut(
                self.graph,
                source,
                target,
                capacity='capacity'
            )

            # Find cut edges
            reachable, non_reachable = partition
            cut_edges = []

            # Convert to list to avoid iteration issues
            edges_list = list(self.graph.edges())
            for u, v in edges_list:
                if u in reachable and v in non_reachable:
                    cut_edges.append((u, v))

            return cut_value, cut_edges

        except nx.NetworkXError:
            return 0.0, []

    def find_all_critical_edges(
        self,
        capacity_attr: str = 'abs_flow',
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Find all critical edges (bottlenecks) in network.

        An edge is critical if its removal significantly reduces connectivity.

        Parameters
        ----------
        capacity_attr : str
            Edge attribute for capacity

        Returns
        -------
        list
            List of ((source, target), criticality_score) tuples
        """
        critical_edges = []

        # Compute baseline connectivity
        baseline_connectivity = self._compute_avg_connectivity()

        # Convert to list to avoid "dictionary changed during iteration" error
        edges_list = list(self.graph.edges())

        # Test each edge
        for u, v in edges_list:
            # Temporarily remove edge
            edge_data = self.graph[u][v].copy()
            self.graph.remove_edge(u, v)

            # Compute new connectivity
            new_connectivity = self._compute_avg_connectivity()

            # Restore edge
            self.graph.add_edge(u, v, **edge_data)

            # Criticality = drop in connectivity
            criticality = baseline_connectivity - new_connectivity
            critical_edges.append(((u, v), criticality))

        # Sort by criticality
        critical_edges.sort(key=lambda x: x[1], reverse=True)

        return critical_edges

    def _compute_avg_connectivity(self) -> float:
        """Compute average pairwise connectivity."""
        total = 0
        count = 0

        for source in self.nodes:
            for target in self.nodes:
                if source != target:
                    if nx.has_path(self.graph, source, target):
                        total += 1
                    count += 1

        return total / count if count > 0 else 0.0

    def compute_centrality_metrics(
        self,
        stress_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute multiple centrality metrics.

        Parameters
        ----------
        stress_weights : dict, optional
            Node stress levels to weight metrics

        Returns
        -------
        pd.DataFrame
            Centrality scores for each node
        """
        results = []

        # Degree centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        # Weighted degree (by flow)
        weighted_in = {}
        weighted_out = {}

        for node in self.nodes:
            weighted_in[node] = sum(
                abs(self.graph[u][node].get('flow', 0))
                for u in self.graph.predecessors(node)
            )
            weighted_out[node] = sum(
                abs(self.graph[node][v].get('flow', 0))
                for v in self.graph.successors(node)
            )

        # PageRank (weighted by flow)
        try:
            # Create weight dict - convert to list first to avoid iteration issues
            edges_data = list(self.graph.edges(data=True))
            edge_weights = {
                (u, v): abs(data.get('flow', 1))
                for u, v, data in edges_data
            }
            nx.set_edge_attributes(self.graph, edge_weights, 'weight')

            pagerank = nx.pagerank(self.graph, weight='weight', alpha=0.85)
        except Exception as e:
            # Fallback to uniform distribution
            pagerank = {node: 1.0 / self.n_nodes for node in self.nodes}

        # Betweenness centrality (how often node is on shortest path)
        try:
            betweenness = nx.betweenness_centrality(self.graph, weight='abs_flow')
        except:
            betweenness = {node: 0.0 for node in self.nodes}

        # Closeness centrality (average distance to others)
        try:
            closeness = nx.closeness_centrality(self.graph)
        except:
            closeness = {node: 0.0 for node in self.nodes}

        # Eigenvector centrality (connected to important nodes)
        try:
            eigenvector = nx.eigenvector_centrality(
                self.graph,
                weight='abs_flow',
                max_iter=1000
            )
        except:
            eigenvector = {node: 1.0 / self.n_nodes for node in self.nodes}

        # Compile results
        for node in self.nodes:
            row = {
                'node': node,
                'in_degree': in_degree[node],
                'out_degree': out_degree[node],
                'weighted_in': weighted_in[node],
                'weighted_out': weighted_out[node],
                'pagerank': pagerank[node],
                'betweenness': betweenness[node],
                'closeness': closeness[node],
                'eigenvector': eigenvector[node],
            }

            # If stress provided, compute stressed centrality
            if stress_weights:
                stress = stress_weights.get(node, 0)
                row['stress_weighted_pagerank'] = pagerank[node] * stress
                row['stress'] = stress

            results.append(row)

        return pd.DataFrame(results)

    def compute_systemic_importance(
        self,
        stress_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute systemic importance score for each node.

        Combines multiple metrics:
        - Centrality (how connected)
        - Vulnerability (exposed to contagion)
        - Size (total flows)

        Parameters
        ----------
        stress_weights : dict, optional
            Current stress levels

        Returns
        -------
        dict
            Systemic importance scores (0-1)
        """
        centrality_df = self.compute_centrality_metrics(stress_weights)

        importance = {}

        for _, row in centrality_df.iterrows():
            node = row['node']

            # Normalize and combine metrics
            score = (
                0.3 * row['pagerank'] * self.n_nodes +  # PageRank (normalized)
                0.2 * row['betweenness'] +  # Betweenness
                0.2 * row['eigenvector'] +  # Eigenvector
                0.15 * (row['weighted_in'] + row['weighted_out']) / 1000 +  # Size
                0.15 * (row['in_degree'] + row['out_degree']) / self.n_nodes  # Connectivity
            )

            importance[node] = min(score, 1.0)

        return importance

    def identify_systemically_important_nodes(
        self,
        top_k: int = 3,
        stress_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Identify most systemically important nodes (SIFIs).

        Parameters
        ----------
        top_k : int
            Number of top nodes to return
        stress_weights : dict, optional
            Current stress levels

        Returns
        -------
        list
            List of (node, importance_score) tuples
        """
        importance = self.compute_systemic_importance(stress_weights)

        # Sort by importance
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    def compute_network_fragility(self) -> Dict[str, float]:
        """
        Compute various fragility metrics for the network.

        Returns
        -------
        dict
            Fragility metrics
        """
        # Density (how connected)
        density = nx.density(self.graph)

        # Clustering (local connectivity)
        try:
            clustering = nx.average_clustering(self.graph.to_undirected())
        except:
            clustering = 0.0

        # Average shortest path (efficiency)
        try:
            if nx.is_strongly_connected(self.graph):
                avg_path_length = nx.average_shortest_path_length(self.graph)
            else:
                # Use largest strongly connected component
                largest_cc = max(
                    nx.strongly_connected_components(self.graph),
                    key=len
                )
                subgraph = self.graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = float('inf')

        # Assortativity (similar nodes connect?)
        try:
            assortativity = nx.degree_assortativity_coefficient(self.graph)
        except:
            assortativity = 0.0

        # Number of strongly connected components (fragmentation)
        n_components = nx.number_strongly_connected_components(self.graph)

        return {
            'density': density,
            'clustering': clustering,
            'avg_path_length': avg_path_length,
            'assortativity': assortativity,
            'n_components': n_components,
            'fragility_score': self._compute_fragility_score(
                density, clustering, avg_path_length, n_components
            ),
        }

    def _compute_fragility_score(
        self,
        density: float,
        clustering: float,
        avg_path: float,
        n_components: int,
    ) -> float:
        """Aggregate fragility score (0-1, higher = more fragile)."""
        # Normalize components
        density_score = 1 - density  # Low density = fragile
        clustering_score = 1 - clustering  # Low clustering = fragile
        path_score = min(avg_path / 10, 1.0)  # Long paths = fragile
        component_score = min((n_components - 1) / self.n_nodes, 1.0)  # Fragmentation

        fragility = (
            0.3 * density_score +
            0.2 * clustering_score +
            0.3 * path_score +
            0.2 * component_score
        )

        return fragility

    def find_critical_nodes(self) -> List[Tuple[str, float]]:
        """
        Find critical nodes whose removal fragments network.

        Returns
        -------
        list
            List of (node, criticality) tuples
        """
        critical_nodes = []

        baseline_components = nx.number_strongly_connected_components(self.graph)

        for node in self.nodes:
            # Temporarily remove node
            edges_to_restore = list(self.graph.in_edges(node)) + list(
                self.graph.out_edges(node)
            )
            edge_data = {}
            for u, v in edges_to_restore:
                edge_data[(u, v)] = self.graph[u][v].copy()
                self.graph.remove_edge(u, v)

            self.graph.remove_node(node)

            # Check new connectivity
            new_components = nx.number_strongly_connected_components(self.graph)

            # Restore node
            self.graph.add_node(node)
            for (u, v), data in edge_data.items():
                self.graph.add_edge(u, v, **data)

            # Criticality
            criticality = new_components - baseline_components
            critical_nodes.append((node, criticality))

        # Sort by criticality
        critical_nodes.sort(key=lambda x: x[1], reverse=True)

        return critical_nodes


# Example usage
if __name__ == "__main__":
    # Create test graph
    G = nx.DiGraph()

    # Add nodes
    nodes = ["Fed", "Banks", "MMFs", "Treasury", "Dealers"]
    for node in nodes:
        G.add_node(node)

    # Add edges
    edges = [
        ("Banks", "Fed", 100),
        ("MMFs", "Fed", 150),
        ("Dealers", "Banks", 80),
        ("Treasury", "Banks", 120),
        ("Fed", "Banks", 50),
        ("Banks", "MMFs", 60),
        ("MMFs", "Dealers", 40),
    ]

    for source, target, flow in edges:
        G.add_edge(source, target, flow=flow, abs_flow=abs(flow))

    # Initialize analysis
    analysis = LiquidityNetworkAnalysis(G)

    print("=== Centrality Metrics ===")
    centrality = analysis.compute_centrality_metrics()
    print(centrality[['node', 'pagerank', 'betweenness', 'weighted_in', 'weighted_out']])

    print("\n=== Systemically Important Nodes ===")
    sifis = analysis.identify_systemically_important_nodes(top_k=3)
    for node, score in sifis:
        print(f"{node}: {score:.3f}")

    print("\n=== Network Fragility ===")
    fragility = analysis.compute_network_fragility()
    for metric, value in fragility.items():
        print(f"{metric}: {value:.3f}")

    print("\n=== Critical Nodes ===")
    critical = analysis.find_critical_nodes()
    for node, crit in critical[:3]:
        print(f"{node}: {crit:.1f} (components added if removed)")

    print("\n=== Critical Edges (Bottlenecks) ===")
    critical_edges = analysis.find_all_critical_edges()
    for (u, v), crit in critical_edges[:3]:
        print(f"{u} -> {v}: criticality {crit:.3f}")

    print("\n=== Min-Cut: Banks to Fed ===")
    cut_value, cut_edges = analysis.compute_min_cut("Banks", "Fed")
    print(f"Cut value: {cut_value:.1f}")
    print(f"Cut edges: {cut_edges}")
