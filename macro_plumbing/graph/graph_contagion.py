"""
graph_contagion.py
Stress contagion models for liquidity networks using random walks.

Implements:
1. 1-step random walk contagion (immediate neighbors)
2. Multi-step diffusion (k-hop propagation)
3. Susceptible-Infected (SI) model adaptation
4. Stress amplification via feedback loops
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


class StressContagion:
    """Model stress contagion through liquidity network."""

    def __init__(
        self,
        graph: nx.DiGraph,
        damping: float = 0.85,
    ):
        """
        Initialize contagion model.

        Parameters
        ----------
        graph : nx.DiGraph
            Liquidity flow graph
        damping : float
            Damping factor for random walk (0-1)
            Higher = more weight on propagation
        """
        self.graph = graph
        self.damping = damping

        # Build transition matrix
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        self._build_transition_matrix()

    def _build_transition_matrix(self):
        """Build weighted transition matrix for random walk."""
        # Initialize matrix
        W = np.zeros((self.n_nodes, self.n_nodes))

        # Fill with edge weights (flow magnitudes)
        for i, node_i in enumerate(self.nodes):
            # Outgoing edges
            out_edges = self.graph.out_edges(node_i, data=True)
            total_out_weight = 0

            for _, target, data in out_edges:
                j = self.node_to_idx[target]
                weight = abs(data.get('flow', 0))
                W[i, j] = weight
                total_out_weight += weight

            # Normalize row (probability distribution)
            if total_out_weight > 0:
                W[i, :] /= total_out_weight
            else:
                # No outgoing edges: stay in place
                W[i, i] = 1.0

        # Transpose for column-stochastic (standard for PageRank-style)
        self.transition_matrix = W.T

    def one_step_contagion(
        self,
        initial_stress: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute 1-step random walk contagion.

        Parameters
        ----------
        initial_stress : dict
            Initial stress levels {node_name: stress_prob}

        Returns
        -------
        dict
            Propagated stress levels after 1 step
        """
        # Convert to vector
        stress_vec = np.array([initial_stress.get(node, 0) for node in self.nodes])

        # Random walk: s_new = α * W @ s_old + (1-α) * s_old
        propagated = (
            self.damping * (self.transition_matrix @ stress_vec) +
            (1 - self.damping) * stress_vec
        )

        # Convert back to dict
        return {node: propagated[i] for i, node in enumerate(self.nodes)}

    def k_step_contagion(
        self,
        initial_stress: Dict[str, float],
        k: int = 3,
    ) -> pd.DataFrame:
        """
        Simulate k-step contagion diffusion.

        Parameters
        ----------
        initial_stress : dict
            Initial stress levels
        k : int
            Number of steps

        Returns
        -------
        pd.DataFrame
            Evolution of stress over k steps
        """
        results = []
        current_stress = initial_stress.copy()

        for step in range(k + 1):
            results.append({
                'step': step,
                **current_stress,
            })

            if step < k:
                current_stress = self.one_step_contagion(current_stress)

        return pd.DataFrame(results)

    def compute_stress_pagerank(
        self,
        initial_stress: Dict[str, float],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Compute steady-state stress distribution (PageRank-style).

        Parameters
        ----------
        initial_stress : dict
            Initial stress seeds
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        dict
            Steady-state stress levels
        """
        stress_vec = np.array([initial_stress.get(node, 0) for node in self.nodes])
        initial_vec = stress_vec.copy()

        for i in range(max_iter):
            new_stress = (
                self.damping * (self.transition_matrix @ stress_vec) +
                (1 - self.damping) * initial_vec
            )

            # Check convergence
            if np.max(np.abs(new_stress - stress_vec)) < tol:
                break

            stress_vec = new_stress

        return {node: stress_vec[i] for i, node in enumerate(self.nodes)}

    def compute_amplification_factor(
        self,
        node: str,
        k: int = 3,
    ) -> float:
        """
        Compute stress amplification factor for a node.

        Measures how much a unit stress at this node amplifies
        through the network after k steps.

        Parameters
        ----------
        node : str
            Source node
        k : int
            Number of propagation steps

        Returns
        -------
        float
            Amplification factor (>1 means amplification, <1 damping)
        """
        # Initialize with unit stress at target node
        initial_stress = {n: 0.0 for n in self.nodes}
        initial_stress[node] = 1.0

        # Propagate k steps
        final_stress = self.k_step_contagion(initial_stress, k=k).iloc[-1]

        # Sum total stress in system
        total_stress = sum(final_stress[n] for n in self.nodes)

        return total_stress

    def identify_superspreaders(
        self,
        top_k: int = 3,
        steps: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Identify nodes that amplify stress most (superspreaders).

        Parameters
        ----------
        top_k : int
            Number of top spreaders to return
        steps : int
            Propagation steps for amplification

        Returns
        -------
        list
            List of (node, amplification_factor) tuples
        """
        amplifications = []

        for node in self.nodes:
            amp = self.compute_amplification_factor(node, k=steps)
            amplifications.append((node, amp))

        # Sort by amplification
        amplifications.sort(key=lambda x: x[1], reverse=True)

        return amplifications[:top_k]

    def compute_vulnerability_score(
        self,
        node: str,
        stress_scenario: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute vulnerability of a node to contagion from others.

        Parameters
        ----------
        node : str
            Target node
        stress_scenario : dict, optional
            Stress levels at other nodes (uniform if None)

        Returns
        -------
        float
            Vulnerability score (0-1)
        """
        if stress_scenario is None:
            # Uniform stress at all other nodes
            stress_scenario = {n: 1.0 if n != node else 0.0 for n in self.nodes}

        # Propagate 1 step
        propagated = self.one_step_contagion(stress_scenario)

        return propagated[node]

    def compute_contagion_matrix(self, k: int = 1) -> pd.DataFrame:
        """
        Compute full contagion matrix: C[i,j] = stress at j due to unit stress at i.

        Parameters
        ----------
        k : int
            Number of propagation steps

        Returns
        -------
        pd.DataFrame
            Contagion matrix (rows=source, cols=target)
        """
        contagion_matrix = np.zeros((self.n_nodes, self.n_nodes))

        for i, source_node in enumerate(self.nodes):
            # Unit stress at source
            initial_stress = {n: 0.0 for n in self.nodes}
            initial_stress[source_node] = 1.0

            # Propagate
            final_stress = self.k_step_contagion(initial_stress, k=k).iloc[-1]

            # Store in matrix
            for j, target_node in enumerate(self.nodes):
                contagion_matrix[i, j] = final_stress[target_node]

        return pd.DataFrame(
            contagion_matrix,
            index=self.nodes,
            columns=self.nodes,
        )

    def simulate_shock(
        self,
        shock_node: str,
        shock_magnitude: float = 1.0,
        steps: int = 5,
    ) -> pd.DataFrame:
        """
        Simulate a stress shock at a specific node.

        Parameters
        ----------
        shock_node : str
            Node receiving initial shock
        shock_magnitude : float
            Size of initial shock
        steps : int
            Simulation steps

        Returns
        -------
        pd.DataFrame
            Time series of stress at all nodes
        """
        initial_stress = {n: 0.0 for n in self.nodes}
        initial_stress[shock_node] = shock_magnitude

        return self.k_step_contagion(initial_stress, k=steps)


class FeedbackContagion(StressContagion):
    """Extended contagion model with feedback loops."""

    def __init__(
        self,
        graph: nx.DiGraph,
        damping: float = 0.85,
        feedback_strength: float = 0.2,
    ):
        """
        Initialize with feedback effects.

        Parameters
        ----------
        graph : nx.DiGraph
            Liquidity flow graph
        damping : float
            Damping factor
        feedback_strength : float
            Strength of feedback amplification (0-1)
        """
        super().__init__(graph, damping)
        self.feedback_strength = feedback_strength

    def one_step_with_feedback(
        self,
        initial_stress: Dict[str, float],
    ) -> Dict[str, float]:
        """
        1-step contagion with feedback amplification.

        Feedback: high stress -> more stress via behavioral effects
        """
        # Base propagation
        propagated = self.one_step_contagion(initial_stress)

        # Feedback: stress -> amplification
        for node in self.nodes:
            current_stress = propagated[node]
            # Quadratic feedback (stress amplifies nonlinearly)
            feedback = self.feedback_strength * current_stress ** 2
            propagated[node] = min(current_stress + feedback, 1.0)

        return propagated

    def simulate_with_feedback(
        self,
        initial_stress: Dict[str, float],
        steps: int = 10,
    ) -> pd.DataFrame:
        """Simulate contagion with feedback effects."""
        results = []
        current_stress = initial_stress.copy()

        for step in range(steps + 1):
            results.append({
                'step': step,
                **current_stress,
            })

            if step < steps:
                current_stress = self.one_step_with_feedback(current_stress)

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Create test graph
    G = nx.DiGraph()

    # Add nodes
    nodes = ["Fed", "Banks", "MMFs", "Treasury", "Dealers"]
    for node in nodes:
        G.add_node(node)

    # Add weighted edges (flow magnitudes)
    edges = [
        ("Banks", "Fed", 100),
        ("MMFs", "Fed", 150),
        ("Dealers", "Banks", 80),
        ("Treasury", "Banks", 120),
        ("Fed", "Banks", 50),
        ("Banks", "MMFs", 60),
    ]

    for source, target, flow in edges:
        G.add_edge(source, target, flow=flow)

    # Initialize contagion model
    contagion = StressContagion(G, damping=0.7)

    # Initial stress: MMFs stressed
    initial_stress = {
        "Fed": 0.1,
        "Banks": 0.2,
        "MMFs": 0.9,  # High stress
        "Treasury": 0.1,
        "Dealers": 0.3,
    }

    print("=== 1-Step Contagion ===")
    one_step = contagion.one_step_contagion(initial_stress)
    for node, stress in one_step.items():
        print(f"{node}: {stress:.3f}")

    print("\n=== 5-Step Evolution ===")
    evolution = contagion.k_step_contagion(initial_stress, k=5)
    print(evolution)

    print("\n=== Superspreaders (Amplification) ===")
    superspreaders = contagion.identify_superspreaders(top_k=3, steps=3)
    for node, amp in superspreaders:
        print(f"{node}: {amp:.3f}x amplification")

    print("\n=== Shock Simulation (shock at Dealers) ===")
    shock = contagion.simulate_shock("Dealers", shock_magnitude=1.0, steps=5)
    print(shock[['step', 'Dealers', 'Banks', 'Fed']].to_string())

    # Feedback model
    print("\n=== Feedback Contagion ===")
    feedback_model = FeedbackContagion(G, damping=0.7, feedback_strength=0.3)
    feedback_result = feedback_model.simulate_with_feedback(initial_stress, steps=5)
    print(feedback_result[['step', 'MMFs', 'Banks', 'Fed']].to_string())
