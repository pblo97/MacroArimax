"""
graph_dynamics.py
Node-level Markov dynamics for liquidity stress states.

Each node maintains its own Hidden Markov state:
- State 0: Calm (normal liquidity conditions)
- State 1: Stressed (tight/abnormal liquidity)

Transitions depend on:
1. Node's own indicators (balance, z-score, flows)
2. Contagion from neighbors (weighted by edge stress)
3. Global regime state
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.special import expit  # Logistic sigmoid
import networkx as nx


class NodeMarkovState:
    """Markov state tracker for a single node."""

    def __init__(
        self,
        node_name: str,
        initial_prob: float = 0.5,
        transition_speed: float = 0.3,
    ):
        """
        Initialize node state.

        Parameters
        ----------
        node_name : str
            Name of the node
        initial_prob : float
            Initial stress probability
        transition_speed : float
            Speed of state transitions (0-1, higher = faster)
        """
        self.name = node_name
        self.stress_prob = initial_prob
        self.transition_speed = transition_speed
        self.history = []

    def update(
        self,
        own_signal: float,
        neighbor_stress: float,
        global_stress: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Update stress probability based on multiple signals.

        Parameters
        ----------
        own_signal : float
            Node's own stress indicator (z-score, percentile, etc.)
        neighbor_stress : float
            Average stress from neighboring nodes (contagion)
        global_stress : float
            Global regime stress level
        weights : dict, optional
            Weights for each signal component

        Returns
        -------
        float
            Updated stress probability
        """
        if weights is None:
            weights = {
                'own': 0.5,
                'neighbor': 0.3,
                'global': 0.2,
            }

        # Combine signals with weights
        combined_signal = (
            weights['own'] * own_signal +
            weights['neighbor'] * neighbor_stress +
            weights['global'] * global_stress
        )

        # Map to probability using logistic function
        target_prob = expit(combined_signal)

        # Smooth transition (moving average)
        self.stress_prob = (
            (1 - self.transition_speed) * self.stress_prob +
            self.transition_speed * target_prob
        )

        # Record history
        self.history.append({
            'stress_prob': self.stress_prob,
            'own_signal': own_signal,
            'neighbor_stress': neighbor_stress,
            'global_stress': global_stress,
        })

        return self.stress_prob

    def get_state(self, threshold: float = 0.5) -> int:
        """Get discrete state (0=calm, 1=stressed)."""
        return 1 if self.stress_prob > threshold else 0

    def reset(self):
        """Reset to initial state."""
        self.stress_prob = 0.5
        self.history = []


class GraphMarkovDynamics:
    """Manage Markov dynamics for all nodes in liquidity graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        transition_speed: float = 0.3,
        contagion_weight: float = 0.3,
    ):
        """
        Initialize graph dynamics.

        Parameters
        ----------
        graph : nx.DiGraph
            Liquidity flow graph
        transition_speed : float
            Speed of state transitions
        contagion_weight : float
            Weight of contagion effect from neighbors
        """
        self.graph = graph
        self.transition_speed = transition_speed
        self.contagion_weight = contagion_weight

        # Initialize node states
        self.node_states = {}
        for node in graph.nodes():
            self.node_states[node] = NodeMarkovState(
                node_name=node,
                transition_speed=transition_speed,
            )

    def compute_node_signal(self, node: str) -> float:
        """
        Compute stress signal from node's own attributes.

        Parameters
        ----------
        node : str
            Node name

        Returns
        -------
        float
            Stress signal (standardized)
        """
        attrs = self.graph.nodes[node]

        # Combine z-score and percentile
        z_score = attrs.get('z_score', 0)
        percentile = attrs.get('percentile', 0.5)

        # Delta signals (negative = draining = stress)
        delta_1d = attrs.get('delta_1d', 0)
        delta_5d = attrs.get('delta_5d', 0)

        # Stress when:
        # - High absolute z-score
        # - Extreme percentiles (very high or very low)
        # - Negative deltas (draining)

        signal = (
            0.4 * z_score +  # Z-score component
            0.3 * (1 - 2 * abs(percentile - 0.5)) +  # Extreme percentiles
            0.3 * (-delta_1d / max(abs(delta_1d), 1e-6))  # Drain direction
        )

        return signal

    def compute_neighbor_stress(self, node: str) -> float:
        """
        Compute average stress from neighboring nodes (contagion).

        Parameters
        ----------
        node : str
            Target node

        Returns
        -------
        float
            Weighted average stress from neighbors
        """
        # Predecessors (nodes flowing into this node)
        predecessors = list(self.graph.predecessors(node))

        # Successors (nodes this node flows to)
        successors = list(self.graph.successors(node))

        all_neighbors = predecessors + successors

        if not all_neighbors:
            return 0.0

        # Weight by edge flow magnitude
        total_stress = 0.0
        total_weight = 0.0

        for neighbor in all_neighbors:
            neighbor_stress = self.node_states[neighbor].stress_prob

            # Get edge weight (flow magnitude)
            if neighbor in predecessors:
                edge_weight = abs(self.graph[neighbor][node].get('flow', 0))
            else:
                edge_weight = abs(self.graph[node][neighbor].get('flow', 0))

            total_stress += neighbor_stress * edge_weight
            total_weight += edge_weight

        if total_weight == 0:
            return np.mean([self.node_states[n].stress_prob for n in all_neighbors])

        return total_stress / total_weight

    def step(self, global_stress: float = 0.0) -> Dict[str, float]:
        """
        Execute one time step of dynamics.

        Parameters
        ----------
        global_stress : float
            Global stress level (from HMM or other global indicator)

        Returns
        -------
        dict
            Updated stress probabilities for all nodes
        """
        # Store new states (update all at once to avoid order effects)
        new_probs = {}

        for node in self.graph.nodes():
            # Own signal
            own_signal = self.compute_node_signal(node)

            # Neighbor contagion
            neighbor_stress = self.compute_neighbor_stress(node)

            # Update state
            new_prob = self.node_states[node].update(
                own_signal=own_signal,
                neighbor_stress=neighbor_stress,
                global_stress=global_stress,
                weights={
                    'own': 0.5,
                    'neighbor': self.contagion_weight,
                    'global': 0.2,
                }
            )

            new_probs[node] = new_prob

        return new_probs

    def simulate(
        self,
        n_steps: int,
        global_stress_series: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Simulate dynamics over multiple steps.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        global_stress_series : np.ndarray, optional
            Time series of global stress (if None, uses 0)

        Returns
        -------
        pd.DataFrame
            Time series of stress probabilities for each node
        """
        if global_stress_series is None:
            global_stress_series = np.zeros(n_steps)

        results = []

        for t in range(n_steps):
            global_stress = global_stress_series[t]
            probs = self.step(global_stress=global_stress)

            results.append({
                'step': t,
                **probs,
            })

        return pd.DataFrame(results)

    def get_steady_state(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        global_stress: float = 0.0,
    ) -> Dict[str, float]:
        """
        Find steady-state stress probabilities.

        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        global_stress : float
            Fixed global stress level

        Returns
        -------
        dict
            Steady-state stress probabilities
        """
        for i in range(max_iter):
            old_probs = {n: s.stress_prob for n, s in self.node_states.items()}
            new_probs = self.step(global_stress=global_stress)

            # Check convergence
            max_change = max(abs(new_probs[n] - old_probs[n]) for n in new_probs)
            if max_change < tol:
                print(f"Converged in {i+1} iterations")
                break

        return new_probs

    def get_state_vector(self) -> np.ndarray:
        """Get current stress probabilities as vector."""
        return np.array([
            self.node_states[node].stress_prob
            for node in self.graph.nodes()
        ])

    def set_state_vector(self, probs: np.ndarray):
        """Set stress probabilities from vector."""
        for i, node in enumerate(self.graph.nodes()):
            self.node_states[node].stress_prob = probs[i]


# Example usage
if __name__ == "__main__":
    # Create simple test graph
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node("Fed", z_score=0.5, percentile=0.7, delta_1d=-10, balance=3000)
    G.add_node("Banks", z_score=-1.0, percentile=0.3, delta_1d=5, balance=2000)
    G.add_node("MMFs", z_score=2.0, percentile=0.9, delta_1d=-20, balance=1000)

    # Add edges with flows
    G.add_edge("Banks", "Fed", flow=-10)
    G.add_edge("MMFs", "Fed", flow=-20)
    G.add_edge("Fed", "Banks", flow=5)

    # Initialize dynamics
    dynamics = GraphMarkovDynamics(G, transition_speed=0.3, contagion_weight=0.3)

    # Simulate
    results = dynamics.simulate(n_steps=20, global_stress_series=np.linspace(0, 1, 20))

    print("Stress probability evolution:")
    print(results[['step', 'Fed', 'Banks', 'MMFs']].tail(10))

    # Steady state
    print("\nSteady state (global_stress=0.5):")
    steady = dynamics.get_steady_state(global_stress=0.5)
    for node, prob in steady.items():
        print(f"{node}: {prob:.3f}")
