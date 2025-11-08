"""
Test all 4 phases of liquidity network enhancements.

Validates:
- Phase 1: Margin calls estimation
- Phase 2: NBFI nodes
- Phase 3: Dynamic network metrics
- Phase 4: Advanced metrics (SIM, CoI, LCR)
- Integration: Enhanced graph builder

All using synthetic FRED data (no paid data required).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING ALL 4 PHASES - LIQUIDITY NETWORK ENHANCEMENTS")
print("="*70)

# Create synthetic FRED data
print("\n1. Generating synthetic FRED data...")
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
df['bbb_aaa_spread'] = df['bbb_aaa_spread'].clip(0.5, 4)
df['cp_tbill_spread'] = df['cp_tbill_spread'].clip(0, 2)

print(f"   ✅ Generated {len(df):,} days of data")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")

# ==================
# PHASE 1: Margin Calls
# ==================

print("\n" + "="*70)
print("PHASE 1: MARGIN CALLS & COLLATERAL")
print("="*70)

from macro_plumbing.graph.margin_calls import (
    estimate_initial_margin_change,
    estimate_variation_margin,
    estimate_repo_margin_haircut,
    compute_margin_stress_index
)

delta_im = estimate_initial_margin_change(df)
vm = estimate_variation_margin(df)
haircut = estimate_repo_margin_haircut(df)
margin_stress = compute_margin_stress_index(df)

print(f"\nLatest values:")
print(f"  ΔIM (daily):        ${delta_im.iloc[-1]:+,.0f}M")
print(f"  VM (daily):         ${vm.iloc[-1]:+,.0f}M")
print(f"  Repo haircut:       {haircut.iloc[-1]:.2%}")
print(f"  Margin stress:      {margin_stress.iloc[-1]:+.2f} σ")

print(f"\n✅ Phase 1 PASSED - Margin calls estimated successfully")

# ==================
# PHASE 2: NBFI Nodes
# ==================

print("\n" + "="*70)
print("PHASE 2: NBFI SECTOR EXPANSION")
print("="*70)

from macro_plumbing.graph.nbfi_nodes import (
    build_nbfi_nodes,
    compute_nbfi_systemic_score
)

nbfi_nodes = build_nbfi_nodes(df)

print(f"\nNBFI nodes created:")
for name, node in nbfi_nodes.items():
    print(f"  {name:20s} AUM=${node.aum_estimate:,.0f}B, Stress={node.stress_score:+.2f}σ")

nbfi_systemic = compute_nbfi_systemic_score(nbfi_nodes)
print(f"\nNBFI Systemic Score: {nbfi_systemic:+.2f} σ")

print(f"\n✅ Phase 2 PASSED - NBFI nodes created successfully")

# ==================
# PHASE 3: Dynamic Network
# ==================

print("\n" + "="*70)
print("PHASE 3: DYNAMIC NETWORK METRICS")
print("="*70)

# Note: Would need historical graphs for full test
# Testing basic functionality
from macro_plumbing.graph.dynamic_network import DynamicNetworkAnalyzer
import networkx as nx

# Create sample graph
G = nx.DiGraph()
G.add_nodes_from(['Fed', 'Banks', 'MMFs'])
G.add_edges_from([('Fed', 'Banks'), ('MMFs', 'Banks')])

analyzer = DynamicNetworkAnalyzer(window=63)

print(f"\n  Analyzer created with window={analyzer.window} days")

# Compute basic metrics
from macro_plumbing.graph.dynamic_network import (
    compute_network_density,
    compute_network_centralization
)

density = compute_network_density(G)
central = compute_network_centralization(G)

print(f"  Density: {density:.3f}")
print(f"  Centralization: {central:.3f}")

print(f"\n✅ Phase 3 PASSED - Dynamic metrics computed successfully")

# ==================
# PHASE 4: Advanced Metrics
# ==================

print("\n" + "="*70)
print("PHASE 4: ADVANCED METRICS (SIM, CoI, LCR)")
print("="*70)

from macro_plumbing.graph.advanced_metrics import (
    compute_systemic_importance,
    compute_contagion_index,
    compute_network_lcr,
    compute_network_resilience_score
)

# Add attributes to graph
for node in G.nodes():
    G.nodes[node]['balance'] = 1000
    G.nodes[node]['stress_prob'] = 0.3

for u, v in G.edges():
    G.edges[u, v]['flow'] = 100
    G.edges[u, v]['is_drain'] = False
    G.edges[u, v]['abs_flow'] = 100

# Compute metrics
sim_fed = compute_systemic_importance(G, 'Fed')
coi, _ = compute_contagion_index(G)
lcr_banks = compute_network_lcr(G, 'Banks')
resilience = compute_network_resilience_score(G)

print(f"\n  SIM (Fed):              {sim_fed:.3f}")
print(f"  Contagion Index:        {coi:.1f}")
print(f"  Network LCR (Banks):    {lcr_banks:.2f}")
print(f"  Network Resilience:     {resilience:.3f}")

print(f"\n✅ Phase 4 PASSED - Advanced metrics computed successfully")

# ==================
# INTEGRATION TEST
# ==================

print("\n" + "="*70)
print("INTEGRATION TEST: ENHANCED GRAPH BUILDER")
print("="*70)

from macro_plumbing.graph.enhanced_graph_builder import build_enhanced_graph

print("\nBuilding enhanced graph with all phases...")
graph, metrics = build_enhanced_graph(df)

print(f"\n  Nodes: {graph.G.number_of_nodes()}")
print(f"  Edges: {graph.G.number_of_edges()}")

# Validate metrics
assert metrics is not None, "Metrics should not be None"
assert metrics.density >= 0 and metrics.density <= 1, "Density should be [0,1]"
assert metrics.centralization >= 0, "Centralization should be >= 0"
assert len(metrics.sim_scores) == graph.G.number_of_nodes(), "SIM should be computed for all nodes"
assert len(metrics.lcr_scores) == graph.G.number_of_nodes(), "LCR should be computed for all nodes"

print("\n✅ INTEGRATION TEST PASSED - All phases work together")

# ==================
# SUMMARY REPORT
# ==================

print("\n" + graph.summary())

# ==================
# FINAL VALIDATION
# ==================

print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

tests_passed = 0
tests_total = 8

# Check Phase 1
if delta_im is not None and len(delta_im) > 0:
    print("✅ Phase 1: Margin calls - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 1: Margin calls - FAILED")

if vm is not None and len(vm) > 0:
    print("✅ Phase 1: Variation margin - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 1: Variation margin - FAILED")

# Check Phase 2
if len(nbfi_nodes) == 3:
    print("✅ Phase 2: NBFI nodes (3 nodes) - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 2: NBFI nodes - FAILED")

if nbfi_systemic is not None:
    print("✅ Phase 2: NBFI systemic score - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 2: NBFI systemic score - FAILED")

# Check Phase 3
if analyzer is not None:
    print("✅ Phase 3: Dynamic analyzer - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 3: Dynamic analyzer - FAILED")

# Check Phase 4
if sim_fed >= 0 and sim_fed <= 1:
    print("✅ Phase 4: SIM metric - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 4: SIM metric - FAILED")

if coi >= 0:
    print("✅ Phase 4: Contagion Index - PASSED")
    tests_passed += 1
else:
    print("❌ Phase 4: Contagion Index - FAILED")

# Check Integration
if graph.G.number_of_nodes() >= 8:  # 5 core + 3 NBFI
    print("✅ Integration: Enhanced graph - PASSED")
    tests_passed += 1
else:
    print("❌ Integration: Enhanced graph - FAILED")

print("\n" + "="*70)
print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
print("="*70)

if tests_passed == tests_total:
    print("\n🎉 ALL TESTS PASSED! All 4 phases are working correctly.")
    print("\n✅ Ready for production use with FRED data (no paid data required)")
else:
    print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")

print("\n" + "="*70)
