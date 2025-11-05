"""
Position Overlay & Playbooks
=============================

Convert stress signals into actionable position sizing and hedging rules.
Automatic playbooks based on hotspot types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PositionRecommendation:
    """Position sizing recommendation."""
    target_beta: float
    reason: str
    confidence: float
    hedge_instruments: List[str]
    avoid_instruments: List[str]
    action_items: List[str]


def compute_target_beta(
    sfi_z: float,
    stress_score: float,
    current_beta: float = 1.0,
    kappa: float = 0.2,
    aggressive: bool = False
) -> float:
    """
    Compute smooth target beta from stress signals.

    Rule: β_t = clip(β_{t-1} + κ·(β* - β_{t-1}), 0, 1)
    where β* = 1 - sigmoid(SFI_z)

    Parameters
    ----------
    sfi_z : float
        Stress Flow Index z-score
    stress_score : float
        Composite stress score (0-2+)
    current_beta : float
        Current beta position
    kappa : float
        Adjustment speed (0.1 = slow, 0.5 = fast)
    aggressive : bool
        If True, use stronger reaction

    Returns
    -------
    float
        Target beta (0-1 range)
    """
    # Compute ideal beta from SFI
    sigmoid_sfi = 1 / (1 + np.exp(-sfi_z))
    beta_star_sfi = 1 - sigmoid_sfi

    # Compute ideal beta from stress score
    sigmoid_stress = 1 / (1 + np.exp(-(stress_score - 1) * 2))  # Center at 1
    beta_star_stress = 1 - sigmoid_stress

    # Combine (equal weight)
    beta_star = (beta_star_sfi + beta_star_stress) / 2

    # Apply stronger adjustment if aggressive
    if aggressive:
        kappa *= 1.5

    # Smooth adjustment
    new_beta = current_beta + kappa * (beta_star - current_beta)

    # Clip to [0, 1]
    return np.clip(new_beta, 0, 1)


def generate_playbook(
    hotspots: List[Tuple[str, str]],
    graph_edges: Dict[Tuple[str, str], Dict],
    stress_flow_index: float,
    quarter_end: bool = False
) -> PositionRecommendation:
    """
    Generate automatic playbook based on hotspot pattern.

    Playbook logic:
    - TGA↑ & ONRRP↑ → "double drain": reduce beta, hedge indices
    - Repo stress (TGCR↑, specials) → hedge UST duration + cash
    - Unsecured (SOFR-EFFR↑) → avoid HY credit, shorten QVM windows
    - Quarter-end → relax thresholds, expect reversals

    Parameters
    ----------
    hotspots : List[Tuple[str, str]]
        List of (source, target) hotspot edges
    graph_edges : Dict
        Full edge data from graph
    stress_flow_index : float
        Current SFI value
    quarter_end : bool
        Whether we're in quarter-end window

    Returns
    -------
    PositionRecommendation
        Structured recommendation
    """
    # Parse hotspot types
    has_tga_drain = False
    has_rrp_drain = False
    has_repo_stress = False
    has_unsecured_stress = False
    has_fhlb_stress = False
    has_ust_liquidity_stress = False
    has_credit_stress = False

    for source, target in hotspots:
        edge_key = (source, target)
        if edge_key not in graph_edges:
            continue

        edge = graph_edges[edge_key]
        driver = edge.get('driver', '').lower()

        if 'tga' in driver or (source == 'Treasury' or target == 'Treasury'):
            has_tga_drain = True
        if 'rrp' in driver or 'onrrp' in driver or (source == 'ON_RRP' or target == 'ON_RRP'):
            has_rrp_drain = True
        if 'repo' in driver or 'tgcr' in driver or 'gc' in driver:
            has_repo_stress = True
        if 'sofr' in driver or 'effr' in driver or 'unsecured' in driver:
            has_unsecured_stress = True
        if 'fhlb' in driver:
            has_fhlb_stress = True
        if 'ust' in driver or 'treasury market' in driver:
            has_ust_liquidity_stress = True
        if 'credit' in driver or 'hy' in driver:
            has_credit_stress = True

    # Initialize recommendation
    target_beta = 1.0
    reason = "Normal conditions"
    confidence = 0.5
    hedge_instruments = []
    avoid_instruments = []
    action_items = []

    # Apply playbook logic
    if quarter_end:
        reason = "Quarter-End Mode: Temporary stress expected"
        confidence = 0.6
        action_items.append("⏰ Quarter-end window: expect reversals post-settlement")
        action_items.append("📊 Monitor reserve identity validation closely")
        target_beta = 0.8

    if has_tga_drain and has_rrp_drain:
        reason = "🚨 DOUBLE DRAIN: TGA↑ & ONRRP↑"
        confidence = 0.9
        target_beta = 0.3
        hedge_instruments = ["SPX Put Spreads", "VIX Call Options", "UST 2Y"]
        avoid_instruments = ["Single-name equities", "Levered ETFs", "EM credit"]
        action_items.append("🔴 Reduce beta aggressively to ~30%")
        action_items.append("🛡️ Prioritize index hedges over single names")
        action_items.append("💵 Raise cash buffer to 20%+")
        action_items.append("📉 Consider shorting small-cap (IWM) vs SPX")

    elif has_repo_stress:
        reason = "⚠️ REPO STRESS: TGCR↑, secured funding tightening"
        confidence = 0.85
        target_beta = 0.5
        hedge_instruments = ["UST 5-10Y futures", "Cash", "SOFR futures (short)"]
        avoid_instruments = ["Derivative-heavy strategies", "High cash-drag names"]
        action_items.append("🟡 Reduce beta to ~50%")
        action_items.append("📈 Hedge with UST duration (not derivatives)")
        action_items.append("💰 Prefer cash over repo for collateral")
        action_items.append("🔍 Monitor TGCR - SOFR spread daily")

    elif has_unsecured_stress:
        reason = "⚠️ UNSECURED STRESS: SOFR-EFFR↑, bank funding pressure"
        confidence = 0.8
        target_beta = 0.6
        hedge_instruments = ["IG Credit CDX", "Bank CDS", "Cash"]
        avoid_instruments = ["New HY issuance", "Bank subordinated debt", "Levered loans"]
        action_items.append("🟡 Reduce beta to ~60%")
        action_items.append("🏦 Avoid new HY credit exposure")
        action_items.append("⏱️ Shorten QVM liquidation windows")
        action_items.append("📊 Monitor SOFR-EFFR and CP spreads")

    elif has_fhlb_stress:
        reason = "⚠️ FHLB STRESS: Advances tightening, bank liquidity strain"
        confidence = 0.75
        target_beta = 0.65
        hedge_instruments = ["Regional bank puts", "Cash", "IG Financials short"]
        avoid_instruments = ["Regional bank stocks", "Mortgage REITs", "Bank loans"]
        action_items.append("🟡 Reduce beta to ~65%")
        action_items.append("🏦 Avoid regional bank exposure")
        action_items.append("📉 Consider hedging financials")

    elif has_ust_liquidity_stress:
        reason = "⚠️ UST LIQUIDITY STRESS: Depth deteriorating"
        confidence = 0.75
        target_beta = 0.65
        hedge_instruments = ["Cash", "Short-duration UST", "Gold"]
        avoid_instruments = ["Long-duration credit", "Illiquid bonds", "EM sovereigns"]
        action_items.append("🟡 Reduce beta to ~65%")
        action_items.append("💎 Shift to higher-quality, liquid assets")
        action_items.append("⏰ Expect volatility in rate-sensitive sectors")

    elif has_credit_stress:
        reason = "⚠️ CREDIT STRESS: HY OAS widening"
        confidence = 0.8
        target_beta = 0.6
        hedge_instruments = ["HY CDX", "IG Credit", "Cash"]
        avoid_instruments = ["HY bonds", "Distressed debt", "Junk-rated loans"]
        action_items.append("🟡 Reduce beta to ~60%")
        action_items.append("📉 Avoid HY credit entirely")
        action_items.append("🔼 Upgrade to IG if maintaining credit exposure")

    elif stress_flow_index > 1.5:
        reason = "⚠️ ELEVATED SFI: General stress building"
        confidence = 0.7
        target_beta = 0.7
        hedge_instruments = ["VIX calls", "SPX put spreads", "Cash"]
        avoid_instruments = ["Momentum stocks", "Levered ETFs"]
        action_items.append("🟡 Reduce beta to ~70%")
        action_items.append("🛡️ Add tail risk hedges")
        action_items.append("📊 Monitor all liquidity channels daily")

    else:
        reason = "✅ LOW STRESS: No major hotspots detected"
        confidence = 0.6
        target_beta = 0.9
        action_items.append("✅ Maintain normal positioning")
        action_items.append("👁️ Stay vigilant for emerging hotspots")

    return PositionRecommendation(
        target_beta=target_beta,
        reason=reason,
        confidence=confidence,
        hedge_instruments=hedge_instruments,
        avoid_instruments=avoid_instruments,
        action_items=action_items
    )


def create_pre_close_checklist(
    graph,
    stress_score: float,
    sfi_z: float,
    quarter_end: bool
) -> Dict[str, any]:
    """
    Pre-close checklist for risk management.

    Returns
    -------
    Dict
        {
            'reserve_residual_ok': bool,
            'quarter_end_flag': bool,
            'hotspots_present': bool,
            'global_regime_tense': bool,
            'mode': str  # "DEFENSE" or "NORMAL"
        }
    """
    # Check reserve residual
    reserve_ok = True
    if hasattr(graph, 'reserve_identity') and len(graph.reserve_identity) > 0:
        residual_pct = abs(graph.reserve_identity['residual_pct'].iloc[-1])
        reserve_ok = residual_pct < 5.0  # <5% error is acceptable

    # Check hotspots
    hotspots_present = len(graph.hotspots) > 0 if hasattr(graph, 'hotspots') else False

    # Check global regime
    global_regime_tense = stress_score > 1.2 or sfi_z > 1.0

    # Determine mode
    if not reserve_ok or quarter_end or hotspots_present or global_regime_tense:
        mode = "DEFENSE"
    else:
        mode = "NORMAL"

    return {
        'reserve_residual_ok': reserve_ok,
        'quarter_end_flag': quarter_end,
        'hotspots_present': hotspots_present,
        'global_regime_tense': global_regime_tense,
        'mode': mode
    }


def compute_rolling_beta_path(
    sfi_series: pd.Series,
    stress_series: pd.Series,
    kappa: float = 0.2,
    initial_beta: float = 1.0
) -> pd.Series:
    """
    Compute full path of target beta over time.

    Parameters
    ----------
    sfi_series : pd.Series
        SFI z-score time series
    stress_series : pd.Series
        Stress score time series
    kappa : float
        Adjustment speed
    initial_beta : float
        Starting beta

    Returns
    -------
    pd.Series
        Target beta over time
    """
    # Align series
    common_idx = sfi_series.index.intersection(stress_series.index)
    sfi = sfi_series.loc[common_idx]
    stress = stress_series.loc[common_idx]

    beta_path = []
    current_beta = initial_beta

    for i in range(len(sfi)):
        sfi_z = sfi.iloc[i]
        stress_val = stress.iloc[i]

        target_beta = compute_target_beta(
            sfi_z=sfi_z,
            stress_score=stress_val,
            current_beta=current_beta,
            kappa=kappa
        )

        beta_path.append(target_beta)
        current_beta = target_beta

    return pd.Series(beta_path, index=common_idx, name='Target_Beta')
