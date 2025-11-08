"""
nbfi_nodes.py
Add NBFI (Non-Bank Financial Intermediary) nodes using FRED proxies.

Based on ECB FSR 2024, ESRB 2025, IMF GFSR 2025.

NBFIs to add:
1. Hedge Funds - High leverage, derivatives, redemption risk
2. Asset Managers - Mutual funds, ETFs, flow-driven
3. Insurance/Pensions - Long-term, but vulnerable to rates

All estimated using FREE FRED data (no proprietary data needed).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class NBFINode:
    """NBFI node data."""
    name: str
    type: str
    aum_estimate: float  # Assets Under Management ($B)
    leverage_ratio: float
    liquidity_ratio: float  # Liquid assets / Total assets
    stress_score: float  # Current stress level


def estimate_hedge_fund_stress(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    """
    Estimate hedge fund stress using FRED proxies.

    Proxies:
    - VIX: Volatility hurts long-short equity
    - HY_OAS: Credit stress hurts credit funds
    - MOVE: Rates vol hurts relative value funds
    - CP-Tbill spread: Funding stress
    - Leverage estimate: ~3-5x typical

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    tuple
        (stress_index, metadata)
    """
    # Industry estimates (no direct FRED data)
    INDUSTRY_AUM = 4_000  # ~$4T global HF AUM
    AVG_LEVERAGE = 3.5  # Typical 3-5x

    # Stress components
    vix = df.get('VIX', pd.Series(15, index=df.index)).fillna(15)
    hy_oas = df.get('HY_OAS', pd.Series(4, index=df.index)).fillna(4)
    move = df.get('MOVE', pd.Series(80, index=df.index)).fillna(80)
    cp_tbill = df.get('cp_tbill_spread', pd.Series(0.1, index=df.index)).fillna(0.1)

    # Normalize
    vix_z = (vix - 15) / 10  # Normalize around 15 ± 10
    hy_z = (hy_oas - 4) / 2  # Normalize around 4% ± 2%
    move_z = (move - 80) / 30  # Normalize around 80 ± 30
    cp_z = (cp_tbill - 0.1) / 0.2  # Normalize around 10bp ± 20bp

    # Composite stress (weighted by HF strategy mix)
    # ~40% equity, 30% credit, 20% rates, 10% macro
    hf_stress = (
        0.40 * vix_z +
        0.30 * hy_z +
        0.20 * move_z +
        0.10 * cp_z
    )

    metadata = {
        'aum_estimate': INDUSTRY_AUM,
        'leverage': AVG_LEVERAGE,
        'liquidity_ratio': 0.30,  # ~30% liquid
        'synthetic_leverage': True,  # Derivatives exposure
    }

    return hf_stress, metadata


def estimate_asset_manager_stress(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    """
    Estimate asset manager (mutual funds, ETFs) stress.

    Proxies:
    - VIX: Equity fund stress
    - HY_OAS: Bond fund stress
    - Flow stress: Rapid outflows (proxy from credit spreads widening)

    Redemption risk is KEY vulnerability (ECB FSR 2024)

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    tuple
        (stress_index, metadata)
    """
    # Industry estimates
    INDUSTRY_AUM = 25_000  # ~$25T US mutual fund + ETF AUM
    AVG_LEVERAGE = 1.05  # Very low leverage (reg limits)

    # Stress components
    vix = df.get('VIX', pd.Series(15, index=df.index)).fillna(15)
    hy_oas = df.get('HY_OAS', pd.Series(4, index=df.index)).fillna(4)

    # Flow stress proxy: Sharp credit widening → redemptions
    # Use HY_OAS changes as proxy
    hy_change = hy_oas.diff().rolling(21).mean()  # 1-month change

    # Normalize
    vix_z = (vix - 15) / 10
    hy_z = (hy_oas - 4) / 2
    flow_z = hy_change / hy_change.rolling(252).std()  # Normalize flow stress

    # Composite (equal weight equity, credit, flow)
    am_stress = (
        0.40 * vix_z +
        0.30 * hy_z +
        0.30 * flow_z.fillna(0)
    )

    metadata = {
        'aum_estimate': INDUSTRY_AUM,
        'leverage': AVG_LEVERAGE,
        'liquidity_ratio': 0.80,  # ~80% liquid (daily redemption)
        'redemption_risk': True,
    }

    return am_stress, metadata


def estimate_insurance_pension_stress(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    """
    Estimate insurance/pension fund stress.

    Key vulnerability: Duration mismatch (ECB FSR 2024)

    Proxies:
    - DGS10: Rates level (low rates = underfunding)
    - MOVE: Rates volatility (marks assets)
    - Credit spreads: Asset portfolio stress

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    tuple
        (stress_index, metadata)
    """
    # Industry estimates
    INDUSTRY_AUM = 35_000  # ~$35T US insurance + pensions
    AVG_LEVERAGE = 1.10  # Low leverage but duration mismatch

    # Stress components
    dgs10 = df.get('DGS10', pd.Series(3, index=df.index)).fillna(3)
    move = df.get('MOVE', pd.Series(80, index=df.index)).fillna(80)
    bbb_aaa = df.get('bbb_aaa_spread', pd.Series(1.5, index=df.index)).fillna(1.5)

    # Low rates = stress (underfunding)
    # Target rate ~4%, below that is stress
    rate_stress = (4.0 - dgs10) / 2.0  # Normalize

    # Volatility stress
    move_z = (move - 80) / 30

    # Credit stress
    credit_z = (bbb_aaa - 1.5) / 0.5

    # Composite
    ins_stress = (
        0.50 * rate_stress +  # Rates most important
        0.30 * move_z +
        0.20 * credit_z
    )

    metadata = {
        'aum_estimate': INDUSTRY_AUM,
        'leverage': AVG_LEVERAGE,
        'liquidity_ratio': 0.20,  # ~20% liquid (long-term liabilities)
        'duration_mismatch': True,
    }

    return ins_stress, metadata


def build_nbfi_nodes(df: pd.DataFrame) -> Dict[str, NBFINode]:
    """
    Build all NBFI nodes from FRED data.

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    dict
        {node_name: NBFINode}
    """
    nodes = {}

    # 1. Hedge Funds
    hf_stress, hf_meta = estimate_hedge_fund_stress(df)
    nodes['Hedge_Funds'] = NBFINode(
        name='Hedge_Funds',
        type='nbfi_hedge',
        aum_estimate=hf_meta['aum_estimate'],
        leverage_ratio=hf_meta['leverage'],
        liquidity_ratio=hf_meta['liquidity_ratio'],
        stress_score=hf_stress.iloc[-1] if len(hf_stress) > 0 else 0
    )

    # 2. Asset Managers
    am_stress, am_meta = estimate_asset_manager_stress(df)
    nodes['Asset_Managers'] = NBFINode(
        name='Asset_Managers',
        type='nbfi_am',
        aum_estimate=am_meta['aum_estimate'],
        leverage_ratio=am_meta['leverage'],
        liquidity_ratio=am_meta['liquidity_ratio'],
        stress_score=am_stress.iloc[-1] if len(am_stress) > 0 else 0
    )

    # 3. Insurance/Pensions
    ins_stress, ins_meta = estimate_insurance_pension_stress(df)
    nodes['Insurance_Pensions'] = NBFINode(
        name='Insurance_Pensions',
        type='nbfi_ins',
        aum_estimate=ins_meta['aum_estimate'],
        leverage_ratio=ins_meta['leverage'],
        liquidity_ratio=ins_meta['liquidity_ratio'],
        stress_score=ins_stress.iloc[-1] if len(ins_stress) > 0 else 0
    )

    return nodes


def compute_nbfi_systemic_score(nodes: Dict[str, NBFINode]) -> float:
    """
    Compute aggregate NBFI systemic risk score.

    Weighted by AUM and leverage.

    Parameters
    ----------
    nodes : dict
        NBFI nodes

    Returns
    -------
    float
        Systemic risk score (z-score)
    """
    total_risk = 0
    total_weight = 0

    for node in nodes.values():
        # Weight by AUM * leverage (exposure)
        weight = node.aum_estimate * node.leverage_ratio
        total_risk += weight * node.stress_score
        total_weight += weight

    if total_weight == 0:
        return 0

    systemic_score = total_risk / total_weight

    return systemic_score


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
        'MOVE': 80 + 20 * np.random.randn(n).cumsum() * 0.1,
        'HY_OAS': 4 + 2 * np.random.randn(n).cumsum() * 0.1,
        'DGS10': 3.0 + 0.5 * np.random.randn(n).cumsum() * 0.1,
        'bbb_aaa_spread': 1.5 + 0.3 * np.random.randn(n).cumsum() * 0.1,
        'cp_tbill_spread': 0.1 + 0.05 * np.random.randn(n).cumsum() * 0.05,
    }, index=dates)

    # Clip to realistic ranges
    df['VIX'] = df['VIX'].clip(10, 80)
    df['MOVE'] = df['MOVE'].clip(50, 200)
    df['HY_OAS'] = df['HY_OAS'].clip(2, 15)
    df['DGS10'] = df['DGS10'].clip(0.5, 6)
    df['bbb_aaa_spread'] = df['bbb_aaa_spread'].clip(0.5, 4)
    df['cp_tbill_spread'] = df['cp_tbill_spread'].clip(0, 2)

    print("="*70)
    print("NBFI STRESS ESTIMATION (FRED PROXIES)")
    print("="*70)

    # Build nodes
    nodes = build_nbfi_nodes(df)

    # Display
    print(f"\n{'='*70}")
    print("NBFI NODES")
    print(f"{'='*70}")

    for name, node in nodes.items():
        print(f"\n{node.name}:")
        print(f"  Type:           {node.type}")
        print(f"  AUM estimate:   ${node.aum_estimate:,.0f}B")
        print(f"  Leverage:       {node.leverage_ratio:.2f}x")
        print(f"  Liquidity ratio: {node.liquidity_ratio:.1%}")
        print(f"  Stress score:   {node.stress_score:+.2f} σ")

        if node.stress_score > 2:
            print(f"  Status:         🔴 SEVERE stress")
        elif node.stress_score > 1:
            print(f"  Status:         🟡 ELEVATED stress")
        elif node.stress_score > -1:
            print(f"  Status:         ✅ NORMAL")
        else:
            print(f"  Status:         💚 CALM")

    # Systemic score
    systemic = compute_nbfi_systemic_score(nodes)

    print(f"\n{'='*70}")
    print(f"AGGREGATE NBFI SYSTEMIC RISK: {systemic:+.2f} σ")
    print(f"{'='*70}")

    if systemic > 2:
        print("🔴 SEVERE systemic risk from NBFI sector")
    elif systemic > 1:
        print("🟡 ELEVATED systemic risk from NBFI sector")
    else:
        print("✅ NORMAL NBFI sector conditions")

    print(f"\n{'='*70}")
    print("MARKET CONDITIONS (current)")
    print(f"{'='*70}")
    latest = df.index[-1]
    print(f"Date:           {latest.strftime('%Y-%m-%d')}")
    print(f"VIX:            {df['VIX'].iloc[-1]:.1f}")
    print(f"MOVE:           {df['MOVE'].iloc[-1]:.1f}")
    print(f"HY OAS:         {df['HY_OAS'].iloc[-1]:.2f}%")
    print(f"10Y Treasury:   {df['DGS10'].iloc[-1]:.2f}%")
    print(f"BBB-AAA spread: {df['bbb_aaa_spread'].iloc[-1]:.2f}%")
    print(f"CP-Tbill:       {df['cp_tbill_spread'].iloc[-1]:.2f}%")

    print(f"\n{'='*70}")
