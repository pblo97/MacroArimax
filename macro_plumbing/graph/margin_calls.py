"""
margin_calls.py
Estimate margin call flows using FRED data proxies.

Based on IMF FSAP 2025, ECB FSR 2024 methodology.

Data sources (all FRED):
- VIX: Proxy for volatility → Initial Margin increases
- HY_OAS, BBB_AAA: Proxy for credit stress → Variation Margin
- MOVE: Treasury volatility → IM for rates derivatives
- FX swap implied vol: Proxy for FX margin (if available)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def estimate_initial_margin_change(
    df: pd.DataFrame,
    notional_estimate: float = 10_000_000  # $10T notional (rough estimate)
) -> pd.Series:
    """
    Estimate change in Initial Margin requirements.

    IM = f(volatility, confidence_level, horizon)

    ISDA SIMM approach:
    IM ≈ Notional * σ * sqrt(horizon) * z_score

    Proxies from FRED:
    - VIX: Equity vol
    - MOVE: Rates vol
    - HY_OAS changes: Credit spread vol

    Parameters
    ----------
    df : pd.DataFrame
        FRED data with VIX, MOVE, HY_OAS
    notional_estimate : float
        Estimated derivative notional (industry-wide)

    Returns
    -------
    pd.Series
        Estimated daily ΔIM (in $M)
    """
    # Parameters
    horizon_days = 10  # MPOR (Margin Period of Risk) - standard
    confidence_z = 2.33  # 99% confidence
    sqrt_horizon = np.sqrt(horizon_days / 252)

    # 1. Equity component (VIX)
    vix = df.get('VIX', pd.Series(0, index=df.index)).ffill()
    equity_vol = vix / 100  # Convert to decimal
    equity_im = notional_estimate * 0.3 * equity_vol * sqrt_horizon * confidence_z

    # 2. Rates component (MOVE)
    move = df.get('MOVE', pd.Series(0, index=df.index)).ffill()
    # MOVE is in bps, convert to decimal
    rates_vol = move / 10000
    rates_im = notional_estimate * 0.4 * rates_vol * sqrt_horizon * confidence_z

    # 3. Credit component (HY_OAS volatility)
    hy_oas = df.get('HY_OAS', pd.Series(0, index=df.index)).ffill()
    # Use rolling vol of HY_OAS as credit vol proxy
    credit_vol = hy_oas.pct_change().rolling(21).std().fillna(0)
    credit_im = notional_estimate * 0.3 * credit_vol * sqrt_horizon * confidence_z

    # Total IM
    total_im = equity_im + rates_im + credit_im

    # Change in IM (what matters for flows)
    delta_im = total_im.diff().fillna(0)

    return delta_im


def estimate_variation_margin(
    df: pd.DataFrame,
    notional_estimate: float = 10_000_000
) -> pd.Series:
    """
    Estimate Variation Margin flows.

    VM = MTM changes daily

    Proxies from FRED:
    - S&P 500 returns: Equity derivatives MTM
    - 10Y Treasury returns: Rates derivatives MTM
    - HY_OAS changes: Credit derivatives MTM

    Parameters
    ----------
    df : pd.DataFrame
        FRED data
    notional_estimate : float
        Estimated notional

    Returns
    -------
    pd.Series
        Daily VM flows (in $M)
    """
    # 1. Equity VM (from equity returns)
    sp500 = df.get('SP500', pd.Series(0, index=df.index)).ffill()
    equity_returns = sp500.pct_change().fillna(0)
    equity_vm = notional_estimate * 0.3 * equity_returns

    # 2. Rates VM (from 10Y Treasury returns)
    dgs10 = df.get('DGS10', pd.Series(0, index=df.index)).ffill()
    # Duration approximation: -duration * Δy
    duration = 8.0  # 10Y bond duration ~8 years
    rates_returns = -duration * dgs10.diff() / 100  # bps to decimal
    rates_vm = notional_estimate * 0.4 * rates_returns

    # 3. Credit VM (from HY_OAS changes)
    hy_oas = df.get('HY_OAS', pd.Series(0, index=df.index)).ffill()
    credit_returns = -hy_oas.diff() / 100  # Wider spread = loss
    credit_vm = notional_estimate * 0.3 * credit_returns

    # Total VM
    total_vm = equity_vm + rates_vm + credit_vm

    return total_vm.fillna(0)


def estimate_repo_margin_haircut(
    df: pd.DataFrame
) -> pd.Series:
    """
    Estimate procyclical repo haircuts.

    Haircut = baseline + sensitivity * stress_indicators

    Stress indicators (FRED):
    - VIX (market stress)
    - HY_OAS (credit stress)
    - MOVE (rates volatility)

    Based on Gorton-Metrick (2012), Adrian-Shin (2010)

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    pd.Series
        Estimated repo haircut (as decimal, e.g., 0.05 = 5%)
    """
    # Baseline haircuts (normal times)
    baseline_treasury = 0.02  # 2% for Treasuries
    baseline_agency = 0.04  # 4% for Agency MBS
    baseline_corporate = 0.10  # 10% for corporate

    # Get stress indicators
    vix = df.get('VIX', pd.Series(15, index=df.index)).fillna(15)
    hy_oas = df.get('HY_OAS', pd.Series(4, index=df.index)).fillna(4)
    move = df.get('MOVE', pd.Series(80, index=df.index)).fillna(80)

    # Normalize stress indicators (z-score)
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    hy_z = (hy_oas - hy_oas.rolling(252).mean()) / hy_oas.rolling(252).std()
    move_z = (move - move.rolling(252).mean()) / move.rolling(252).std()

    # Fill NaN
    vix_z = vix_z.fillna(0)
    hy_z = hy_z.fillna(0)
    move_z = move_z.fillna(0)

    # Composite stress
    stress = (vix_z + hy_z + move_z) / 3

    # Haircut adjustments (procyclical)
    # Positive stress → higher haircut
    haircut_adjustment = 0.01 * stress  # 1% per 1 std stress

    # Treasury haircut
    haircut_treasury = baseline_treasury + haircut_adjustment
    haircut_treasury = haircut_treasury.clip(0.01, 0.50)

    # Agency haircut
    haircut_agency = baseline_agency + 1.5 * haircut_adjustment
    haircut_agency = haircut_agency.clip(0.02, 0.70)

    # Corporate haircut
    haircut_corporate = baseline_corporate + 2.0 * haircut_adjustment
    haircut_corporate = haircut_corporate.clip(0.05, 0.90)

    # Weighted average (based on repo market composition)
    # ~60% Treasury, 30% Agency, 10% Corporate
    haircut = (
        0.60 * haircut_treasury +
        0.30 * haircut_agency +
        0.10 * haircut_corporate
    )

    return haircut


def compute_margin_stress_index(df: pd.DataFrame) -> pd.Series:
    """
    Composite margin stress index.

    Combines:
    - ΔIM (procyclical)
    - |VM| (MTM volatility)
    - Haircut increases

    Returns z-score normalized index.

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    pd.Series
        Margin stress index (z-score)
    """
    # Components
    delta_im = estimate_initial_margin_change(df)
    vm = estimate_variation_margin(df)
    haircut = estimate_repo_margin_haircut(df)

    # Normalize each component
    im_z = (delta_im - delta_im.rolling(252).mean()) / delta_im.rolling(252).std()
    vm_z = (vm.abs() - vm.abs().rolling(252).mean()) / vm.abs().rolling(252).std()
    haircut_z = (haircut - haircut.rolling(252).mean()) / haircut.rolling(252).std()

    # Composite (equal weight)
    margin_stress = (im_z + vm_z + haircut_z) / 3

    return margin_stress.fillna(0)


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
        'SP500': 3000 * np.exp(0.0001 * np.arange(n) + 0.01 * np.random.randn(n).cumsum()),
        'DGS10': 2.0 + 0.5 * np.random.randn(n).cumsum() * 0.1,
    }, index=dates)

    # Clip to realistic ranges
    df['VIX'] = df['VIX'].clip(10, 80)
    df['MOVE'] = df['MOVE'].clip(50, 200)
    df['HY_OAS'] = df['HY_OAS'].clip(2, 15)

    print("="*70)
    print("MARGIN CALL ESTIMATION (FRED PROXIES)")
    print("="*70)

    # Estimate components
    delta_im = estimate_initial_margin_change(df)
    vm = estimate_variation_margin(df)
    haircut = estimate_repo_margin_haircut(df)
    margin_stress = compute_margin_stress_index(df)

    # Latest values
    latest = df.index[-1]
    print(f"\nLatest date: {latest.strftime('%Y-%m-%d')}")
    print(f"\nMarket conditions:")
    print(f"  VIX:    {df['VIX'].iloc[-1]:.1f}")
    print(f"  MOVE:   {df['MOVE'].iloc[-1]:.1f}")
    print(f"  HY OAS: {df['HY_OAS'].iloc[-1]:.2f}%")

    print(f"\nMargin estimates:")
    print(f"  ΔIM (daily):     ${delta_im.iloc[-1]:+,.0f}M")
    print(f"  VM (daily):      ${vm.iloc[-1]:+,.0f}M")
    print(f"  Repo haircut:    {haircut.iloc[-1]:.2%}")

    print(f"\nMargin stress index: {margin_stress.iloc[-1]:+.2f} σ")

    if margin_stress.iloc[-1] > 2:
        print("  🔴 SEVERE margin stress")
    elif margin_stress.iloc[-1] > 1:
        print("  🟡 ELEVATED margin stress")
    else:
        print("  ✅ NORMAL margin conditions")

    # Summary stats
    print(f"\n{'='*70}")
    print("HISTORICAL STATISTICS (last 252 days)")
    print(f"{'='*70}")

    recent = df.iloc[-252:]
    recent_im = delta_im.iloc[-252:]
    recent_vm = vm.iloc[-252:]
    recent_haircut = haircut.iloc[-252:]

    print(f"\nΔIM (daily):")
    print(f"  Mean:  ${recent_im.mean():+,.0f}M")
    print(f"  Std:   ${recent_im.std():,.0f}M")
    print(f"  Min:   ${recent_im.min():+,.0f}M")
    print(f"  Max:   ${recent_im.max():+,.0f}M")

    print(f"\nVM (daily):")
    print(f"  Mean:  ${recent_vm.mean():+,.0f}M")
    print(f"  |Mean|: ${recent_vm.abs().mean():,.0f}M")
    print(f"  Std:   ${recent_vm.std():,.0f}M")

    print(f"\nRepo haircut:")
    print(f"  Mean:  {recent_haircut.mean():.2%}")
    print(f"  Min:   {recent_haircut.min():.2%}")
    print(f"  Max:   {recent_haircut.max():.2%}")

    print(f"\n{'='*70}")
