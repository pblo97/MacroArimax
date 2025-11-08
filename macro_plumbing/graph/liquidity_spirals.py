"""
liquidity_spirals.py
Fire sale cascades and liquidity spirals with procyclical haircuts.

Implements Brunnermeier-Pedersen (2009) framework:
- Market liquidity ↔ Funding liquidity feedback
- Fire sales → Price impact → Haircut increases → More sales

Based on:
- ESRB (2025) Systemic Liquidity Risk Framework
- IMF (2022) Systemwide Liquidity Stress Testing
- ECB FSR (2024) NBFI vulnerabilities
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Asset:
    """Asset for fire sale modeling."""
    name: str
    market_depth: float  # $ billions
    lambda_impact: float  # Price impact coefficient
    baseline_haircut: float  # Normal haircut (e.g., 0.10 = 10%)
    haircut_sensitivity: float  # How quickly haircut responds to price


@dataclass
class Institution:
    """Institution that can experience fire sales."""
    name: str
    holdings: Dict[str, float]  # {asset_name: holdings_$B}
    leverage: float  # Total assets / equity
    margin_call_threshold: float  # Trigger level (e.g., 0.15 = 15% equity loss)


class LiquiditySpiralModel:
    """
    Model fire sale cascades with price-haircut feedback loops.

    Key mechanisms:
    1. Shock → Forced sales
    2. Sales → Price impact (Δp = -λ * volume / depth)
    3. Price ↓ → Haircut ↑ (procyclical)
    4. Haircut ↑ → Margin calls → More sales
    5. Repeat until convergence

    Based on Brunnermeier-Pedersen (2009), ESRB (2025)
    """

    def __init__(
        self,
        assets: List[Asset],
        institutions: List[Institution],
        max_iterations: int = 20,
        convergence_tol: float = 0.01,
    ):
        """
        Initialize liquidity spiral model.

        Parameters
        ----------
        assets : List[Asset]
            Assets that can be sold
        institutions : List[Institution]
            Institutions that can experience fire sales
        max_iterations : int
            Max cascade iterations
        convergence_tol : float
            Convergence threshold (as fraction of initial sales)
        """
        self.assets = {a.name: a for a in assets}
        self.institutions = {i.name: i for i in institutions}
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        # Initialize state
        self.current_prices = {a: 1.0 for a in self.assets}
        self.current_haircuts = {a: self.assets[a].baseline_haircut
                                 for a in self.assets}

    def simulate_cascade(
        self,
        initial_shock: Dict[str, float],
        shock_type: str = 'portfolio_loss'
    ) -> pd.DataFrame:
        """
        Simulate fire sale cascade.

        Parameters
        ----------
        initial_shock : dict
            Initial shock: {institution: shock_magnitude_$B}
        shock_type : str
            'portfolio_loss': Loss to portfolio value
            'redemption': Redemption shock (AUM withdrawal)
            'margin_call': Sudden margin call

        Returns
        -------
        pd.DataFrame
            Cascade evolution with prices, haircuts, sales by iteration
        """
        results = []

        # Reset to initial state
        self.current_prices = {a: 1.0 for a in self.assets}
        self.current_haircuts = {
            a: self.assets[a].baseline_haircut
            for a in self.assets
        }

        # Track cumulative sales
        cumulative_sales = {a: 0.0 for a in self.assets}

        for iteration in range(self.max_iterations):
            # 1. Compute forced sales
            if iteration == 0:
                # Initial shock
                forced_sales = self._compute_initial_sales(
                    initial_shock,
                    shock_type
                )
            else:
                # Margin calls from price decline
                forced_sales = self._compute_margin_call_sales(
                    self.current_prices,
                    self.current_haircuts
                )

            # 2. Price impact
            price_impact = self._compute_price_impact(forced_sales)

            # Update prices
            new_prices = {}
            for asset in self.assets:
                new_prices[asset] = self.current_prices[asset] * (
                    1 + price_impact[asset]
                )
                # Floor at 10% of initial price
                new_prices[asset] = max(new_prices[asset], 0.10)

            # 3. Update haircuts (procyclical)
            new_haircuts = self._update_haircuts(new_prices)

            # 4. Store results
            cumulative_sales = {
                asset: cumulative_sales[asset] + forced_sales[asset]
                for asset in self.assets
            }

            results.append({
                'iteration': iteration,
                **{f'{asset}_price': new_prices[asset]
                   for asset in self.assets},
                **{f'{asset}_haircut': new_haircuts[asset]
                   for asset in self.assets},
                **{f'{asset}_sales': forced_sales[asset]
                   for asset in self.assets},
                **{f'{asset}_cumulative_sales': cumulative_sales[asset]
                   for asset in self.assets},
            })

            # Update state
            self.current_prices = new_prices
            self.current_haircuts = new_haircuts

            # 5. Check convergence
            total_sales = sum(forced_sales.values())
            if iteration > 0 and total_sales < self.convergence_tol:
                print(f"Converged at iteration {iteration}")
                break

        return pd.DataFrame(results)

    def _compute_initial_sales(
        self,
        shock: Dict[str, float],
        shock_type: str
    ) -> Dict[str, float]:
        """
        Compute initial forced sales from shock.

        Parameters
        ----------
        shock : dict
            {institution: shock_magnitude}
        shock_type : str
            Type of shock

        Returns
        -------
        dict
            {asset: forced_sales_volume_$B}
        """
        forced_sales = {a: 0.0 for a in self.assets}

        for inst_name, shock_mag in shock.items():
            inst = self.institutions[inst_name]

            if shock_type == 'portfolio_loss':
                # Loss reduces equity
                # Need to deleverage: sell assets
                # Delever amount = loss * leverage
                delever_amount = shock_mag * inst.leverage

            elif shock_type == 'redemption':
                # Redemption: need to raise cash
                delever_amount = shock_mag

            elif shock_type == 'margin_call':
                # Margin call: post collateral
                delever_amount = shock_mag
            else:
                delever_amount = shock_mag

            # Allocate sales proportionally across holdings
            total_holdings = sum(inst.holdings.values())
            if total_holdings > 0:
                for asset, holding in inst.holdings.items():
                    sale_fraction = holding / total_holdings
                    forced_sales[asset] += delever_amount * sale_fraction

        return forced_sales

    def _compute_margin_call_sales(
        self,
        prices: Dict[str, float],
        haircuts: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute forced sales from margin calls.

        Margin call triggered when:
        Equity / Assets < threshold

        Forced sale = (Required equity - Actual equity) / (1 - haircut)
        """
        forced_sales = {a: 0.0 for a in self.assets}

        for inst in self.institutions.values():
            # Compute current equity
            current_portfolio_value = sum(
                inst.holdings[asset] * prices[asset]
                for asset in inst.holdings
            )

            # Assuming initial equity = portfolio_value / leverage
            initial_equity = current_portfolio_value / inst.leverage
            current_equity = current_portfolio_value - (
                (inst.leverage - 1) * initial_equity / inst.leverage * current_portfolio_value
            )

            equity_ratio = current_equity / current_portfolio_value if current_portfolio_value > 0 else 0

            # Margin call?
            if equity_ratio < inst.margin_call_threshold:
                # Need to raise equity ratio back to threshold
                required_equity = inst.margin_call_threshold * current_portfolio_value
                equity_shortfall = max(required_equity - current_equity, 0)

                # Sell assets to meet shortfall
                # Sale proceeds = sale_amount * price * (1 - haircut)
                for asset in inst.holdings:
                    if inst.holdings[asset] > 0:
                        haircut = haircuts[asset]
                        # Allocate proportionally
                        sale_fraction = inst.holdings[asset] / sum(inst.holdings.values())
                        sale_amount = (equity_shortfall * sale_fraction) / (
                            (1 - haircut) * prices[asset]
                        )
                        forced_sales[asset] += sale_amount

        return forced_sales

    def _compute_price_impact(
        self,
        sales_volume: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute price impact of sales.

        Price impact function (Kyle 1985):
        Δp / p = -λ * (volume / market_depth)

        λ (lambda) calibrated from historical episodes:
        - Treasuries: λ = 0.01-0.05 (very liquid)
        - IG Corporate: λ = 0.05-0.15 (liquid)
        - HY Corporate: λ = 0.15-0.50 (less liquid)
        - Structured: λ = 0.50-2.00 (illiquid)

        Parameters
        ----------
        sales_volume : dict
            {asset: volume_$B}

        Returns
        -------
        dict
            {asset: Δp/p} (fractional price change, negative)
        """
        impact = {}

        for asset_name, volume in sales_volume.items():
            asset = self.assets[asset_name]

            # Kyle's lambda impact
            fractional_impact = -asset.lambda_impact * (
                volume / asset.market_depth
            )

            # Floor at -50% per iteration (avoid numerical instability)
            impact[asset_name] = max(fractional_impact, -0.50)

        return impact

    def _update_haircuts(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update haircuts (procyclical).

        Haircut function (Geanakoplos 2010):
        haircut(t) = baseline + sensitivity * max(0, (1 - price(t)))

        When price ↓ → haircut ↑

        Calibrated from historical episodes:
        - 2008: Subprime MBS haircuts 0% → 50% (sensitivity ~0.5)
        - 2020 March: IG corporate haircuts 2% → 10% (sensitivity ~0.2)

        Parameters
        ----------
        prices : dict
            {asset: current_price}

        Returns
        -------
        dict
            {asset: haircut}
        """
        new_haircuts = {}

        for asset_name, price in prices.items():
            asset = self.assets[asset_name]

            # Procyclical haircut
            price_decline = max(1.0 - price, 0)  # How much below par
            haircut = asset.baseline_haircut + (
                asset.haircut_sensitivity * price_decline
            )

            # Bound haircuts [0, 0.95]
            new_haircuts[asset_name] = np.clip(haircut, 0.0, 0.95)

        return new_haircuts

    def compute_amplification_ratio(self, results: pd.DataFrame) -> float:
        """
        Compute amplification ratio.

        Ratio = Total sales (all iterations) / Initial sales

        Ratio > 1 → amplification
        Ratio < 1 → dampening

        Parameters
        ----------
        results : pd.DataFrame
            Cascade results from simulate_cascade()

        Returns
        -------
        float
            Amplification ratio
        """
        if len(results) == 0:
            return 1.0

        # Initial sales (iteration 0)
        initial_sales = sum(
            results.iloc[0][f'{asset}_sales']
            for asset in self.assets
        )

        # Total cumulative sales (last iteration)
        total_sales = sum(
            results.iloc[-1][f'{asset}_cumulative_sales']
            for asset in self.assets
        )

        if initial_sales == 0:
            return 1.0

        return total_sales / initial_sales


def estimate_market_depth_from_fred(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate market depth from FRED data.

    Proxies:
    - Treasury depth: Average daily volume (SIFMA data if available)
    - Corporate depth: Outstanding amount * turnover ratio
    - HY depth: Outstanding * turnover (lower)

    Parameters
    ----------
    df : pd.DataFrame
        FRED data

    Returns
    -------
    dict
        {asset: market_depth_$B}
    """
    # Rough estimates (would calibrate with actual data)
    market_depths = {
        'UST_10Y': 50_000,  # $50B daily volume
        'IG_Corporate': 5_000,  # $5B daily
        'HY_Corporate': 500,  # $500M daily
        'Agency_MBS': 10_000,  # $10B daily
    }

    return market_depths


def calibrate_lambda_from_crisis(
    crisis_data: pd.DataFrame,
    asset: str
) -> float:
    """
    Calibrate λ (price impact) from historical crisis.

    Method:
    1. Identify fire sale episodes (large volume + price decline)
    2. Estimate: Δp/p = -λ * (volume / depth)
    3. Solve for λ

    Parameters
    ----------
    crisis_data : pd.DataFrame
        Data from crisis episode (e.g., March 2020, March 2023)
    asset : str
        Asset name

    Returns
    -------
    float
        Calibrated λ
    """
    # Example for March 2020 Treasury market
    # (would use actual TRACE/FINRA data)

    # Observed: 10Y Treasury dropped 30bp in March 16-20, 2020
    # Estimated forced sales: ~$100B (dealers + mutual funds)
    # Market depth: ~$50B/day

    delta_price = -0.003  # -30bp = -0.3%
    volume = 100  # $B
    depth = 50  # $B

    # Solve: -0.003 = -λ * (100 / 50)
    lambda_estimate = abs(delta_price) / (volume / depth)

    return lambda_estimate


# ==================
# Example Usage
# ==================

if __name__ == "__main__":
    # Define assets
    assets = [
        Asset(
            name='UST_10Y',
            market_depth=50_000,  # $50B daily
            lambda_impact=0.02,  # Liquid
            baseline_haircut=0.02,  # 2% haircut
            haircut_sensitivity=0.10
        ),
        Asset(
            name='IG_Corporate',
            market_depth=5_000,
            lambda_impact=0.10,
            baseline_haircut=0.05,
            haircut_sensitivity=0.20
        ),
        Asset(
            name='HY_Corporate',
            market_depth=500,
            lambda_impact=0.30,  # Illiquid
            baseline_haircut=0.15,
            haircut_sensitivity=0.50
        ),
    ]

    # Define institutions
    institutions = [
        Institution(
            name='Hedge_Fund_1',
            holdings={'UST_10Y': 10_000, 'IG_Corporate': 5_000, 'HY_Corporate': 2_000},
            leverage=5.0,
            margin_call_threshold=0.15
        ),
        Institution(
            name='Mutual_Fund_1',
            holdings={'UST_10Y': 20_000, 'IG_Corporate': 15_000, 'HY_Corporate': 5_000},
            leverage=1.1,  # Low leverage
            margin_call_threshold=0.05
        ),
    ]

    # Initialize model
    model = LiquiditySpiralModel(assets, institutions)

    # Simulate shock: Hedge Fund 1 experiences $1B portfolio loss
    initial_shock = {'Hedge_Fund_1': 1_000}  # $1B loss

    results = model.simulate_cascade(
        initial_shock,
        shock_type='portfolio_loss'
    )

    print("\n" + "="*70)
    print("LIQUIDITY SPIRAL SIMULATION")
    print("="*70)
    print(f"\nInitial shock: ${initial_shock['Hedge_Fund_1']/1000:.1f}B portfolio loss (Hedge Fund 1)")
    print(f"\nIterations to convergence: {len(results)}")

    # Final prices
    print("\n" + "="*70)
    print("FINAL PRICES (as % of initial)")
    print("="*70)
    for asset in assets:
        final_price = results.iloc[-1][f'{asset.name}_price']
        print(f"{asset.name:20s} {final_price:>6.1%}")

    # Final haircuts
    print("\n" + "="*70)
    print("FINAL HAIRCUTS")
    print("="*70)
    for asset in assets:
        initial_haircut = asset.baseline_haircut
        final_haircut = results.iloc[-1][f'{asset.name}_haircut']
        change = final_haircut - initial_haircut
        print(f"{asset.name:20s} {initial_haircut:>5.1%} → {final_haircut:>5.1%} (Δ{change:>+5.1%})")

    # Total sales
    print("\n" + "="*70)
    print("CUMULATIVE FORCED SALES")
    print("="*70)
    for asset in assets:
        total_sales = results.iloc[-1][f'{asset.name}_cumulative_sales']
        print(f"{asset.name:20s} ${total_sales:>10,.0f}M")

    # Amplification
    amplification = model.compute_amplification_ratio(results)
    print("\n" + "="*70)
    print(f"AMPLIFICATION RATIO: {amplification:.2f}x")
    print("="*70)
    if amplification > 1.5:
        print("🔴 HIGH AMPLIFICATION - Liquidity spiral risk")
    elif amplification > 1.1:
        print("🟡 MODERATE AMPLIFICATION")
    else:
        print("✅ LOW AMPLIFICATION")

    print("\n" + "="*70)
    print("DETAILED EVOLUTION")
    print("="*70)
    print(results[['iteration', 'UST_10Y_price', 'IG_Corporate_price', 'HY_Corporate_price',
                   'UST_10Y_haircut', 'IG_Corporate_haircut', 'HY_Corporate_haircut']].to_string(index=False))
