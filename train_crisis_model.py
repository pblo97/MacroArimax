#!/usr/bin/env python3
"""
Train and Backtest Crisis Prediction Model

This script:
1. Loads FRED data with all features
2. Trains Random Forest crisis classifier
3. Backtests with time-series cross-validation
4. Shows feature importance
5. Predicts current crisis probability

Usage:
    python train_crisis_model.py

Requirements:
    - FRED_API_KEY environment variable
    - Data fetched (or will fetch automatically)

Author: MacroArimax
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from macro_plumbing.data.fred_client import FREDClient
from macro_plumbing.models.crisis_classifier import CrisisPredictor


def main():
    print("="*80)
    print("CRISIS PREDICTION MODEL - TRAINING & BACKTESTING")
    print("="*80)
    print()

    # Check for API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ ERROR: FRED_API_KEY environment variable not set")
        print()
        print("Set it with:")
        print("  export FRED_API_KEY='your_key_here'")
        print()
        print("Get a key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    print(f"✅ FRED API key found: {api_key[:8]}...")
    print()

    # Load or fetch data
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    cache_file = Path('.fred_cache/full_data.pkl')

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_pickle(cache_file)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    else:
        print("Fetching data from FRED API (this may take a minute)...")
        fred = FREDClient(api_key=api_key)

        # Fetch all series
        df = fred.fetch_all(start_date='2010-01-01')

        # Compute derived features
        print("Computing derived features...")
        df = fred.compute_derived_features(df)

        print(f"✅ Fetched {len(df)} rows, {len(df.columns)} columns")

        # Cache for next time
        cache_file.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
        print(f"Cached to {cache_file}")

    print()
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total days: {len(df)}")
    print()

    # Show key features
    print("Sample features available:")
    key_features = [
        'VIX', 'cp_tbill_spread', 'HY_OAS', 'bbb_aaa_spread',
        'DISCOUNT_WINDOW', 'credit_cascade', 'T10Y2Y'
    ]
    for feat in key_features:
        if feat in df.columns:
            latest = df[feat].dropna().iloc[-1] if feat in df.columns else np.nan
            print(f"  {feat:25s} {latest:>10.2f}")

    # Train model
    print()
    print("="*80)
    print("STEP 2: TRAINING MODEL")
    print("="*80)
    print()

    # Split: train on all data except last year (for final prediction)
    train_end = df.index[-252] if len(df) > 252 else df.index[-1]  # Leave 1 year for testing

    predictor = CrisisPredictor(horizon=5)
    predictor.train(df.loc[:train_end])

    # Backtest
    print()
    print("="*80)
    print("STEP 3: BACKTESTING")
    print("="*80)

    results = predictor.backtest(df)

    # Current prediction
    print()
    print("="*80)
    print("STEP 4: CURRENT PREDICTION")
    print("="*80)

    # Get last 30 days of predictions
    recent_df = df.iloc[-30:]
    recent_proba = predictor.predict_proba(recent_df)

    print()
    print("Last 10 Days Crisis Probability:")
    print(f"{'Date':<12} {'Probability':>12} {'Status':>15}")
    print("-"*40)

    for date, proba in zip(recent_df.index[-10:], recent_proba[-10:]):
        if proba > 0.70:
            status = "🔴 CRISIS LIKELY"
        elif proba > 0.50:
            status = "🟠 ELEVATED"
        elif proba > 0.30:
            status = "🟡 MODERATE"
        else:
            status = "🟢 NORMAL"

        print(f"{str(date)[:10]:<12} {proba:>11.1%} {status:>15}")

    # Current status
    current_proba = recent_proba[-1]
    current_date = recent_df.index[-1]

    print()
    print("="*80)
    print(f"CURRENT STATUS ({current_date.strftime('%Y-%m-%d')})")
    print("="*80)
    print()
    print(f"Crisis Probability (next 5 days): {current_proba:.1%}")
    print()

    if current_proba > 0.70:
        print("🔴 STATUS: CRISIS LIKELY")
        print("   ⚠️  High probability of liquidity stress event")
        print("   ⚠️  Consider risk reduction measures")
    elif current_proba > 0.50:
        print("🟠 STATUS: ELEVATED RISK")
        print("   ⚠️  Elevated probability of stress")
        print("   Monitor closely for deterioration")
    elif current_proba > 0.30:
        print("🟡 STATUS: MODERATE RISK")
        print("   Some stress indicators elevated")
        print("   Normal monitoring recommended")
    else:
        print("🟢 STATUS: NORMAL")
        print("   Low probability of crisis")
        print("   Markets functioning normally")

    # Explain prediction
    print()
    print("="*80)
    print("STEP 5: PREDICTION EXPLANATION")
    print("="*80)

    predictor.explain_prediction(df, date=current_date)

    # Save model
    print()
    print("="*80)
    print("STEP 6: SAVING MODEL")
    print("="*80)

    import pickle
    model_file = Path('macro_plumbing/models/trained_crisis_predictor.pkl')
    model_file.parent.mkdir(exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump(predictor, f)

    print(f"✅ Model saved to {model_file}")
    print()

    print("="*80)
    print("DONE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review backtest results above")
    print("2. Check feature importance")
    print("3. Monitor daily crisis probability")
    print("4. Integrate into Streamlit dashboard")
    print()
    print("To load trained model:")
    print("  import pickle")
    print(f"  with open('{model_file}', 'rb') as f:")
    print("      predictor = pickle.load(f)")
    print("  proba = predictor.predict_proba(df)")
    print()


if __name__ == "__main__":
    main()
