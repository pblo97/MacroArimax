#!/usr/bin/env python3
"""
Quick test of Logistic Regression implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from macro_plumbing.models.crisis_classifier import CrisisPredictor

# Load synthetic data
df = pd.read_pickle('.fred_cache/full_data.pkl')

print("="*60)
print("TESTING LOGISTIC REGRESSION IMPLEMENTATION")
print("="*60)
print()

# Train
predictor = CrisisPredictor(horizon=5)
predictor.train(df)

# Predict
print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

recent_proba = predictor.predict_proba(df.tail(10))

print("\nLast 10 days:")
for date, proba in zip(df.tail(10).index, recent_proba):
    status = "🔴 CRISIS" if proba > 0.7 else "🟡 ELEVATED" if proba > 0.5 else "🟢 NORMAL"
    print(f"{str(date)[:10]}  {proba:>6.1%}  {status}")

# Explain last prediction
predictor.explain_prediction(df)

print("\n✅ Test complete!")
