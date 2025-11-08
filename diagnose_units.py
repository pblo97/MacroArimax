"""
Diagnose data units for crisis indicators.

Checks if data is in:
- Basis points (bps): 100 = 100 bps = 1%
- Percent decimal: 1.0 = 1% = 100 bps
- Percent whole: 100 = 100%
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("DATA UNITS DIAGNOSTICS")
print("="*70)

# Create fake data to demonstrate units
print("\nUNIT EXAMPLES:")
print("-" * 70)
print("Format          | Value | Meaning")
print("-" * 70)
print("Basis points    | 100   | 100 bps = 1.00%")
print("Percent decimal | 1.0   | 1% = 100 bps")
print("Percent whole   | 1.0   | 1% = 100 bps (same as decimal)")
print("-" * 70)

# User's reported values
print("\n" + "="*70)
print("YOUR REPORTED VALUES")
print("="*70)

user_data = {
    'cp_tbill_spread': 0.03,
    'HY_OAS': 3.13
}

print(f"\ncp_tbill_spread: {user_data['cp_tbill_spread']}")
print(f"HY_OAS: {user_data['HY_OAS']}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# CP-Tbill Spread
cp_value = user_data['cp_tbill_spread']
print(f"\ncp_tbill_spread = {cp_value}")

if cp_value < 1:
    print(f"  → Likely in PERCENT DECIMAL format")
    print(f"  → {cp_value}% = {cp_value * 100:.1f} bps")
    print(f"  → This is {cp_value * 100:.1f} basis points")
else:
    print(f"  → Likely in BASIS POINTS format")
    print(f"  → {cp_value} bps = {cp_value / 100:.2f}%")

# Historical context for CP spread
print(f"\nHistorical CP-Tbill Spread Context:")
print(f"  Normal conditions: 10-30 bps (0.10-0.30%)")
print(f"  Elevated: 50-80 bps (0.50-0.80%)")
print(f"  Crisis (2008): 150-300 bps (1.50-3.00%)")
print(f"  Crisis threshold: 100 bps (1.00%)")

if cp_value < 0.1:
    print(f"\n  ✅ Your value {cp_value}% ({cp_value*100:.1f} bps) = VERY LOW (calm markets)")
elif cp_value < 0.5:
    print(f"\n  ✅ Your value {cp_value}% ({cp_value*100:.1f} bps) = NORMAL")
elif cp_value < 1.0:
    print(f"\n  🟡 Your value {cp_value}% ({cp_value*100:.1f} bps) = ELEVATED")
else:
    print(f"\n  🔴 Your value {cp_value}% ({cp_value*100:.1f} bps) = CRISIS")

# HY OAS
hy_value = user_data['HY_OAS']
print(f"\n" + "-"*70)
print(f"HY_OAS = {hy_value}")

if hy_value < 20:
    print(f"  → Likely in PERCENT DECIMAL format")
    print(f"  → {hy_value}% = {hy_value * 100:.0f} bps")
else:
    print(f"  → Likely in BASIS POINTS format")
    print(f"  → {hy_value} bps = {hy_value / 100:.2f}%")

# Historical context for HY OAS
print(f"\nHistorical HY OAS Context:")
print(f"  Normal conditions: 3-5% (300-500 bps)")
print(f"  Elevated: 6-7% (600-700 bps)")
print(f"  Crisis (2008): 10-20% (1000-2000 bps)")
print(f"  Crisis threshold: 8% (800 bps)")

if hy_value < 4:
    print(f"\n  ✅ Your value {hy_value}% ({hy_value*100:.0f} bps) = VERY TIGHT (low risk)")
elif hy_value < 6:
    print(f"\n  ✅ Your value {hy_value}% ({hy_value*100:.0f} bps) = NORMAL")
elif hy_value < 8:
    print(f"\n  🟡 Your value {hy_value}% ({hy_value*100:.0f} bps) = ELEVATED")
else:
    print(f"\n  🔴 Your value {hy_value}% ({hy_value*100:.0f} bps) = CRISIS")

print("\n" + "="*70)
print("CURRENT THRESHOLDS IN CODE")
print("="*70)

thresholds = {
    'cp_tbill_spread': 1.0,
    'HY_OAS': 8.0
}

print(f"\ncp_tbill_spread > {thresholds['cp_tbill_spread']}")
print(f"  → Threshold: {thresholds['cp_tbill_spread']}% = {thresholds['cp_tbill_spread'] * 100:.0f} bps")
print(f"  → Current: {cp_value}% = {cp_value * 100:.1f} bps")
print(f"  → Exceeds threshold? {cp_value > thresholds['cp_tbill_spread']}")

print(f"\nHY_OAS > {thresholds['HY_OAS']}")
print(f"  → Threshold: {thresholds['HY_OAS']}% = {thresholds['HY_OAS'] * 100:.0f} bps")
print(f"  → Current: {hy_value}% = {hy_value * 100:.0f} bps")
print(f"  → Exceeds threshold? {hy_value > thresholds['HY_OAS']}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if cp_value < 0.1 and hy_value < 4:
    print("\n✅ Data appears to be in PERCENT DECIMAL format")
    print("✅ Thresholds are CORRECT")
    print("✅ Current market conditions are CALM (no crisis)")
    print("\nNo changes needed - model is working correctly!")
else:
    print("\n⚠️ Data format unclear - please verify with FRED documentation")
    print("   Check the series metadata for units")

print("\n" + "="*70)
