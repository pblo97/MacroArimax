"""
Force retrain the model to use only 3 features (delete old cached model).
"""

import sys
from pathlib import Path

# Find and delete old model
model_path = Path("macro_plumbing/models/trained_crisis_predictor.pkl")

if model_path.exists():
    print(f"Found old model at: {model_path}")
    print("This model was trained with 5 features (VIX, HY_OAS, cp_tbill_spread, T10Y2Y, NFCI)")
    print()

    # Delete it
    model_path.unlink()
    print("✅ DELETED old model")
    print()
    print("Next steps:")
    print("1. Reload the Streamlit app")
    print("2. The model will automatically retrain with only 3 features:")
    print("   - cp_tbill_spread (VIF=1.24)")
    print("   - T10Y2Y (VIF=1.96)")
    print("   - NFCI (VIF=1.99)")
    print()
    print("3. Verify in the UI that only 3 features appear in 'Prediction Explanation'")
else:
    print("No cached model found - model will train automatically on next run")
    print()
    print("Expected features (3 only):")
    print("   - cp_tbill_spread (VIF=1.24)")
    print("   - T10Y2Y (VIF=1.96)")
    print("   - NFCI (VIF=1.99)")
