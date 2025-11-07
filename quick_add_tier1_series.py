#!/usr/bin/env python3
"""
Quick Add Tier 1 Series to MacroArimax
========================================

Adds the 10 most critical missing series to series_map.yaml.
These are all FRED series - no external APIs needed.

Run this script to instantly upgrade your system from 3.5 to 4.0 rating.

Usage:
    python quick_add_tier1_series.py
"""

import yaml
from pathlib import Path

# Path to series_map.yaml
SERIES_MAP_PATH = Path(__file__).parent / "macro_plumbing" / "data" / "series_map.yaml"

# Tier 1 Critical Series (10 total)
TIER1_SERIES = {
    # Commercial Paper (CRITICAL for early warning)
    "stress_indicators": {
        "CP_FINANCIAL_3M": {
            "code": "DCPF3M",
            "description": "3-Month AA Financial Commercial Paper Rate",
            "transform": "level",
            "winsorize": False,
            "category": "money_markets"
        },
        "CP_NONFINANCIAL_3M": {
            "code": "DCPN3M",
            "description": "3-Month AA Nonfinancial Commercial Paper Rate",
            "transform": "level",
            "winsorize": False,
            "category": "money_markets"
        }
    },

    # Fed Facilities (CRITICAL for crisis detection)
    "core_plumbing": {
        "DISCOUNT_WINDOW": {
            "code": "WLODLL",
            "description": "Federal Reserve Discount Window Borrowing",
            "transform": "level",
            "winsorize": True,
            "category": "fed_facilities"
        },
        "PRIMARY_CREDIT": {
            "code": "H41RESPPALDKNWW",
            "description": "Primary Credit Outstanding (Discount Window)",
            "transform": "level",
            "winsorize": True,
            "category": "fed_facilities"
        }
    },

    # Repo Markets (HIGH VALUE)
    "reference_rates": {
        "BGCR": {
            "code": "BGCR",
            "description": "Broad General Collateral Rate",
            "transform": "level",
            "winsorize": False,
            "category": "repo"
        },
        "SOFR_30D_AVG": {
            "code": "SOFR30DAYAVG",
            "description": "30-Day Average SOFR",
            "transform": "level",
            "winsorize": False,
            "category": "repo"
        },
        "SOFR_90D_AVG": {
            "code": "SOFR90DAYAVG",
            "description": "90-Day Average SOFR",
            "transform": "level",
            "winsorize": False,
            "category": "repo"
        }
    },

    # Credit Markets (HIGH VALUE)
    "market_indicators": {
        "CORP_AAA_OAS": {
            "code": "BAMLC0A0CM",
            "description": "ICE BofA AAA US Corporate OAS",
            "transform": "level",
            "winsorize": True,
            "category": "credit"
        },
        "CORP_BBB_OAS": {
            "code": "BAMLC0A4CBBB",
            "description": "ICE BofA BBB US Corporate OAS",
            "transform": "level",
            "winsorize": True,
            "category": "credit"
        },
        "DOLLAR_INDEX": {
            "code": "DTWEXBGS",
            "description": "Trade Weighted U.S. Dollar Index: Broad",
            "transform": "level",
            "winsorize": False,
            "category": "fx"
        }
    }
}

# Derived features to add
DERIVED_FEATURES = {
    "cp_tbill_spread": {
        "formula": "CP_FINANCIAL_3M - TB3MS",
        "description": "Commercial Paper - T-Bill spread (funding stress)"
    },
    "cp_nonfinancial_spread": {
        "formula": "CP_NONFINANCIAL_3M - TB3MS",
        "description": "Nonfinancial CP - T-Bill spread"
    },
    "discount_window_alarm": {
        "formula": "DISCOUNT_WINDOW > 5000",
        "description": "Discount Window usage alarm (>$5B = crisis)"
    },
    "bgcr_sofr_spread": {
        "formula": "BGCR - SOFR",
        "description": "BGCR-SOFR spread (collateral scarcity)"
    },
    "sofr_term_premium": {
        "formula": "SOFR_90D_AVG - SOFR",
        "description": "SOFR term premium (90-day vs overnight)"
    },
    "bbb_aaa_spread": {
        "formula": "CORP_BBB_OAS - CORP_AAA_OAS",
        "description": "BBB-AAA spread (credit stress)"
    },
    "dollar_strength_zscore": {
        "formula": "zscore(DOLLAR_INDEX, window=252)",
        "description": "Dollar strength z-score (USD shortage indicator)"
    }
}


def add_tier1_series():
    """Add Tier 1 series to series_map.yaml"""

    print("="*80)
    print("ADDING TIER 1 SERIES TO MACROARIMAX")
    print("="*80)
    print()

    # Load existing config
    print(f"Loading {SERIES_MAP_PATH}...")
    with open(SERIES_MAP_PATH, "r") as f:
        config = yaml.safe_load(f)

    print(f"✅ Loaded existing config ({len(config)} top-level categories)")
    print()

    # Add Tier 1 series
    added_count = 0
    for category, series in TIER1_SERIES.items():
        if category not in config:
            config[category] = {}

        for series_name, series_info in series.items():
            if series_name not in config[category]:
                config[category][series_name] = series_info
                added_count += 1
                print(f"✅ Added: {series_name} ({series_info['code']})")
            else:
                print(f"⚠️  Skipped: {series_name} (already exists)")

    print()
    print(f"Added {added_count} new series")
    print()

    # Add derived features
    if "derived_features" not in config:
        config["derived_features"] = {}

    derived_added = 0
    for feature_name, feature_info in DERIVED_FEATURES.items():
        if feature_name not in config["derived_features"]:
            config["derived_features"][feature_name] = feature_info
            derived_added += 1
            print(f"✅ Added derived: {feature_name}")

    print()
    print(f"Added {derived_added} derived features")
    print()

    # Save updated config
    print(f"Saving updated config to {SERIES_MAP_PATH}...")
    with open(SERIES_MAP_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("✅ Config saved successfully")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"New series added: {added_count}")
    print(f"New derived features: {derived_added}")
    print()
    print("Categories updated:")
    for category in TIER1_SERIES.keys():
        count = len(TIER1_SERIES[category])
        print(f"  - {category}: +{count} series")

    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Test locally:")
    print("   streamlit run macro_plumbing/app/app.py")
    print()
    print("2. Verify new columns appear:")
    print("   - CP_FINANCIAL_3M, CP_NONFINANCIAL_3M")
    print("   - DISCOUNT_WINDOW, PRIMARY_CREDIT")
    print("   - BGCR, SOFR_30D_AVG, SOFR_90D_AVG")
    print("   - CORP_AAA_OAS, CORP_BBB_OAS, DOLLAR_INDEX")
    print()
    print("3. Check derived features:")
    print("   - cp_tbill_spread")
    print("   - discount_window_alarm")
    print("   - bgcr_sofr_spread")
    print("   - bbb_aaa_spread")
    print()
    print("4. Commit and push:")
    print("   git add macro_plumbing/data/series_map.yaml")
    print("   git commit -m 'Add Tier 1 critical series (10 new series)'")
    print("   git push")
    print()
    print("Expected improvement: 3.5/5.0 → 4.0/5.0")
    print("="*80)


if __name__ == "__main__":
    add_tier1_series()
