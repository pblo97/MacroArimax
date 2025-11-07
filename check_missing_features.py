#!/usr/bin/env python3
"""
Check which derived features are missing from the dashboard.
"""

# Features the user HAS (69 total)
user_columns = [
    "BANK_RESERVES_WEEKLY",
    "BREAKEVEN_10Y",
    "BREAKEVEN_5Y",
    "BUILDING_PERMITS",
    "CAPACITY_UTILIZATION",
    "CI_LOANS",
    "CONSUMER_CREDIT",
    "CONTINUED_CLAIMS",
    "CORP_AAA_OAS",
    "CORP_BBB_OAS",
    "CORP_BB_OAS",
    "CORP_CCC_OAS",
    "CP_FINANCIAL_3M",
    "CP_NONFINANCIAL_3M",
    "CREDIT_CARD_DELINQ",
    "DGS10",
    "DGS3MO",
    "DISCOUNT_WINDOW",
    "DOLLAR_INDEX",
    "EFFR",
    "HOUSING_STARTS",
    "HY_OAS",
    "INDUSTRIAL_PRODUCTION",
    "JAPAN_10Y",
    "JOBLESS_CLAIMS",
    "LABOR_PARTICIPATION",
    "MORTGAGE_30Y",
    "NFCI",
    "OBFR",
    "PART_TIME_ECONOMIC",
    "PRIMARY_CREDIT",
    "REAL_ESTATE_LOANS",
    "RESERVES",
    "RETAIL_SALES",
    "RRP",
    "SOFR",
    "SOFR_30D_AVG",
    "SOFR_90D_AVG",
    "SP500",
    "STLFSI4",
    "T10Y2Y",
    "TB3MS",
    "TGA",
    "TIPS_10Y",
    "TOTAL_BANK_CREDIT",
    "UK_10Y",
    "UNEMPLOYMENT_RATE",
    "VIX",
    "WALCL",
    "WTI_OIL",
    "YIELD_2Y",
    "YIELD_30Y",
    "YIELD_5Y",
    "convenience_yield",
    "delta_net_liquidity",
    "delta_reserves",
    "delta_rrp",
    "delta_tga",
    "eur_usd_3m_basis",
    "mbs_holdings",
    "month_end",
    "net_liquidity",
    "obfr_sofr_spread",
    "quarter_end",
    "sofr_effr_spread",
    "spread",
    "total_holdings",
    "vrp",
    "year_end"
]

# Features that SHOULD exist (from Tier 1, 2, 3)
expected_derived_features = [
    # Tier 1 (7 features)
    "cp_tbill_spread",
    "cp_nonfinancial_spread",
    "discount_window_alarm",
    "bgcr_sofr_spread",
    "sofr_term_premium",
    "bbb_aaa_spread",
    "dollar_strength_zscore",

    # Tier 2 (12 features)
    "bb_bbb_spread",
    "ccc_bb_spread",
    "credit_cascade",
    "real_rate_5y",
    "real_rate_10y",
    "breakeven_slope",
    "term_spread_10y30y",
    "term_spread_5y10y",
    "term_spread_2y5y",
    "delta_bank_credit",
    "delta_ci_loans",
    "jobless_claims_zscore",

    # Tier 3 (13 features)
    "gold_zscore",
    "oil_zscore",
    "vix_alarm",
    "euribor_ois_proxy",
    "us_japan_spread",
    "continued_claims_zscore",
    "labor_slack",
    "mortgage_spread",
    "housing_momentum",
    "consumer_credit_growth",
    "delinquency_index",
    "retail_sales_momentum",
    "capacity_gap",

    # Additional from original
    "tgcr_sofr_spread",
]

# Check which are missing
missing = [f for f in expected_derived_features if f not in user_columns]

print("="*80)
print("MISSING DERIVED FEATURES")
print("="*80)
print(f"\nTotal missing: {len(missing)}/{len(expected_derived_features)}")
print(f"\nUser has: {len(user_columns)} columns")
print(f"Should have: {len(user_columns) + len(missing)} columns\n")

print("Missing features by tier:\n")

tier1 = ["cp_tbill_spread", "cp_nonfinancial_spread", "discount_window_alarm",
         "bgcr_sofr_spread", "sofr_term_premium", "bbb_aaa_spread", "dollar_strength_zscore"]
tier1_missing = [f for f in tier1 if f in missing]

tier2 = ["bb_bbb_spread", "ccc_bb_spread", "credit_cascade", "real_rate_5y",
         "real_rate_10y", "breakeven_slope", "term_spread_10y30y", "term_spread_5y10y",
         "term_spread_2y5y", "delta_bank_credit", "delta_ci_loans", "jobless_claims_zscore"]
tier2_missing = [f for f in tier2 if f in missing]

tier3 = ["gold_zscore", "oil_zscore", "vix_alarm", "euribor_ois_proxy", "us_japan_spread",
         "continued_claims_zscore", "labor_slack", "mortgage_spread", "housing_momentum",
         "consumer_credit_growth", "delinquency_index", "retail_sales_momentum", "capacity_gap"]
tier3_missing = [f for f in tier3 if f in missing]

print(f"TIER 1 MISSING ({len(tier1_missing)}/{len(tier1)}):")
for f in tier1_missing:
    print(f"  ❌ {f}")

print(f"\nTIER 2 MISSING ({len(tier2_missing)}/{len(tier2)}):")
for f in tier2_missing:
    print(f"  ❌ {f}")

print(f"\nTIER 3 MISSING ({len(tier3_missing)}/{len(tier3)}):")
for f in tier3_missing:
    print(f"  ❌ {f}")

# Also check for series that might be missing
expected_series = [
    "BGCR",  # For bgcr_sofr_spread
    "TGCR",  # For tgcr_sofr_spread
    "GOLD_PRICE",  # For gold_zscore
    "EURIBOR_3M",  # For euribor_ois_proxy
    "AUTO_LOAN_DELINQ",  # For delinquency_index
]

missing_series = [s for s in expected_series if s not in user_columns]

if missing_series:
    print(f"\n{'='*80}")
    print("MISSING RAW SERIES (preventing feature computation)")
    print("="*80)
    for s in missing_series:
        print(f"  ❌ {s}")
