"""
Quick Test Script for Scraping System
======================================

Verifica que el sistema de scraping funciona correctamente.

Uso:
    python test_scraping.py

Muestra:
- ✅ Datos que se obtuvieron exitosamente
- ❌ Datos que fallaron
- 📊 Summary de calidad de datos
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("🧪 TESTING SCRAPING SYSTEM")
print("="*80)

# Test 1: Test infrastructure
print("\n[1/5] Testing infrastructure...")
try:
    from macro_plumbing.data.scraping_infrastructure import cache, RateLimiter, retry_with_backoff
    print("   ✅ Infrastructure imported successfully")
except Exception as e:
    print(f"   ❌ Infrastructure import failed: {e}")
    sys.exit(1)

# Test 2: Test FRBNY scrapers
print("\n[2/5] Testing FRBNY scrapers...")
try:
    from macro_plumbing.data.frbny_scrapers import fetch_dealer_leverage

    print("   Testing dealer leverage scraper...")
    df_dealer = fetch_dealer_leverage()

    if df_dealer is not None and len(df_dealer) > 0:
        print(f"   ✅ Dealer Leverage: {len(df_dealer)} rows")
        print(f"      Latest: {df_dealer.index[-1].strftime('%Y-%m-%d')}")
        print(f"      Value: {df_dealer.iloc[-1, 0]:.2f}")
    else:
        print("   ⚠️  Dealer Leverage: Using cached/proxy data")

except Exception as e:
    print(f"   ❌ FRBNY scrapers failed: {e}")

# Test 3: Test International scrapers
print("\n[3/5] Testing International scrapers...")
try:
    from macro_plumbing.data.international_scrapers import (
        fetch_ecb_fx_basis,
        compute_variance_risk_premium,
        compute_convenience_yield
    )

    print("   Testing FX Basis scraper...")
    df_fx = fetch_ecb_fx_basis()
    if df_fx is not None and len(df_fx) > 0:
        print(f"   ✅ FX Basis: {len(df_fx)} rows")
        print(f"      Latest: {df_fx.index[-1].strftime('%Y-%m-%d')}")
        print(f"      Value: {df_fx.iloc[-1, 0]:.2f} bp")
    else:
        print("   ⚠️  FX Basis: Failed (may use proxy in production)")

    print("   Testing VRP calculator...")
    df_vrp = compute_variance_risk_premium()
    if df_vrp is not None and len(df_vrp) > 0:
        print(f"   ✅ VRP: {len(df_vrp)} rows")
    else:
        print("   ❌ VRP: Failed")

    print("   Testing Convenience Yield...")
    df_cy = compute_convenience_yield()
    if df_cy is not None and len(df_cy) > 0:
        print(f"   ✅ Convenience Yield: {len(df_cy)} rows")
    else:
        print("   ❌ Convenience Yield: Failed")

except Exception as e:
    print(f"   ❌ International scrapers failed: {e}")

# Test 4: Test Master Scraper (FULL SYSTEM)
print("\n[4/5] Testing Master Scraper (FULL DATA FETCH)...")
print("   This may take 30-60 seconds on first run...")
print("   (Subsequent runs will be ~2-5s with cache)")

try:
    from macro_plumbing.data.master_scraper import fetch_with_summary

    # Fetch all data
    data, summary = fetch_with_summary(use_cache=True)

    print("\n   📊 DATA SUMMARY:")
    print("   " + "="*76)

    # Print summary
    for _, row in summary.iterrows():
        status_icon = row['Status']
        source = row['Source']
        rows = row['Rows']
        missing = row['Missing_Pct']

        print(f"   {status_icon} {source:25s} {rows:6d} rows  {missing:5.1f}% missing")

    print("   " + "="*76)

    # Count successes
    success_count = (summary['Status'] == '✅ OK').sum()
    total_count = len(summary)

    print(f"\n   ✅ Success Rate: {success_count}/{total_count} sources ({success_count/total_count*100:.1f}%)")

except Exception as e:
    print(f"   ❌ Master scraper failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test Quick Fetch (Single Function)
print("\n[5/5] Testing Quick Fetch (Production API)...")
try:
    from macro_plumbing.data.master_scraper import quick_fetch

    # This is what you'd use in production
    df = quick_fetch(use_cache=True, parallel=True)

    if df is not None and len(df) > 0:
        print(f"   ✅ Quick Fetch SUCCESS!")
        print(f"      Shape: {len(df)} rows × {len(df.columns)} columns")
        print(f"      Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

        # Show new Phase 2 columns
        phase2_cols = [
            'dealer_leverage', 'eur_usd_3m_basis', 'mmf_net_flows',
            'vrp', 'convenience_yield', 'sofr_p75', 'effr_p75'
        ]

        available_phase2 = [col for col in phase2_cols if col in df.columns]

        if available_phase2:
            print(f"\n      🎉 NEW PHASE 2 DATA AVAILABLE:")
            for col in available_phase2:
                latest_val = df[col].dropna().iloc[-1] if col in df.columns and not df[col].dropna().empty else None
                if latest_val is not None:
                    print(f"         - {col:25s} = {latest_val:.2f}")
        else:
            print("\n      ⚠️  No Phase 2 columns detected (check integration)")

    else:
        print(f"   ❌ Quick Fetch returned empty data")

except Exception as e:
    print(f"   ❌ Quick fetch failed: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("\n" + "="*80)
print("🎯 TEST COMPLETE")
print("="*80)
print("""
NEXT STEPS:

1. ✅ If tests passed → Proceed to integrate into app.py
2. ⚠️  If tests show warnings → Check cache and retry
3. ❌ If tests failed → Check error messages above

To integrate into your app:
    from macro_plumbing.data.master_scraper import quick_fetch

    df = quick_fetch(fred_api_key='YOUR_KEY', use_cache=True, parallel=True)

To run the Streamlit app:
    streamlit run macro_plumbing/app/app.py
""")
