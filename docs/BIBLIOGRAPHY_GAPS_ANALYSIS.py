"""
BIBLIOGRAPHIC REVIEW & GAP ANALYSIS
Liquidity Stress Detection System vs. State-of-the-Art Literature
==================================================================

Author: Claude Code Review
Date: 2025-11-07
System: MacroArimax Liquidity Stress Detection System

This document provides a comprehensive literature review organized by topic,
identifies what the current system implements, and highlights critical gaps
that should be addressed for world-class performance.

"""

# ============================================================================
# 1. FINANCIAL PLUMBING & MONEY MARKETS ARCHITECTURE
# ============================================================================

LITERATURE_PLUMBING = {
    "FOUNDATIONAL_THEORY": {
        "Pozsar_2014": {
            "title": "Shadow Banking and the Four Pillars of Modern Finance",
            "source": "FRBNY Staff Report 458",
            "key_concepts": [
                "Money view vs Credit view",
                "Institutional cash pools",
                "Dealer balance sheet constraints",
                "Reverse repo as shadow banking infrastructure"
            ],
            "in_system": "✅ Partial - Models RRP, dealers, MMFs as nodes",
            "missing": [
                "❌ Institutional cash pool size estimates",
                "❌ Dealer leverage constraints (SLR)",
                "❌ Securities lending vs repo distinction"
            ]
        },

        "Singh_2020": {
            "title": "Global Liquidity: The Elusive 'It'",
            "source": "IMF Working Paper",
            "key_concepts": [
                "Collateral velocity (pledging chains)",
                "Balance sheet capacity of dealers",
                "US dollar shortage indicators (FX basis)",
                "HQLA vs HQLA'less assets"
            ],
            "in_system": "⚠️ Missing collateral velocity",
            "missing": [
                "❌ Collateral reuse rates (from Flow of Funds)",
                "❌ FX cross-currency basis (USD/EUR, USD/JPY)",
                "❌ Securities lending indicators",
                "❌ Repo fails tracking (DVP fails)"
            ]
        },

        "Copeland_Martin_2012": {
            "title": "The Evolution of Clearinghouse Practices in U.S. Repo Markets",
            "source": "FRBNY Staff Report 576",
            "key_concepts": [
                "Tri-party vs bilateral repo",
                "FICC GCF repo",
                "Collateral substitution",
                "Intraday credit risk"
            ],
            "in_system": "⚠️ Only tracks TGCR (aggregate)",
            "missing": [
                "❌ Tri-party repo volume (DTCC data)",
                "❌ GCF vs DVP split",
                "❌ Fails-to-deliver rates",
                "❌ Cleared vs uncleared repo distinction"
            ]
        },

        "Adrian_Shin_2010": {
            "title": "Liquidity and Leverage",
            "source": "JF 2010",
            "key_concepts": [
                "Financial intermediary leverage procyclicality",
                "VaR-based leverage targeting",
                "Balance sheet expansion/contraction",
                "Margin spirals"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Primary dealer leverage (from H.15 data)",
                "❌ Margin requirements (haircuts) time series",
                "❌ Mark-to-market losses triggering deleveraging",
                "❌ VaR shock scenarios"
            ]
        }
    },

    "RESERVE_SCARCITY": {
        "Afonso_Lagos_2015": {
            "title": "Trade Dynamics in the Market for Federal Funds",
            "source": "Econometrica 2015",
            "key_concepts": [
                "Reserve scarcity indicators",
                "Bargaining power shifts (banks vs non-banks)",
                "EFFR dispersion as stress signal",
                "Interbank network structure"
            ],
            "in_system": "⚠️ Tracks EFFR level, not dispersion",
            "missing": [
                "❌ EFFR volume-weighted dispersion (75th-25th pct)",
                "❌ Distribution moments (skewness, kurtosis)",
                "❌ Trade-level data (if accessible via SOMA)",
                "❌ Interbank network topology"
            ]
        },

        "Sims_Wu_2021": {
            "title": "Evaluating Central Bank Tool Kit",
            "source": "JME 2021",
            "key_concepts": [
                "Interest on reserves (IOR) vs ON RRP",
                "Federal funds corridor management",
                "Reserve demand elasticity",
                "QT vs balance sheet normalization paths"
            ],
            "in_system": "⚠️ Tracks RRP, reserves separately",
            "missing": [
                "❌ IOR rate vs EFFR vs ON RRP spread dynamics",
                "❌ Reserve demand curve estimation",
                "❌ QT announcement effect decomposition",
                "❌ Forward guidance integration"
            ]
        }
    },

    "TGA_DYNAMICS": {
        "Correa_Waller_2021": {
            "title": "What Happens When the Treasury Refills Its Account?",
            "source": "FRBNY Liberty Street Economics",
            "key_concepts": [
                "TGA replenishment drains reserves",
                "Treasury bill issuance vs RRP substitute",
                "Debt ceiling effects on TGA",
                "Bank reserve preferences"
            ],
            "in_system": "✅ Tracks TGA levels and deltas",
            "missing": [
                "❌ Treasury auction calendar integration",
                "❌ T-bill yields vs ON RRP rate arbitrage",
                "❌ Debt ceiling countdown indicator",
                "❌ TAS (Treasury auction surprises)"
            ]
        }
    }
}

# ============================================================================
# 2. NETWORK MODELS & SYSTEMIC RISK
# ============================================================================

LITERATURE_NETWORKS = {
    "CONTAGION_MODELS": {
        "Acemoglu_etal_2015": {
            "title": "Systemic Risk and Stability in Financial Networks",
            "source": "AER 2015",
            "key_concepts": [
                "Phase transition in contagion",
                "Diversification vs concentration trade-off",
                "Shock amplification via network",
                "Default cascades"
            ],
            "in_system": "⚠️ Has random walk contagion",
            "missing": [
                "❌ Default threshold modeling",
                "❌ Capital buffer dynamics",
                "❌ Cascading failures simulation",
                "❌ Phase transition detection"
            ]
        },

        "Battiston_etal_2016": {
            "title": "DebtRank: Too Central to Fail?",
            "source": "Nature Scientific Reports",
            "key_concepts": [
                "DebtRank centrality measure",
                "Stress propagation via balance sheets",
                "Too-interconnected-to-fail",
                "Equity depletion paths"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ DebtRank algorithm",
                "❌ Balance sheet contagion (equity-debt links)",
                "❌ Loss amplification factor",
                "❌ Systemic impact measure"
            ]
        },

        "Glasserman_Young_2016": {
            "title": "Contagion in Financial Networks",
            "source": "JEL 2016 (Review)",
            "key_concepts": [
                "Threshold contagion models",
                "Network topology (core-periphery)",
                "Fire sale externalities",
                "Clearing equilibrium"
            ],
            "in_system": "⚠️ Basic SI model, no threshold",
            "missing": [
                "❌ Fire sale price impact modeling",
                "❌ Core-periphery decomposition",
                "❌ Clearing vector computation",
                "❌ Multiple equilibria detection"
            ]
        }
    },

    "NETWORK_TOPOLOGY": {
        "Craig_vonPeter_2014": {
            "title": "Interbank Tiering and Money Center Banks",
            "source": "JFI 2014",
            "key_concepts": [
                "Money center banks as hubs",
                "Tiered network structure",
                "Regional banks vs globals",
                "Liquidity hoarding at core"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Bank tiering classification",
                "❌ Core-periphery detection algorithms",
                "❌ Liquidity hoarding indicators",
                "❌ Hub stress spillover effects"
            ]
        }
    },

    "MULTIPLEX_NETWORKS": {
        "Poledna_etal_2015": {
            "title": "The Multi-Layer Network Nature of Systemic Risk",
            "source": "Nature Communications",
            "key_concepts": [
                "Multiple network layers (credit, equity, derivatives)",
                "Cross-layer contagion",
                "Interacting cascade mechanisms",
                "Multiplex centrality"
            ],
            "in_system": "❌ Single layer only",
            "missing": [
                "❌ Multi-layer graph structure",
                "❌ Derivatives exposures layer",
                "❌ Cross-layer amplification",
                "❌ Multiplex PageRank"
            ]
        }
    }
}

# ============================================================================
# 3. EARLY WARNING INDICATORS & CRISIS PREDICTION
# ============================================================================

LITERATURE_EARLY_WARNING = {
    "BANKING_CRISES": {
        "Aldasoro_Borio_2018": {
            "title": "Financial Cycles and the Real Economy",
            "source": "BIS Quarterly Review",
            "key_concepts": [
                "Credit-to-GDP gap (Basel III indicator)",
                "Property price misalignments",
                "DSR (debt service ratio)",
                "Financial cycle peaks 15-20 years"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Credit-to-GDP gap (BIS methodology)",
                "❌ DSR calculations",
                "❌ Real estate price gaps",
                "❌ Financial cycle dating"
            ]
        },

        "Kaminsky_Reinhart_1999": {
            "title": "The Twin Crises: Banking and Balance-of-Payments",
            "source": "AER 1999",
            "key_concepts": [
                "Noise-to-signal ratio for EWIs",
                "Real exchange rate overvaluation",
                "M2/reserves ratio",
                "Export growth collapse"
            ],
            "in_system": "❌ US-only, no EM indicators",
            "missing": [
                "❌ FX reserve adequacy",
                "❌ Real effective exchange rate",
                "❌ Current account imbalances",
                "❌ Noise-to-signal optimization"
            ]
        }
    },

    "LIQUIDITY_CRISES": {
        "Morris_Shin_2016": {
            "title": "Illiquidity Component of Credit Risk",
            "source": "Princeton Economics",
            "key_concepts": [
                "Market vs funding liquidity interaction",
                "CDS-bond basis as liquidity indicator",
                "Roll's measure of illiquidity",
                "Bid-ask spreads in UST"
            ],
            "in_system": "⚠️ Missing microstructure indicators",
            "missing": [
                "❌ UST bid-ask spreads (TRACE data)",
                "❌ Roll's measure (serial covariance)",
                "❌ CDS-bond basis",
                "❌ Trade size vs price impact"
            ]
        },

        "Nagel_2016": {
            "title": "The Liquidity Premium of Near-Money Assets",
            "source": "QJE 2016",
            "key_concepts": [
                "Convenience yield on Treasuries",
                "T-bill shortage premium",
                "Money-like assets hierarchy",
                "Flight to quality during stress"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Convenience yield estimates (T-bill - OIS)",
                "❌ Bill-bond yield spread",
                "❌ Float-adjusted Treasury supply",
                "❌ Safe asset scarcity measure"
            ]
        }
    },

    "MACHINE_LEARNING_EW": {
        "Beutel_etal_2019": {
            "title": "Machine Learning for Financial Stability",
            "source": "BIS FSI Insights",
            "key_concepts": [
                "Random forests for crisis prediction",
                "Feature importance from tree ensembles",
                "Class imbalance handling (SMOTE)",
                "Precision-recall trade-off"
            ],
            "in_system": "⚠️ Has IsolationForest, not RF",
            "missing": [
                "❌ Random Forest classifier",
                "❌ SMOTE for rare events",
                "❌ Precision-recall curves",
                "❌ Cost-sensitive loss functions"
            ]
        },

        "Aldasoro_etal_2022": {
            "title": "Machine Learning Models for Banking Crises",
            "source": "BIS Working Paper 1001",
            "key_concepts": [
                "Gradient boosting (XGBoost)",
                "SHAP values for explainability",
                "Time-series cross-validation",
                "Nowcasting vs forecasting distinction"
            ],
            "in_system": "⚠️ Has logistic nowcast",
            "missing": [
                "❌ Gradient boosting models (XGBoost, LightGBM)",
                "❌ SHAP value decomposition",
                "❌ Nested cross-validation",
                "❌ Forecast horizon optimization (1d vs 5d vs 20d)"
            ]
        }
    }
}

# ============================================================================
# 4. HIGH-FREQUENCY INDICATORS & MARKET MICROSTRUCTURE
# ============================================================================

LITERATURE_MICROSTRUCTURE = {
    "INTRADAY_LIQUIDITY": {
        "Fleming_etal_2020": {
            "title": "The March 2020 Treasury Market Dysfunction",
            "source": "FRBNY Staff Report 963",
            "key_concepts": [
                "Bid-ask spreads widening",
                "Depth deterioration (top-of-book)",
                "Quote volatility (cancellation rates)",
                "Dealer inventory constraints"
            ],
            "in_system": "❌ No intraday data",
            "missing": [
                "❌ Intraday bid-ask spreads (tick data)",
                "❌ Market depth at best quotes",
                "❌ Order flow imbalance",
                "❌ Trade-through rate (fragmentation)"
            ]
        },

        "HuGao_etal_2019": {
            "title": "Measuring Liquidity in UST Markets",
            "source": "Federal Reserve Board FEDS Notes",
            "key_concepts": [
                "Amihud illiquidity ratio",
                "Effective spread vs quoted spread",
                "Kyle's lambda (price impact)",
                "Time-to-execution"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Amihud ratio (daily |return|/volume)",
                "❌ Effective spread from transaction data",
                "❌ Kyle's lambda estimation",
                "❌ Volume time series from TRACE"
            ]
        }
    },

    "REPO_SPECIALNESS": {
        "Duffie_1996": {
            "title": "Special Repo Rates",
            "source": "JF 1996",
            "key_concepts": [
                "On-the-run vs off-the-run spread",
                "Specialness = GC rate - special rate",
                "Short squeeze detection",
                "Fails charges"
            ],
            "in_system": "⚠️ Has TGCR, but not specialness",
            "missing": [
                "❌ Specific security repo rates",
                "❌ On-the-run premiums",
                "❌ Fails-to-deliver statistics (FINRA)",
                "❌ Collateral substitution costs"
            ]
        }
    }
}

# ============================================================================
# 5. CENTRAL BANK LIQUIDITY INDICATORS (OPERATIONAL)
# ============================================================================

LITERATURE_CENTRAL_BANK = {
    "FED_OPERATIONS": {
        "Logan_2022": {
            "title": "Observations on Implementing Monetary Policy",
            "source": "Dallas Fed Speech",
            "key_concepts": [
                "Abundant reserves regime",
                "Standing repo facility (SRF) usage",
                "Balance sheet normalization speed",
                "SOMA portfolio composition"
            ],
            "in_system": "⚠️ Tracks reserves, RRP, not SRF",
            "missing": [
                "❌ SRF usage data (daily)",
                "❌ SOMA holdings by maturity bucket",
                "❌ QT pace announcements",
                "❌ Reserve demand estimates (LCLoR survey)"
            ]
        },

        "Duffie_Krishnamurthy_2016": {
            "title": "Passthrough Efficiency in the Fed's New Monetary Policy",
            "source": "Jackson Hole Symposium",
            "key_concepts": [
                "IOR vs ON RRP pass-through",
                "Arbitrage constraints (balance sheet costs)",
                "Fed Funds - IOR spread (IOER spread)",
                "Money market segmentation"
            ],
            "in_system": "⚠️ Missing IOR series",
            "missing": [
                "❌ IOR rate time series",
                "❌ EFFR - IOR spread",
                "❌ ON RRP - IOR spread",
                "❌ Arbitrage bounds estimation"
            ]
        }
    },

    "ECB_BOE_BOJ": {
        "Note": "System is US-only. For global coverage:",
        "missing": [
            "❌ ECB deposit facility usage (ESTR dynamics)",
            "❌ BOE money market fragmentation",
            "❌ BOJ YCC distortions",
            "❌ SNB sight deposits",
            "❌ PBOC RRR changes"
        ]
    }
}

# ============================================================================
# 6. DERIVATIVES & VOLATILITY MARKETS
# ============================================================================

LITERATURE_DERIVATIVES = {
    "VOLATILITY_REGIME": {
        "Bekaert_Hoerova_2014": {
            "title": "The VIX, the Variance Premium, and Stock Market Volatility",
            "source": "JEL 2014",
            "key_concepts": [
                "VIX as risk aversion proxy",
                "Variance risk premium (VRP)",
                "Volatility of volatility (VVIX)",
                "Term structure of VIX (VIX1M, VIX3M)"
            ],
            "in_system": "✅ Has VIX level",
            "missing": [
                "❌ VRP calculation (VIX² - realized variance)",
                "❌ VVIX (vol of vol)",
                "❌ VIX term structure (contango/backwardation)",
                "❌ Skew indicators (OTM put premium)"
            ]
        }
    },

    "SWAPTION_VOLATILITY": {
        "MOVE_Index": {
            "title": "Merrill Lynch Option Volatility Estimate",
            "source": "ICE Data Services",
            "key_concepts": [
                "Bond market implied volatility",
                "Swaption volatility surface",
                "Rate volatility stress",
                "Term premium uncertainty"
            ],
            "in_system": "⚠️ Optional (not core)",
            "missing": [
                "❌ MOVE index daily integration",
                "❌ Swaption skew (payer vs receiver)",
                "❌ Cap/floor volatility",
                "❌ Cross-asset vol correlation"
            ]
        }
    }
}

# ============================================================================
# 7. FUNDING MARKETS & FX
# ============================================================================

LITERATURE_FUNDING_FX = {
    "FX_BASIS": {
        "Du_etal_2018": {
            "title": "Deviations from Covered Interest Parity",
            "source": "JF 2018",
            "key_concepts": [
                "CIP deviations as USD shortage",
                "Cross-currency basis (EUR/USD, JPY/USD)",
                "Balance sheet constraints (G-SIB)",
                "Limits to arbitrage"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ EUR/USD 3M basis swap",
                "❌ JPY/USD 3M basis swap",
                "❌ GBP/USD, CHF/USD basis",
                "❌ Quarter-end spikes in basis"
            ]
        },

        "Borio_etal_2016": {
            "title": "The Failure of Covered Interest Parity",
            "source": "BIS Quarterly Review",
            "key_concepts": [
                "Post-2008 CIP breakdown",
                "Non-US banks' dollar funding needs",
                "Central bank swap lines usage",
                "Hedging costs for foreign investors"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ Central bank swap line usage (Fed, ECB)",
                "❌ Foreign official holdings of UST",
                "❌ Basis swap term structure",
                "❌ Dealer hedging costs"
            ]
        }
    },

    "COMMERCIAL_PAPER": {
        "Kacperczyk_Schnabl_2010": {
            "title": "When Safe Proved Risky: Money Market Fund Run",
            "source": "JF 2010",
            "key_concepts": [
                "MMF run dynamics (2008)",
                "CP spreads (AA vs A2P2)",
                "Redemption suspensions",
                "Implicit guarantees"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ CP spreads (FRED: CPF3M, DCPN3M)",
                "❌ MMF outflow data (ICI weekly)",
                "❌ Prime vs Govt MMF distinction",
                "❌ CP outstanding (FRED: COMPOUT)"
            ]
        }
    }
}

# ============================================================================
# 8. STRUCTURAL MODELS & MACRO-FINANCE
# ============================================================================

LITERATURE_MACRO_FINANCE = {
    "DSGE_FINANCIAL_FRICTIONS": {
        "Gertler_Karadi_2011": {
            "title": "A Model of Unconventional Monetary Policy",
            "source": "JME 2011",
            "key_concepts": [
                "Financial intermediary constraints",
                "QE transmission mechanism",
                "Credit spreads from balance sheet",
                "Capital quality shocks"
            ],
            "in_system": "❌ Reduced-form only",
            "missing": [
                "❌ DSGE-based nowcast",
                "❌ Intermediary leverage ratio",
                "❌ Credit spread decomposition (default vs liquidity)",
                "❌ Structural shocks identification"
            ]
        }
    },

    "TERM_STRUCTURE": {
        "Adrian_Crump_Moench_2013": {
            "title": "Pricing the Term Structure with Linear Regressions",
            "source": "JFE 2013",
            "key_concepts": [
                "ACM term premium model",
                "Unspanned macro factors",
                "Forward rate decomposition",
                "Real rate vs inflation expectations"
            ],
            "in_system": "❌ Not implemented",
            "missing": [
                "❌ ACM term premium (10Y)",
                "❌ Kim-Wright term premium (alternative)",
                "❌ Forward rate regressions",
                "❌ Breakeven inflation decomposition"
            ]
        }
    }
}

# ============================================================================
# 9. GAPS SUMMARY & PRIORITY RANKING
# ============================================================================

CRITICAL_GAPS = {
    "TIER_1_HIGH_PRIORITY": {
        "description": "Essential for world-class early warning system",
        "gaps": [
            {
                "gap": "FX Cross-Currency Basis (EUR/USD, JPY/USD)",
                "why_critical": "USD shortage is #1 stress signal in global markets",
                "data_source": "Bloomberg XCCY indices, or FED H.15",
                "implementation": "Add to series_map.yaml, new edge: FX_Funding → Banks",
                "papers": ["Du et al. 2018", "Borio et al. 2016"]
            },
            {
                "gap": "Primary Dealer Leverage Ratio",
                "why_critical": "Adrian-Shin channel for liquidity crises",
                "data_source": "Fed H.15 Primary Dealer Statistics (weekly)",
                "implementation": "New node attribute: dealer_leverage_z",
                "papers": ["Adrian-Shin 2010", "Fleming et al. 2020"]
            },
            {
                "gap": "UST Market Microstructure (Bid-Ask, Depth)",
                "why_critical": "Early signal of dealer stress before spreads widen",
                "data_source": "BestX, TRACE, or Fed Market Monitoring",
                "implementation": "New edge: UST_Market → Dealers with depth driver",
                "papers": ["Fleming et al. 2020", "Hu & Gao 2019"]
            },
            {
                "gap": "IOR (Interest on Reserves) Rate",
                "why_critical": "Core to Fed's monetary policy transmission",
                "data_source": "FRED: IORB (Interest on Reserve Balances)",
                "implementation": "Add to series_map, compute EFFR-IOR spread",
                "papers": ["Duffie-Krishnamurthy 2016", "Sims-Wu 2021"]
            },
            {
                "gap": "Commercial Paper Spreads (AA vs A2P2)",
                "why_critical": "Credit funding stress indicator for corporates",
                "data_source": "FRED: CPF3M (financial CP), DCPN3M (non-financial)",
                "implementation": "New edge: Corporates → MMFs with CP spread driver",
                "papers": ["Kacperczyk-Schnabl 2010"]
            },
            {
                "gap": "Standing Repo Facility (SRF) Usage",
                "why_critical": "Measures reserve scarcity at Fed backstop",
                "data_source": "FRED: WORAL (Weekly SRF Operations)",
                "implementation": "New node: SRF, edge: Banks → Fed (backstop usage)",
                "papers": ["Logan 2022"]
            },
            {
                "gap": "Tri-Party Repo Volume & Fails",
                "why_critical": "Measures collateral chain health",
                "data_source": "FRBNY Tri-Party Repo Statistics",
                "implementation": "New edge attribute: repo_fails_rate",
                "papers": ["Copeland-Martin 2012", "Duffie 1996"]
            },
            {
                "gap": "Convenience Yield on Treasuries",
                "why_critical": "Safe asset scarcity premium",
                "data_source": "T-bill yield - OIS spread",
                "implementation": "Derived feature: convenience_yield = TB3MS - OIS",
                "papers": ["Nagel 2016"]
            },
            {
                "gap": "Gradient Boosting Models (XGBoost)",
                "why_critical": "State-of-art ML for crisis prediction",
                "data_source": "Internal - retrain on existing features",
                "implementation": "New model: XGBClassifier in models/xgboost_crisis.py",
                "papers": ["Aldasoro et al. 2022", "Beutel et al. 2019"]
            },
            {
                "gap": "SHAP Values for Explainability",
                "why_critical": "Regulatory requirement, stakeholder trust",
                "data_source": "shap library on fitted models",
                "implementation": "New module: metrics/shap_explainer.py",
                "papers": ["Lundberg-Lee 2017", "Aldasoro et al. 2022"]
            }
        ]
    },

    "TIER_2_IMPORTANT": {
        "description": "Significant improvements, not blocking",
        "gaps": [
            {
                "gap": "DebtRank Centrality",
                "why_critical": "Better systemic importance than PageRank",
                "implementation": "graph_analysis.py: add compute_debtrank()",
                "papers": ["Battiston et al. 2016"]
            },
            {
                "gap": "Fire Sale Spillovers",
                "why_critical": "Amplification mechanism in crisis",
                "implementation": "graph_contagion.py: add fire_sale_model()",
                "papers": ["Glasserman-Young 2016"]
            },
            {
                "gap": "Variance Risk Premium (VRP)",
                "why_critical": "Market-implied risk aversion",
                "implementation": "Derived feature: VRP = VIX² - realized_variance",
                "papers": ["Bekaert-Hoerova 2014"]
            },
            {
                "gap": "MOVE Index Integration",
                "why_critical": "Bond market stress analog to VIX",
                "implementation": "Add to series_map.yaml: MOVE",
                "papers": ["Fleming et al. 2020"]
            },
            {
                "gap": "Credit-to-GDP Gap",
                "why_critical": "Basel III early warning indicator",
                "implementation": "New feature: compute_credit_gap() using FRED: TOTCI",
                "papers": ["Aldasoro-Borio 2018"]
            },
            {
                "gap": "ACM Term Premium",
                "why_critical": "Decompose yield into real rate + premium",
                "implementation": "Add ACM term premium from FRED (if available)",
                "papers": ["Adrian-Crump-Moench 2013"]
            },
            {
                "gap": "Repo Fails-to-Deliver Rate",
                "why_critical": "Collateral shortage signal",
                "implementation": "New series from FINRA/DTCC if available",
                "papers": ["Copeland-Martin 2012"]
            },
            {
                "gap": "MMF Flow Data",
                "why_critical": "Outflow spike = funding stress",
                "implementation": "ICI weekly data (requires scraping)",
                "papers": ["Kacperczyk-Schnabl 2010"]
            },
            {
                "gap": "Precision-Recall Curves",
                "why_critical": "Better metric for rare events than AUROC",
                "implementation": "metrics.py: add compute_pr_curve()",
                "papers": ["Beutel et al. 2019"]
            },
            {
                "gap": "Cost-Sensitive Loss Function",
                "why_critical": "Weight false negatives > false positives",
                "implementation": "Update fusion.py with custom loss",
                "papers": ["Aldasoro et al. 2022"]
            }
        ]
    },

    "TIER_3_NICE_TO_HAVE": {
        "description": "Academic interest, lower operational priority",
        "gaps": [
            {
                "gap": "Multi-Layer Networks",
                "why_critical": "Captures credit + derivatives + equity links",
                "implementation": "Requires major refactor to multiplex graphs",
                "papers": ["Poledna et al. 2015"]
            },
            {
                "gap": "International Coverage (ECB, BOE, BOJ)",
                "why_critical": "Global liquidity stress can originate abroad",
                "implementation": "New data sources for EUR, GBP, JPY markets",
                "papers": ["Singh 2020 (IMF)"]
            },
            {
                "gap": "Intraday Tick Data",
                "why_critical": "Flash crashes, high-frequency stress",
                "implementation": "Requires tick database, order book",
                "papers": ["Fleming et al. 2020"]
            },
            {
                "gap": "DSGE Nowcast",
                "why_critical": "Structural interpretation of shocks",
                "implementation": "Integrate with IRIS or Dynare",
                "papers": ["Gertler-Karadi 2011"]
            },
            {
                "gap": "Collateral Velocity (Singh)",
                "why_critical": "Pledgeability of collateral in chains",
                "implementation": "Flow of Funds Z.1 table parsing",
                "papers": ["Singh 2020"]
            }
        ]
    }
}

# ============================================================================
# 10. RECOMMENDED READING LIST (ANNOTATED)
# ============================================================================

ESSENTIAL_READING = {
    "BOOKS": [
        {
            "author": "Mehrling, Perry",
            "title": "The New Lombard Street: How the Fed Became the Dealer of Last Resort",
            "year": 2011,
            "why_read": "Foundational for understanding Fed's role in shadow banking",
            "key_chapters": "Ch 3 (Dealer of Last Resort), Ch 5 (Money Market)"
        },
        {
            "author": "Pozsar, Zoltan et al.",
            "title": "Shadow Banking (FRBNY Staff Report 458)",
            "year": 2014,
            "why_read": "THE reference for money view vs credit view",
            "key_sections": "Institutional cash pools, Reverse repo plumbing"
        },
        {
            "author": "Singh, Manmohan",
            "title": "Collateral and Financial Plumbing (Risk Books)",
            "year": 2020,
            "why_read": "Deep dive into collateral chains, velocity, pledgeability"
        }
    ],

    "PAPERS_THEORY": [
        {
            "citation": "Adrian, T., & Shin, H. S. (2010). Liquidity and leverage. JF.",
            "why_read": "VaR-based leverage procyclicality mechanism",
            "core_equation": "Leverage = VaR target / Market volatility"
        },
        {
            "citation": "Acemoglu, D., et al. (2015). Systemic risk and stability in financial networks. AER.",
            "why_read": "Phase transitions in contagion, diversification paradox",
            "core_result": "Dense networks more robust to small shocks, fragile to large"
        },
        {
            "citation": "Morris, S., & Shin, H. S. (2016). Illiquidity component of credit risk.",
            "why_read": "Feedback loop between market and funding liquidity",
            "core_mechanism": "Funding dry-up → Asset sales → Price fall → More dry-up"
        }
    ],

    "PAPERS_EMPIRICAL": [
        {
            "citation": "Du, W., et al. (2018). Deviations from covered interest parity. JF.",
            "why_read": "CIP breakdown = USD shortage, not arbitrage opportunity",
            "data": "FX basis swaps, LIBOR-OIS"
        },
        {
            "citation": "Fleming, M., et al. (2020). March 2020 Treasury market dysfunction. FRBNY SR 963.",
            "why_read": "Real-time anatomy of UST market stress",
            "data": "Bid-ask spreads, depth, fails, dealer inventories"
        },
        {
            "citation": "Afonso, G., & Lagos, R. (2015). Trade dynamics in fed funds market. Econometrica.",
            "why_read": "Microstructure of reserve market with bargaining",
            "data": "EFFR dispersion, trade-level data"
        }
    ],

    "PAPERS_ML": [
        {
            "citation": "Aldasoro, I., et al. (2022). Machine learning for banking crises. BIS WP 1001.",
            "why_read": "State-of-art ML methods for crisis forecasting",
            "methods": "XGBoost, SHAP, time-series CV"
        },
        {
            "citation": "Beutel, J., et al. (2019). Machine learning for financial stability. BIS FSI Insights.",
            "why_read": "Practical guide to RF, class imbalance, precision-recall",
            "implementation": "Python code examples included"
        }
    ],

    "CENTRAL_BANK_REPORTS": [
        {
            "source": "BIS Quarterly Review",
            "frequency": "Quarterly",
            "why_read": "Cutting-edge research on money markets, FX, plumbing",
            "url": "https://www.bis.org/publ/qtrpdf/r_qt.htm"
        },
        {
            "source": "FRBNY Liberty Street Economics Blog",
            "frequency": "Weekly",
            "why_read": "Fed staff research on repo, RRP, TGA dynamics",
            "url": "https://libertystreeteconomics.newyorkfed.org/"
        },
        {
            "source": "Fed Financial Stability Report",
            "frequency": "Semi-annual",
            "why_read": "Official Fed view on systemic vulnerabilities",
            "sections": "Money market vulnerabilities, Leverage indicators"
        }
    ]
}

# ============================================================================
# 11. IMPLEMENTATION ROADMAP (PRIORITY ORDER)
# ============================================================================

IMPLEMENTATION_ROADMAP = {
    "PHASE_1_QUICK_WINS": {
        "duration": "1-2 weeks",
        "tasks": [
            "✅ Add IOR rate to series_map.yaml (FRED: IORB)",
            "✅ Compute EFFR-IOR spread, ON RRP-IOR spread",
            "✅ Add MOVE index (bond vol) to stress indicators",
            "✅ Add Commercial Paper spreads (CPF3M, DCPN3M)",
            "✅ Implement SHAP explainability for existing models",
            "✅ Add Precision-Recall curves to metrics",
            "✅ Compute Convenience Yield = TB3MS - SOFR",
            "✅ Add SRF usage data (WORAL)"
        ]
    },

    "PHASE_2_CRITICAL_FEATURES": {
        "duration": "3-4 weeks",
        "tasks": [
            "🔄 Integrate FX cross-currency basis (EUR/USD, JPY/USD)",
            "🔄 Primary dealer leverage from H.15 weekly data",
            "🔄 UST bid-ask spreads (manual scraping or vendor data)",
            "🔄 Tri-party repo volume from FRBNY data releases",
            "🔄 Add XGBoost model to ensemble",
            "🔄 Implement DebtRank centrality",
            "🔄 Add VRP (Variance Risk Premium) calculation",
            "🔄 MMF flow data from ICI (weekly scraping)"
        ]
    },

    "PHASE_3_ADVANCED_MODELS": {
        "duration": "6-8 weeks",
        "tasks": [
            "⏳ Fire sale spillover model in contagion",
            "⏳ Credit-to-GDP gap feature",
            "⏳ Repo fails-to-deliver rate tracking",
            "⏳ Cost-sensitive loss function in fusion",
            "⏳ Nested walk-forward cross-validation",
            "⏳ Multi-horizon forecasting (1d, 5d, 20d)",
            "⏳ ACM term premium integration",
            "⏳ EFFR dispersion (75th-25th pct) from Fedwire"
        ]
    },

    "PHASE_4_RESEARCH": {
        "duration": "3-6 months",
        "tasks": [
            "🔬 Multi-layer network implementation",
            "🔬 International markets (ECB, BOE, BOJ)",
            "🔬 Intraday tick data integration",
            "🔬 DSGE-based nowcast component",
            "🔬 Collateral velocity from Flow of Funds",
            "🔬 Structural break endogeneity test"
        ]
    }
}

# ============================================================================
# 12. DATA SOURCES EXPANSION
# ============================================================================

NEW_DATA_SOURCES = {
    "FRED_ADDITIONS": {
        "IOR_POLICY": ["IORB", "IOER"],  # Interest on Reserve Balances
        "REPO_FACILITIES": ["WORAL", "WOFAR"],  # SRF, FIMA repo
        "COMMERCIAL_PAPER": ["CPF3M", "DCPN3M", "COMPOUT"],
        "DEALER_STATS": ["DPSACBW027SBOG"],  # Primary dealer balance sheet
        "MMF_ASSETS": ["MMMFFAQ027S"],  # Money market fund assets
        "TERM_PREMIUM": ["ACMTP10", "THREEFYTP10"],  # ACM, Kim-Wright
        "BREAKEVEN_INFL": ["T5YIE", "T10YIE"],  # 5Y, 10Y breakevens
    },

    "FRBNY_DATA": {
        "REPO_REFERENCE_RATES": {
            "url": "https://www.newyorkfed.org/markets/reference-rates",
            "series": ["SOFR", "BGCR", "TGCR", "EFFR"],
            "frequency": "Daily",
            "includes": "Volume, dispersion (25th, 75th, 99th pct)"
        },
        "PRIMARY_DEALER_STATS": {
            "url": "https://www.newyorkfed.org/markets/primarydealer_survey_questions",
            "frequency": "Weekly",
            "data": "Dealer positions, financing, fails"
        },
        "TRI_PARTY_REPO": {
            "url": "https://www.newyorkfed.org/data-and-statistics/data-visualization/tri-party-repo",
            "frequency": "Daily",
            "data": "Volume, collateral type, counterparty"
        },
        "SOMA_HOLDINGS": {
            "url": "https://www.newyorkfed.org/markets/soma-holdings",
            "frequency": "Weekly",
            "data": "UST, MBS, agency debt by maturity"
        }
    },

    "BIS_DATA": {
        "CREDIT_GAPS": {
            "url": "https://www.bis.org/statistics/c_gaps.htm",
            "series": "Credit-to-GDP gaps, DSR",
            "countries": "US, Euro area, UK, Japan, China"
        },
        "EWI_DATABASE": {
            "url": "https://www.bis.org/publ/work963.htm",
            "description": "Early Warning Indicators Database",
            "coverage": "1970-present, 43 countries"
        }
    },

    "VENDOR_DATA": {
        "BLOOMBERG": {
            "FX_BASIS": "XCCY indices (EUR/USD 3M basis swap)",
            "UST_LIQUIDITY": "Bid-ask spreads, depth from ALLQ function",
            "SWAPTION_VOL": "OVSW function for vol surface"
        },
        "ICE_DATA": {
            "HY_OAS": "Already have (free FRED)",
            "MOVE_INDEX": "Bond option vol (MOVE ticker)"
        },
        "DTCC": {
            "REPO_DATA": "General Collateral Finance Repo data",
            "FAILS_DATA": "Fails-to-deliver by security"
        }
    }
}

# ============================================================================
# END OF REVIEW
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print("""
    SYSTEM STATUS: ⭐⭐⭐⭐ (4/5) - Strong foundation, missing critical plumbing data

    STRENGTHS:
    ✅ Comprehensive ensemble approach (7 models)
    ✅ Network-based financial system mapping (9 nodes, 16 edges)
    ✅ Walk-forward validation with proper metrics
    ✅ Operational playbooks with beta overlay
    ✅ Modern ML (IsolationForest, DFM+Kalman, HMM)

    CRITICAL GAPS (Top 10):
    1. ❌ FX Cross-Currency Basis (USD shortage = #1 global stress signal)
    2. ❌ Primary Dealer Leverage Ratio (Adrian-Shin amplification)
    3. ❌ UST Market Microstructure (bid-ask, depth)
    4. ❌ IOR Rate & Spreads (EFFR-IOR, RRP-IOR)
    5. ❌ Commercial Paper Spreads (funding market stress)
    6. ❌ Standing Repo Facility Usage (reserve scarcity backstop)
    7. ❌ Tri-Party Repo Volume & Fails (collateral chain health)
    8. ❌ Convenience Yield (safe asset premium)
    9. ❌ XGBoost / Gradient Boosting (SOTA ML)
    10. ❌ SHAP Explainability (regulatory/trust requirement)

    PRIORITY: Implement Phase 1 (Quick Wins) and Phase 2 (Critical Features)
    TIMEFRAME: 5-6 weeks to reach ⭐⭐⭐⭐⭐ (world-class)

    KEY PAPERS TO READ IMMEDIATELY:
    - Du et al. (2018) - FX basis deviations [JF]
    - Fleming et al. (2020) - March 2020 UST dysfunction [FRBNY SR 963]
    - Aldasoro et al. (2022) - ML for banking crises [BIS WP 1001]
    - Adrian-Shin (2010) - Liquidity and leverage [JF]
    - Pozsar (2014) - Shadow banking [FRBNY SR 458]
    """)
    print("="*80)
