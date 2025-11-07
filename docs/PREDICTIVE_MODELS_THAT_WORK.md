# Predictive Models That Actually Work
## What You CAN Predict vs. What You CAN'T

---

## The Honest Truth About Prediction

### ❌ What You CANNOT Predict (Don't Even Try):

1. **Exact prices**: "S&P 500 will be 4,237.82 next Tuesday" - IMPOSSIBLE
2. **Exact timing**: "Market will crash on March 15th" - BULLSHIT
3. **Turning points**: "This is THE top" - Nobody knows
4. **Volatility levels**: "VIX will be exactly 27.4 tomorrow" - Waste of time

### ✅ What You CAN Predict (With Measurable Accuracy):

1. **Probability of crisis**: "65% chance of stress event in next 5 days" - TESTABLE
2. **Direction of spreads**: "CP spread will widen (not tighten)" - MEASURABLE
3. **Regime transitions**: "70% probability we move to Stress regime" - TRACKABLE
4. **Severity classification**: "Next stress event will be Moderate (not Severe)" - VALIDATABLE
5. **Relative risk**: "Risk higher than 85% of historical days" - COMPARABLE

---

## 1. Random Forest Crisis Classifier

### The Problem: Predict if crisis will occur in next N days

**Why it works:**
- Tree-based models handle non-linearities
- Feature importance shows what matters
- Ensemble reduces overfitting
- Works with missing data
- Fast training and inference

### Implementation:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

class CrisisPredictor:
    """
    Predicts probability of liquidity crisis in next N days.

    NOT predicting prices - predicting crisis probability.
    """

    def __init__(self, horizon=5):
        """
        Parameters:
        - horizon: Days ahead to predict (default 5)
        """
        self.horizon = horizon
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',  # Handle imbalanced data
            random_state=42
        )

    def create_labels(self, df):
        """
        Create binary labels: 1 = crisis in next N days, 0 = no crisis

        Crisis definition:
        - VIX > 35 OR
        - CP spread > 150bp OR
        - HY OAS > 700bp OR
        - Discount Window > $10B
        """
        crisis_conditions = (
            (df['VIX'] > 35) |
            (df['cp_tbill_spread'] > 150) |
            (df['HY_OAS'] > 700) |
            (df['DISCOUNT_WINDOW'] > 10000)
        )

        # Look ahead N days
        df['crisis_ahead'] = crisis_conditions.shift(-self.horizon).fillna(0).astype(int)

        return df

    def prepare_features(self, df):
        """
        Select predictive features.

        Focus on:
        - Spreads (early warning)
        - Momentum (rate of change)
        - Cross-asset signals
        - Term structure
        """
        features = [
            # Spreads (LEADING INDICATORS)
            'cp_tbill_spread', 'bbb_aaa_spread', 'credit_cascade',
            'sofr_effr_spread', 'bgcr_sofr_spread', 'euribor_ois_proxy',

            # Credit stress
            'HY_OAS', 'CORP_BBB_OAS', 'bb_bbb_spread', 'ccc_bb_spread',

            # Volatility
            'VIX', 'vix_alarm',

            # Money markets
            'CP_FINANCIAL_3M', 'CP_NONFINANCIAL_3M',

            # Fed facilities (crisis indicator)
            'DISCOUNT_WINDOW', 'discount_window_alarm',

            # Repo markets
            'SOFR', 'BGCR', 'sofr_term_premium',

            # Term structure
            'T10Y2Y', 'term_spread_5y10y', 'term_spread_2y5y',

            # Liquidity
            'delta_rrp', 'delta_tga', 'delta_reserves',

            # Real economy (recession signals)
            'jobless_claims_zscore', 'continued_claims_zscore',
            'delinquency_index', 'labor_slack',

            # International (global stress)
            'us_japan_spread', 'DOLLAR_INDEX',

            # Momentum features (rate of change)
            'delta_bank_credit', 'delta_ci_loans',
            'housing_momentum', 'retail_sales_momentum'
        ]

        # Add lagged features (what happened yesterday matters)
        for col in ['VIX', 'cp_tbill_spread', 'HY_OAS']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
            features.extend([f'{col}_lag1', f'{col}_lag3'])

        # Add rolling volatility (recent instability predicts crisis)
        for col in ['cp_tbill_spread', 'VIX']:
            df[f'{col}_volatility'] = df[col].rolling(10).std()
            features.append(f'{col}_volatility')

        return features

    def train(self, df):
        """Train model on historical data"""

        # Create labels
        df = self.create_labels(df)

        # Prepare features
        features = self.prepare_features(df)

        # Remove rows with NaN (from lags/rolling)
        X = df[features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        # Train
        self.model.fit(X, y)
        self.features = features

        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Top 10 Most Important Features:")
        print(importance.head(10))

        return self

    def predict_proba(self, df):
        """Predict crisis probability"""

        X = df[self.features]
        proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (crisis)

        return proba

    def backtest(self, df):
        """
        Time-series cross-validation.

        CRITICAL: Use time-series split (no future data leakage)
        """

        # Create labels and features
        df = self.create_labels(df)
        features = self.prepare_features(df)

        X = df[features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        # Time series split (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)

        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Train on past, test on future
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, y_proba)

            results.append({
                'fold': fold,
                'auc': auc,
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1]
            })

            print(f"\nFold {fold}:")
            print(f"  Train: {X_train.index[0]} to {X_train.index[-1]}")
            print(f"  Test:  {X_test.index[0]} to {X_test.index[-1]}")
            print(f"  AUC: {auc:.3f}")
            print(classification_report(y_test, y_pred, target_names=['No Crisis', 'Crisis']))

        # Average performance
        avg_auc = np.mean([r['auc'] for r in results])
        print(f"\n{'='*60}")
        print(f"AVERAGE AUC ACROSS FOLDS: {avg_auc:.3f}")
        print(f"{'='*60}")

        return results


# Usage Example
if __name__ == "__main__":
    # Load data
    df = pd.read_pickle('data.pkl')  # Your FRED data with all features

    # Train model
    predictor = CrisisPredictor(horizon=5)
    predictor.train(df.loc[:'2022-12-31'])  # Train on data up to 2022

    # Predict on recent data
    recent_proba = predictor.predict_proba(df.loc['2023-01-01':])

    print(f"\nCurrent Crisis Probability (next 5 days): {recent_proba[-1]:.1%}")

    # Backtest
    results = predictor.backtest(df)
```

### Expected Performance:

**Good model:**
- AUC: 0.75-0.85 (excellent discrimination)
- Precision (crisis): 40-60% (when it says crisis, it's right 40-60% of time)
- Recall (crisis): 70-90% (catches 70-90% of crises)
- Lead time: 3-7 days before peak stress

**Why precision is "low":**
Because crises are rare! A 50% precision on 5% base rate = 10x better than random.

### Feature Importance Insights:

Typically most important:
1. `cp_tbill_spread` (commercial paper stress)
2. `HY_OAS` (credit market stress)
3. `VIX` (volatility)
4. `discount_window_alarm` (Fed emergency lending)
5. `credit_cascade` (CCC-AAA spread)

---

## 2. Gradient Boosting (XGBoost/LightGBM)

### Why often BETTER than Random Forest:

- Handles imbalanced data better
- Sequential learning (corrects previous errors)
- Faster training
- Better calibrated probabilities

### Implementation:

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

class XGBoostCrisisPredictor:
    """
    XGBoost for crisis prediction.

    Often outperforms Random Forest for imbalanced classification.
    """

    def __init__(self, horizon=5):
        self.horizon = horizon
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            scale_pos_weight=10,  # Handle imbalanced (10:1 ratio)
            random_state=42,
            eval_metric='auc'
        )

    def train(self, df, eval_set=None):
        """Train with early stopping"""

        # Same feature prep as Random Forest
        df = self.create_labels(df)
        features = self.prepare_features(df)

        X = df[features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        # Early stopping (prevent overfitting)
        if eval_set:
            self.model.fit(
                X, y,
                eval_set=[(eval_set[0], eval_set[1])],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        self.features = features

        # Feature importance (gain-based)
        importance = pd.DataFrame({
            'feature': features,
            'gain': self.model.feature_importances_
        }).sort_values('gain', ascending=False)

        print("Top 10 Most Important Features (by gain):")
        print(importance.head(10))

        return self

    def predict_proba(self, df):
        """Predict crisis probability"""
        X = df[self.features]
        return self.model.predict_proba(X)[:, 1]

    def calibration_analysis(self, df):
        """
        Check if predicted probabilities are well-calibrated.

        If model says 30%, it should happen ~30% of time.
        """
        from sklearn.calibration import calibration_curve

        y_true = df['crisis_ahead'].values
        y_proba = self.predict_proba(df)

        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

        # Brier score (lower = better calibrated)
        brier = brier_score_loss(y_true, y_proba)

        print(f"Brier Score: {brier:.4f} (lower = better)")
        print("\nCalibration:")
        for pt, pp in zip(prob_true, prob_pred):
            print(f"  Predicted {pp:.1%} → Actual {pt:.1%}")


# Example: Compare RF vs XGBoost
rf_predictor = CrisisPredictor(horizon=5).train(df_train)
xgb_predictor = XGBoostCrisisPredictor(horizon=5).train(df_train)

rf_proba = rf_predictor.predict_proba(df_test)
xgb_proba = xgb_predictor.predict_proba(df_test)

print(f"Random Forest AUC: {roc_auc_score(y_test, rf_proba):.3f}")
print(f"XGBoost AUC:       {roc_auc_score(y_test, xgb_proba):.3f}")
```

---

## 3. LSTM for Volatility Regime Prediction

### The Problem: Predict next volatility regime (Low/Medium/High)

**Why LSTM:**
- Captures temporal dependencies
- Remembers patterns over weeks
- Good for regime transitions

### Implementation:

```python
import torch
import torch.nn as nn
import numpy as np

class VolatilityRegimeLSTM(nn.Module):
    """
    Predict volatility regime (Low/Medium/High) using LSTM.

    Uses 20-day sequences to predict next 5-day regime.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Use last timestep
        last_output = lstm_out[:, -1, :]

        # Classify
        logits = self.fc(last_output)
        return logits


def create_sequences(df, features, target, seq_length=20, horizon=5):
    """
    Create sequences for LSTM training.

    X: Last 20 days of features
    y: Volatility regime in next 5 days
    """

    X, y = [], []

    for i in range(seq_length, len(df) - horizon):
        # Sequence: days i-20 to i
        X.append(df[features].iloc[i-seq_length:i].values)

        # Target: regime in next 5 days (max VIX)
        future_vix = df['VIX'].iloc[i:i+horizon].max()

        if future_vix < 20:
            regime = 0  # Low
        elif future_vix < 30:
            regime = 1  # Medium
        else:
            regime = 2  # High

        y.append(regime)

    return np.array(X), np.array(y)


# Training
features = ['VIX', 'HY_OAS', 'cp_tbill_spread', 'bbb_aaa_spread',
            'delta_rrp', 'T10Y2Y', 'SOFR', 'credit_cascade']

X, y = create_sequences(df_train, features, 'VIX', seq_length=20, horizon=5)

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Model
model = VolatilityRegimeLSTM(input_dim=len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    logits = model(X_tensor)
    loss = criterion(logits, y_tensor)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = (logits.argmax(dim=1) == y_tensor).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.3f}")

# Predict
model.eval()
with torch.no_grad():
    current_seq = df[features].iloc[-20:].values
    current_tensor = torch.FloatTensor(current_seq).unsqueeze(0)

    probs = torch.softmax(model(current_tensor), dim=1)
    predicted_regime = probs.argmax().item()

    regime_names = ['Low Vol', 'Medium Vol', 'High Vol']
    print(f"\nPredicted Regime (next 5 days): {regime_names[predicted_regime]}")
    print(f"Probabilities: Low={probs[0,0]:.1%}, Med={probs[0,1]:.1%}, High={probs[0,2]:.1%}")
```

### Expected Performance:

- Accuracy: 60-70% (vs 33% random)
- Best at predicting regime persistence (Low stays Low)
- Struggles with sudden shocks (Black Swans)

---

## 4. Spread Direction Prediction (Practical!)

### The Problem: Will CP spread widen or tighten tomorrow?

**This is actually useful for trading/positioning:**

```python
from sklearn.linear_model import LogisticRegression

class SpreadDirectionPredictor:
    """
    Predict if spread will widen (1) or tighten (0) next day.

    MUCH easier than predicting exact levels.
    Useful for positioning: widen = risk off, tighten = risk on
    """

    def prepare_data(self, df):
        """
        Create binary target: 1 = spread widened, 0 = tightened
        """

        # Tomorrow's direction
        df['spread_direction'] = (df['cp_tbill_spread'].diff().shift(-1) > 0).astype(int)

        # Features (what predicts spread widening?)
        features = [
            'VIX',                    # Higher VIX → spread widens
            'delta_rrp',              # RRP drain → spread widens
            'HY_OAS',                 # Credit stress → spread widens
            'T10Y2Y',                 # Inversion → spread widens
            'delta_reserves',         # Liquidity drain → spread widens
            'jobless_claims_zscore',  # Labor market stress → spread widens
        ]

        # Add momentum (is spread already widening?)
        df['spread_momentum'] = df['cp_tbill_spread'].diff()
        features.append('spread_momentum')

        # Add VIX change (rising volatility → spread widens)
        df['vix_change'] = df['VIX'].diff()
        features.append('vix_change')

        return features

    def train(self, df):
        """Simple logistic regression (interpretable coefficients)"""

        features = self.prepare_data(df)

        X = df[features].dropna()
        y = df.loc[X.index, 'spread_direction']

        # Logistic regression
        self.model = LogisticRegression(penalty='l2', C=1.0)
        self.model.fit(X, y)
        self.features = features

        # Coefficients (which features predict widening?)
        coefs = pd.DataFrame({
            'feature': features,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', ascending=False)

        print("Feature Coefficients (positive = predicts widening):")
        print(coefs)

        return self

    def predict(self, df):
        """Predict direction"""
        X = df[self.features]
        proba = self.model.predict_proba(X)[:, 1]  # Probability of widening

        return proba

    def backtest_strategy(self, df):
        """
        Backtest trading strategy:
        - If predict widening (>60%), go risk-off
        - If predict tightening (<40%), go risk-on
        """

        df['predicted_widen_proba'] = self.predict(df)
        df['actual_direction'] = (df['cp_tbill_spread'].diff() > 0).astype(int)

        # Strategy signals
        df['signal'] = 0
        df.loc[df['predicted_widen_proba'] > 0.6, 'signal'] = -1  # Risk off
        df.loc[df['predicted_widen_proba'] < 0.4, 'signal'] = 1   # Risk on

        # Accuracy when we take positions
        positions = df[df['signal'] != 0]
        correct = (
            ((positions['signal'] == -1) & (positions['actual_direction'] == 1)) |
            ((positions['signal'] == 1) & (positions['actual_direction'] == 0))
        ).mean()

        print(f"\nDirectional Accuracy (when positioned): {correct:.1%}")
        print(f"Number of signals: {len(positions)}")

        return correct


# Usage
predictor = SpreadDirectionPredictor()
predictor.train(df.loc[:'2022-12-31'])
accuracy = predictor.backtest_strategy(df.loc['2023-01-01':])
```

### Why This Works:

- **Not predicting exact value** (impossible)
- **Predicting direction** (much easier)
- **Accuracy 55-65%** is enough for profitable positioning
- **Interpretable** (logistic regression shows WHY)

---

## 5. Ensemble Model (Combine All)

### The Ultimate Predictor: Combine multiple models

```python
class EnsemblePredictor:
    """
    Combine Random Forest, XGBoost, and Logistic Regression.

    Often outperforms any single model.
    """

    def __init__(self, horizon=5):
        self.rf = CrisisPredictor(horizon=horizon)
        self.xgb = XGBoostCrisisPredictor(horizon=horizon)
        self.lr = LogisticRegression()
        self.horizon = horizon

    def train(self, df):
        """Train all models"""

        # Train base models
        self.rf.train(df)
        self.xgb.train(df)

        # Simple logistic for speed
        features = ['VIX', 'cp_tbill_spread', 'HY_OAS', 'DISCOUNT_WINDOW']
        df_labels = self.rf.create_labels(df)
        X = df[features].dropna()
        y = df_labels.loc[X.index, 'crisis_ahead']
        self.lr.fit(X, y)
        self.lr_features = features

        return self

    def predict_proba(self, df):
        """
        Ensemble prediction (average of all models).

        You can also use weighted average if some models perform better.
        """

        rf_proba = self.rf.predict_proba(df)
        xgb_proba = self.xgb.predict_proba(df)
        lr_proba = self.lr.predict_proba(df[self.lr_features])[:, 1]

        # Simple average (or use weighted: [0.4, 0.4, 0.2])
        ensemble_proba = (rf_proba + xgb_proba + lr_proba) / 3

        return ensemble_proba

    def predict_with_confidence(self, df):
        """
        Return prediction + confidence.

        High agreement between models = high confidence.
        """

        rf_proba = self.rf.predict_proba(df)
        xgb_proba = self.xgb.predict_proba(df)
        lr_proba = self.lr.predict_proba(df[self.lr_features])[:, 1]

        # Average probability
        avg_proba = (rf_proba + xgb_proba + lr_proba) / 3

        # Confidence = agreement (low std = high confidence)
        std_proba = np.std([rf_proba, xgb_proba, lr_proba], axis=0)
        confidence = 1 - std_proba  # Low std = high confidence

        return avg_proba, confidence


# Usage
ensemble = EnsemblePredictor(horizon=5)
ensemble.train(df.loc[:'2022-12-31'])

proba, confidence = ensemble.predict_with_confidence(df.iloc[[-1]])

print(f"Crisis Probability (next 5 days): {proba[-1]:.1%}")
print(f"Model Confidence: {confidence[-1]:.1%}")

if proba[-1] > 0.5 and confidence[-1] > 0.8:
    print("⚠️  HIGH CONFIDENCE WARNING: Crisis likely")
elif proba[-1] > 0.5:
    print("⚠️  WARNING: Crisis possible (low confidence)")
```

---

## 6. What Can Be Predicted - Summary Table

| Target | Predictable? | Method | Expected Accuracy | Useful? |
|--------|--------------|--------|-------------------|---------|
| **Exact S&P price** | ❌ NO | - | 0% | No |
| **Crisis in 5 days** | ✅ YES | RF/XGBoost | AUC 0.75-0.85 | YES |
| **Spread direction** | ✅ YES | Logistic | 55-65% | YES |
| **Volatility regime** | ✅ YES | LSTM/HMM | 60-70% | YES |
| **Exact VIX level** | ❌ NO | - | Poor | No |
| **Probability of stress** | ✅ YES | Ensemble | AUC 0.80+ | YES |
| **Time to crisis** | ⚠️ MAYBE | Survival | Moderate | Maybe |
| **Severity (mild/severe)** | ✅ YES | Multi-class | 50-60% | YES |
| **Days until recovery** | ❌ NO | - | Poor | No |
| **Feature importance** | ✅ YES | Any tree | 100% | YES |

---

## 7. Practical Implementation Roadmap

### Week 1: Random Forest Crisis Classifier
```python
# File: macro_plumbing/models/crisis_classifier.py
- Implement CrisisPredictor class
- Backtest on 2015-2024 data
- Measure AUC, precision, recall
- Feature importance analysis
```

### Week 2: XGBoost + Spread Direction
```python
# File: macro_plumbing/models/xgboost_predictor.py
# File: macro_plumbing/models/spread_direction.py
- Implement XGBoostCrisisPredictor
- Implement SpreadDirectionPredictor
- Compare with Random Forest
- Calibration analysis
```

### Week 3: LSTM for Regimes
```python
# File: macro_plumbing/models/lstm_regime.py
- Implement VolatilityRegimeLSTM
- Train on sequences
- Predict next regime
```

### Week 4: Ensemble + Dashboard
```python
# File: macro_plumbing/models/ensemble.py
- Combine all models
- Add to Streamlit dashboard
- Real-time predictions
- Confidence intervals
```

---

## 8. Critical Best Practices

### ✅ DO:

1. **Time-series split** (never use future data)
2. **Walk-forward validation** (train on past, test on future)
3. **Report confidence** (not just prediction)
4. **Measure calibration** (predicted 30% = actual 30%)
5. **Feature engineering** (lags, momentum, volatility)
6. **Handle imbalance** (crises are rare, use class weights)
7. **Backtesting on multiple crises** (2008, 2020, 2023)

### ❌ DON'T:

1. **Random CV** (causes leakage in time series)
2. **Predict exact prices** (impossible)
3. **Use only one crisis for validation** (overfits)
4. **Ignore imbalance** (model predicts "no crisis" 100% of time)
5. **Cherry-pick metrics** (report AUC, not just accuracy)
6. **Use future data** (even accidentally)
7. **Over-complicate** (simple models often win)

---

## 9. Honest Performance Expectations

### Random Forest Crisis Classifier (5-day horizon):

**Realistic Performance:**
- AUC: 0.75-0.85
- Recall (crisis detection): 70-85%
- Precision (when it says crisis): 20-40%
- Lead time: 2-7 days

**Why precision is "low":**
- Base rate: 5% of days are crisis
- 40% precision = **8x better than random**
- False alarms are okay (better safe than sorry)

### Spread Direction (1-day horizon):

**Realistic Performance:**
- Accuracy: 55-65%
- 60% accuracy = profitable
- Win rate matters less than sizing

### Volatility Regime (5-day horizon):

**Realistic Performance:**
- Accuracy: 60-70%
- Best at: Regime persistence
- Worst at: Black swans
- 3-class problem (harder than binary)

---

## 10. Code to Get Started Now

```python
# Quick start: 10 minutes to first prediction

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_pickle('macro_data.pkl')

# Define crisis
crisis = (
    (df['VIX'] > 35) |
    (df['cp_tbill_spread'] > 150) |
    (df['DISCOUNT_WINDOW'] > 10000)
)

# Label: crisis in next 5 days
df['crisis_ahead_5d'] = crisis.shift(-5).fillna(0).astype(int)

# Features
features = [
    'VIX', 'cp_tbill_spread', 'HY_OAS', 'bbb_aaa_spread',
    'DISCOUNT_WINDOW', 'delta_rrp', 'T10Y2Y'
]

# Split (time-based!)
X_train = df.loc[:'2022-12-31', features].dropna()
y_train = df.loc[X_train.index, 'crisis_ahead_5d']

X_test = df.loc['2023-01-01':, features].dropna()
y_test = df.loc[X_test.index, 'crisis_ahead_5d']

# Train
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf.fit(X_train, y_train)

# Predict
proba = rf.predict_proba(X_test)[:, 1]

print(f"Current crisis probability: {proba[-1]:.1%}")

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nMost important features:")
print(importance)
```

---

## Conclusion

**You CAN predict:**
- Crisis probability (next 5 days) - **AUC 0.75-0.85**
- Spread direction (widen/tighten) - **Accuracy 55-65%**
- Volatility regime (low/med/high) - **Accuracy 60-70%**
- Feature importance - **Always useful**

**You CANNOT predict:**
- Exact prices
- Exact timing
- Turning points
- Exact volatility levels

**Use Random Forest for:**
- Crisis classification (best all-around)
- Feature importance
- Fast prototyping

**Use XGBoost for:**
- Imbalanced data (crises are rare)
- Better calibrated probabilities
- Production models

**Use LSTM for:**
- Regime transitions
- Temporal patterns
- When you have lots of data

**Use Ensemble for:**
- Best performance
- Confidence estimates
- Production systems

Now let's build something that actually predicts **probabilities we can act on** instead of bullshit "market will hit 4,237" forecasts.
