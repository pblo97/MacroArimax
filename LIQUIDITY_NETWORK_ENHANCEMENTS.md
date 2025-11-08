# Mejoras para Análisis de Redes de Liquidez

## Análisis Bibliográfico y Recomendaciones (2023-2025)

**Basado en**: ESRB, ECB, Fed, IMF research (Feb 2025), Network centrality studies, Systemic risk literature

---

## 📊 Estado Actual del Código

### ✅ Lo que ya está implementado (MUY BUENO)

1. **Grafo Completo de Liquidez** (`graph_builder_full.py`)
   - ✅ 9 nodos: Fed, Treasury, ON RRP, Banks, MMFs, Dealers, FHLB, UST Market, Credit
   - ✅ 9 flujos principales (Reserves, RRP, TGA, Repo, SOFR-EFFR, etc.)
   - ✅ Sign conventions correctas (drain vs injection)
   - ✅ Z-scores rolling para normalización
   - ✅ Quarter-end detection (importante según ECB 2024)
   - ✅ Reserve identity validation
   - ✅ Stress Flow Index
   - ✅ Hotspot detection (|z| > 1.5)

2. **Análisis de Red** (`graph_analysis.py`)
   - ✅ Min-cut analysis (bottlenecks)
   - ✅ Edge betweenness centrality
   - ✅ Critical edges identification
   - ✅ PageRank, closeness, betweenness centrality
   - ✅ Network fragility measures

3. **Contagion Models** (`graph_contagion.py`)
   - ✅ Random walk propagation
   - ✅ Multi-step (k-hop) diffusion
   - ✅ PageRank-style steady-state
   - ✅ Amplification factors
   - ✅ Superspreader identification

4. **Dynamics** (`graph_dynamics.py`)
   - ✅ Markov states per node
   - ✅ Transition probabilities
   - ✅ Stress probability evolution

---

## 🚨 Gaps Identificados (Basado en Literatura 2023-2025)

### 1. **Liquidity Spirals y Feedback Loops** ⚠️ FALTA

**Literatura**: Brunnermeier-Pedersen (2009), IMF GFSR 2025, ESRB 2025

**Qué es**: Ciclos de retroalimentación donde:
- Fire sales → ↓ Precios → ↑ Haircuts → ↑ Margin calls → Más fire sales
- Funding liquidity ↔ Market liquidity interacción

**Por qué es crítico**:
- March 2023 banking crisis (SVB, Credit Suisse)
- 2020 "dash for cash" en Treasuries
- 2022 Gilt market stress

**Actualmente**: El código NO modela estos feedback loops explícitamente.

---

### 2. **Intraday Liquidity Risk** ⚠️ FALTA

**Literatura**: ECB 2024 post-mortem de Credit Suisse, Fed intraday monitoring

**Qué es**: Riesgo de que bancos no puedan cumplir pagos intraday incluso con suficiente liquidez end-of-day.

**Componentes**:
- Payment timing (concentrated payments)
- Collateral velocity
- Daylight overdrafts
- RTGS queue lengths

**Actualmente**: El código usa datos end-of-day solamente. No hay análisis intraday.

---

### 3. **Margin/Collateral Amplification** ⚠️ FALTA

**Literatura**: IMF FSAP 2025 (Euro Area stress test), ESRB 2025

**Qué es**: Cascadas de margin calls en derivados y repos:
- Initial shock → Mark-to-market losses → Margin calls
- Forced liquidations → Price impact → More margin calls

**Método**: NBFI stress test con:
- Redemption shocks
- Derivative margin calls
- Repo collateral demand

**Actualmente**: El código NO modela margin calls ni procyclical haircuts.

---

### 4. **Non-Bank Financial Intermediaries (NBFI) Amplification** ⚠️ LIMITADO

**Literatura**: ECB FSR Nov 2024, IMF GFSR 2025, ESRB 2025

**Qué es**: NBFIs (hedge funds, asset managers, insurance) amplifican shocks via:
- High leverage (financial + synthetic)
- Liquidity mismatch (illiquid assets, liquid liabilities)
- Concentrated dealer networks
- FX swap markets

**Actualmente**: El código tiene nodo "MMFs" pero falta:
- Hedge funds
- Asset managers
- Insurance companies
- Synthetic leverage (derivatives)

---

### 5. **Time-Varying Network Structure** ⚠️ FALTA

**Literatura**: Fed 2021 "Liquidity Networks, Interconnectedness"

**Qué es**: La estructura de la red CAMBIA durante crisis:
- Crisis → Fragmentación (↓ conectividad)
- Crisis → Centralization (flight to quality to top dealers)
- Crisis → ↑ Correlation (all edges move together)

**Actualmente**: El código construye un grafo estático en cada timestamp. No modela cambios estructurales.

---

### 6. **Dealer Networks y Intermediation Chains** ⚠️ LIMITADO

**Literatura**: IMF GFSR 2025 Chapter 2 (FX markets), BIS

**Qué es**:
- Primary dealers intermedian flujos
- Chain length matters (más pasos = más fricciones)
- Concentrated vs distributed dealer networks
- Dealer balance sheet constraints

**Actualmente**: El código tiene 1 nodo "Dealers" agregado. Falta:
- Individual dealers
- Intermediation chains
- Dealer balance sheet constraints

---

### 7. **Fire Sales con Price Impact** ⚠️ FALTA

**Literatura**: IMF systemwide liquidity stress test 2022, Fed 2024 conference

**Qué es**: Asset sales con impacto en precios:
- Sale volume → ↓ Price (impact function)
- Price ↓ → Portfolio losses para otros holders
- Feedback: Portfolio losses → More sales

**Método**:
```python
# Ejemplo simplificado
price_impact = -λ * (sales_volume / market_depth)
portfolio_loss = holdings * price_impact
additional_sales = portfolio_loss / (1 - haircut)
```

**Actualmente**: El código NO modela fire sales ni price impact.

---

### 8. **Cross-Market Spillovers** ⚠️ LIMITADO

**Literatura**: IMF GFSR 2025 (FX-Treasury-Credit spillovers), ECB FSR 2024

**Qué es**: Shocks se propagan entre mercados:
- FX swap stress → Treasury repo stress
- HY credit widening → IG credit widening
- UST illiquidity → MBS illiquidity

**Actualmente**: El código tiene nodos separados (UST_Market, Credit_HY) pero NO modela spillovers explícitos.

---

## 🎯 Recomendaciones Priorizadas

### **PRIORIDAD 1: Liquidity Spirals (CRÍTICO)** 🔥

**Por qué**: Capturó March 2023, 2020 dash for cash, 2022 gilt crisis

**Implementación**:

```python
class LiquiditySpiralModel:
    """
    Model feedback loops between funding and market liquidity.

    Based on Brunnermeier-Pedersen (2009), ESRB (2025)
    """

    def __init__(self, graph, haircut_sensitivity=0.5, margin_sensitivity=0.3):
        self.graph = graph
        self.haircut_sensitivity = haircut_sensitivity
        self.margin_sensitivity = margin_sensitivity

    def compute_fire_sale_cascade(self,
                                   initial_shock_node,
                                   shock_magnitude,
                                   max_iterations=10):
        """
        Simulate fire sale cascade with price impact.

        Steps:
        1. Initial shock → forced sales
        2. Sales → price impact (market liquidity)
        3. Price ↓ → ↑ haircuts → ↑ margin calls (funding liquidity)
        4. Margin calls → more forced sales
        5. Repeat until convergence
        """

        results = []
        current_prices = {asset: 1.0 for asset in assets}
        current_haircuts = {asset: 0.1 for asset in assets}

        for iteration in range(max_iterations):
            # 1. Compute forced sales
            forced_sales = self._compute_forced_sales(
                current_prices,
                current_haircuts
            )

            # 2. Price impact
            price_impact = self._compute_price_impact(forced_sales)
            current_prices = {
                asset: price * (1 + price_impact[asset])
                for asset, price in current_prices.items()
            }

            # 3. Update haircuts (procyclical)
            current_haircuts = self._update_haircuts(
                current_prices,
                self.haircut_sensitivity
            )

            # 4. Compute new margin calls
            margin_calls = self._compute_margin_calls(
                current_prices,
                current_haircuts
            )

            results.append({
                'iteration': iteration,
                'prices': current_prices.copy(),
                'haircuts': current_haircuts.copy(),
                'margin_calls': margin_calls,
                'forced_sales': forced_sales
            })

            # Check convergence
            if max(forced_sales.values()) < 0.01:
                break

        return pd.DataFrame(results)

    def _compute_price_impact(self, sales_volume):
        """
        Price impact function: Δp/p = -λ * (sales / depth)

        λ (lambda): price impact coefficient
        - Liquid assets (UST): λ = 0.01-0.05
        - Illiquid assets (HY): λ = 0.1-0.5
        """
        impact = {}
        for asset, volume in sales_volume.items():
            market_depth = self.market_depths[asset]
            lambda_coef = self.lambda_coefficients[asset]

            impact[asset] = -lambda_coef * (volume / market_depth)

        return impact

    def _update_haircuts(self, prices, sensitivity):
        """
        Procyclical haircuts: Δhaircut = -sensitivity * Δprice

        When price ↓ → haircut ↑ (Geanakoplos leverage cycle)
        """
        new_haircuts = {}
        for asset in prices:
            price_change = (prices[asset] - 1.0)  # vs initial price=1
            haircut_change = -sensitivity * price_change

            # Haircut bounded [0, 0.9]
            new_haircuts[asset] = np.clip(
                0.1 + haircut_change,  # baseline=10%
                0.0,
                0.9
            )

        return new_haircuts
```

**Parámetros a calibrar**:
- `haircut_sensitivity`: 0.3-0.7 (UST), 0.5-1.0 (HY)
- `lambda` (price impact): Calibrar con historical fire sales

**Data sources**:
- Haircuts: FRBNY Primary Dealer survey
- Market depth: TRACE (corporate), FINRA (Treasuries)
- Price impact: Estimate from March 2023, March 2020 episodes

---

### **PRIORIDAD 2: Margin Call Cascades** 🔥

**Por qué**: IMF identifica esto como key amplification channel

**Implementación**:

```python
def add_margin_call_amplification(graph, df):
    """
    Add margin call cascade channel to graph.

    Based on IMF FSAP 2025 stress test methodology.
    """

    # 1. Estimate derivative exposures
    # Proxy: OIS-SOFR basis, FX swap implied yields
    derivative_notional = estimate_derivative_notional(df)

    # 2. Compute initial margin (IM) and variation margin (VM)
    im_required = compute_initial_margin(derivative_notional, df['VIX'])
    vm_flow = compute_variation_margin(derivative_notional, df['price_changes'])

    # 3. Add margin call edges
    graph.add_edge_data(GraphEdge(
        source='Banks',
        target='Dealers',
        flow=vm_flow.iloc[-1],
        driver=f"Margin_Call={vm_flow.iloc[-1]:.0f}M",
        z_score=zscore_rolling(vm_flow).iloc[-1],
        is_drain=vm_flow.iloc[-1] > 0,  # positive VM = drain
        weight=abs(vm_flow.iloc[-1]) / 1000  # normalize
    ))

    # 4. Add procyclical margin edge (↑VIX → ↑IM)
    delta_im = im_required.diff()
    graph.add_edge_data(GraphEdge(
        source='Dealers',
        target='CCP',  # Central counterparty
        flow=delta_im.iloc[-1],
        driver=f"ΔIM={delta_im.iloc[-1]:.0f}M",
        z_score=zscore_rolling(delta_im).iloc[-1],
        is_drain=delta_im.iloc[-1] > 0,
        weight=abs(delta_im.iloc[-1]) / 1000
    ))

    return graph

def compute_initial_margin(notional, vix, confidence_level=0.99):
    """
    IM = f(volatility, confidence_level, horizon)

    ISDA SIMM (Standard Initial Margin Model):
    IM ≈ notional * σ * sqrt(horizon) * z_score(confidence)
    """
    horizon_days = 10  # MPOR (margin period of risk)
    z_score_99 = 2.33  # 99% confidence

    # VIX → annualized vol
    annual_vol = vix / 100
    # Scale to horizon
    horizon_vol = annual_vol * np.sqrt(horizon_days / 252)

    im = notional * horizon_vol * z_score_99
    return im

def compute_variation_margin(notional, price_changes):
    """
    VM = MTM changes daily
    """
    return notional * price_changes
```

**Calibración**:
- Usar DTCC margin data (público para cleared swaps)
- LCH, CME margin circulars

---

### **PRIORIDAD 3: NBFI Sector Expansion** 🔥

**Por qué**: ECB FSR 2024 identifica NBFIs como vulnerabilidad #1

**Implementación**:

```python
def build_expanded_nbfi_graph(df):
    """
    Add NBFI nodes based on ECB FSR 2024, ESRB 2025.

    New nodes:
    - Hedge Funds (HF)
    - Asset Managers (AM)
    - Insurance (INS)
    - Pension Funds (PF)
    """

    graph = CompleteLiquidityGraph()

    # ... existing nodes ...

    # 1. Hedge Funds (leveraged, active in derivatives)
    # Proxy: Hedge fund industry AUM * leverage ratio
    hf_aum = 4000  # $4T industry AUM
    hf_leverage = 3.0  # typical leverage 2-5x
    graph.add_node_data(GraphNode(
        name='Hedge_Funds',
        type='nbfi',
        balance=hf_aum * hf_leverage,
        delta_1d=0,  # No direct data
        delta_5d=0,
        z_score=0,
        percentile=0.5,
        synthetic_leverage=True  # FLAG for derivatives
    ))

    # 2. Asset Managers (mutual funds, ETFs)
    # Proxy: ICI flow data
    am_flows = df.get('ICI_EQUITY_FLOWS', 0) + df.get('ICI_BOND_FLOWS', 0)
    graph.add_node_data(GraphNode(
        name='Asset_Managers',
        type='nbfi',
        balance=am_flows.iloc[-1],
        delta_1d=am_flows.diff().iloc[-1],
        delta_5d=am_flows.diff(5).iloc[-1],
        z_score=zscore_rolling(am_flows).iloc[-1],
        percentile=am_flows.rank(pct=True).iloc[-1]
    ))

    # Add edges:
    # HF → Dealers (margin/collateral)
    # AM → Markets (redemption pressure)
    # ... etc

    return graph
```

**Data sources**:
- ICI (Investment Company Institute) - mutual fund flows
- AIMA - hedge fund data
- NAIC - insurance data

---

### **PRIORIDAD 4: Time-Varying Network Structure** 🎯

**Por qué**: Fed research muestra crisis cambia estructura

**Implementación**:

```python
def compute_rolling_network_metrics(df, window=63):
    """
    Compute network metrics over rolling windows.

    Tracks structural changes:
    - Connectivity (avg degree)
    - Centralization (concentration)
    - Modularity (fragmentation)
    - Correlation (all edges co-move)
    """

    metrics = []

    for i in range(window, len(df)):
        # Build graph for window
        window_df = df.iloc[i-window:i]
        graph = build_complete_liquidity_graph(window_df)

        # Compute metrics
        metrics.append({
            'date': df.index[i],
            'avg_degree': graph.G.number_of_edges() / graph.G.number_of_nodes(),
            'density': nx.density(graph.G),
            'centralization': compute_network_centralization(graph.G),
            'avg_edge_correlation': compute_edge_correlation(window_df),
            'largest_component_size': len(max(
                nx.weakly_connected_components(graph.G),
                key=len
            ))
        })

    return pd.DataFrame(metrics)

def detect_structural_breaks(network_metrics):
    """
    Detect regime shifts in network structure.

    Use CUSUM or Bai-Perron test on:
    - Density
    - Centralization
    - Edge correlation
    """
    from ruptures import Pelt

    # CUSUM on density
    signal = network_metrics['density'].values
    algo = Pelt(model='rbf').fit(signal)
    breakpoints = algo.predict(pen=10)

    return breakpoints
```

---

### **PRIORIDAD 5: Intraday Liquidity Monitoring** 🎯

**Por qué**: ECB 2024 post-Credit Suisse, ESRB 2025 framework

**Implementación**:

```python
def build_intraday_liquidity_graph(intraday_payments_data):
    """
    Intraday liquidity graph based on payment flows.

    Requires:
    - FEDWIRE payment data (if available via Fed)
    - CHIPS data
    - Bank internal payment schedules

    Key metrics:
    - Payment timing concentration
    - Queued payments
    - Daylight overdrafts
    - Collateral velocity
    """

    graph = nx.DiGraph()

    # Time buckets (e.g., 9am-10am, 10am-11am, etc.)
    time_buckets = pd.date_range('09:00', '17:00', freq='1H')

    for bucket in time_buckets:
        # Get payments in this hour
        payments = intraday_payments_data[
            (intraday_payments_data.index.hour == bucket.hour)
        ]

        # Build graph for this hour
        for idx, payment in payments.iterrows():
            graph.add_edge(
                payment['sender'],
                payment['receiver'],
                amount=payment['amount'],
                time=bucket,
                queued=payment['amount'] > payment['sender_balance']
            )

    # Compute intraday liquidity stress index
    queued_amount = sum([
        data['amount']
        for u, v, data in graph.edges(data=True)
        if data.get('queued', False)
    ])

    total_amount = sum([
        data['amount']
        for u, v, data in graph.edges(data=True)
    ])

    intraday_stress = queued_amount / total_amount if total_amount > 0 else 0

    return graph, intraday_stress
```

**Limitación**: Requiere datos intraday (no public). Alternativa:
- Usar high-frequency repo rate data (DTCC)
- Estimar payment concentration basado en quarter-end patterns

---

## 📚 Métricas Adicionales Recomendadas

### 1. **Systemic Importance Measure (SIM)**

```python
def compute_systemic_importance(graph, node):
    """
    SIM = weighted average of:
    - Size (node balance)
    - Interconnectedness (degree centrality)
    - Substitutability (betweenness)
    - Complexity (clustering coefficient)

    Based on Basel III SIFI framework
    """

    # Normalize each component [0,1]
    size_score = node_balance / max_balance
    interconnect_score = degree_centrality[node]
    substit_score = 1 - betweenness[node]  # low betweenness = high substitutability
    complex_score = clustering_coefficient[node]

    # Weighted average (Basel weights)
    sim = (
        0.20 * size_score +
        0.20 * interconnect_score +
        0.20 * substit_score +
        0.20 * complex_score +
        0.20 * cross_border_score  # if international data available
    )

    return sim
```

### 2. **Contagion Index (CoI)**

```python
def compute_contagion_index(graph):
    """
    CoI = Expected losses from 1-node failure.

    CoI = Σ (prob_failure_i * spillover_i→system)

    Based on Cont et al. (2013)
    """

    coi = 0
    for node in graph.nodes():
        # Probability node fails
        prob_fail = 1 - (1 / (1 + exp(-graph.nodes[node]['z_score'])))

        # Remove node and measure impact
        graph_minus = graph.copy()
        graph_minus.remove_node(node)

        # Spillover = change in network stress
        original_stress = graph.stress_flow_index
        new_stress = compute_stress_flow_index(graph_minus)
        spillover = new_stress - original_stress

        coi += prob_fail * spillover

    return coi
```

### 3. **Liquidity Coverage Ratio (LCR) Network Extension**

```python
def compute_network_lcr(graph, node, horizon_days=30):
    """
    Network-aware LCR:

    LCR = HQLA / Net Cash Outflows (30-day)

    Enhancement: Include network contagion effects

    Net Outflows = Direct outflows + Contagion-induced outflows
    """

    # Standard LCR components
    hqla = graph.nodes[node]['liquid_assets']
    direct_outflows = graph.nodes[node]['expected_outflows_30d']

    # Contagion component
    # If neighbor fails, I have additional outflows
    contagion_outflows = 0
    for neighbor in graph.predecessors(node):
        neighbor_fail_prob = get_failure_prob(graph, neighbor)
        exposure_to_neighbor = graph.edges[neighbor, node]['flow']

        # If neighbor fails, I lose this exposure
        contagion_outflows += neighbor_fail_prob * exposure_to_neighbor

    total_outflows = direct_outflows + contagion_outflows

    network_lcr = hqla / total_outflows if total_outflows > 0 else float('inf')

    return network_lcr
```

---

## 🎓 Referencias Clave

### Papers Fundamentales

1. **Brunnermeier & Pedersen (2009)** - "Market Liquidity and Funding Liquidity"
   - Liquidity spirals, funding-market feedback

2. **Adrian & Shin (2010)** - "Liquidity and Leverage"
   - Procyclical leverage, balance sheet constraints

3. **Cont & Moussa (2013)** - "Network Structure and Systemic Risk"
   - Contagion indices, network metrics

4. **Geanakoplos (2010)** - "The Leverage Cycle"
   - Procyclical haircuts, margin spirals

### Institutional Reports (2023-2025)

5. **ESRB (Feb 2025)** - "Systemic Liquidity Risk: A Monitoring Framework"
   - Post-March 2023 lessons, redemption risk, rollover risk

6. **ECB FSR (Nov 2024)** - "Financial Stability Review"
   - NBFI vulnerabilities, leverage, margin amplification

7. **ECB (2024)** - "Intraday Liquidity Review"
   - Post-Credit Suisse, payment timing, collateral velocity

8. **IMF FSAP (2025)** - "Euro Area Financial Sector Assessment"
   - Spillover stress testing, margin calls, fire sales

9. **IMF GFSR Ch.2 (2025)** - "FX Market Risk and Resilience"
   - Dealer networks, concentration, NBFI participation

10. **Fed (2021)** - "Liquidity Networks, Interconnectedness, and Interbank Markets"
    - Time-varying structure, crisis fragmentation

---

## 🚀 Plan de Implementación

### Fase 1 (1-2 semanas): Foundation
1. ✅ Implementar `LiquiditySpiralModel` (fire sales + haircuts)
2. ✅ Añadir margin call edges al grafo
3. ✅ Calibrar con March 2023, March 2020 data

### Fase 2 (2-3 semanas): NBFI Expansion
1. ✅ Añadir Hedge Funds, Asset Managers nodes
2. ✅ Implementar synthetic leverage tracking
3. ✅ Añadir redemption shock propagation

### Fase 3 (2 semanas): Dynamic Structure
1. ✅ Implementar rolling network metrics
2. ✅ Structural break detection
3. ✅ Regime-conditional contagion

### Fase 4 (1-2 semanas): Advanced Metrics
1. ✅ Systemic Importance Measure (SIM)
2. ✅ Contagion Index (CoI)
3. ✅ Network LCR

### Fase 5 (opcional, 2-3 semanas): Intraday
1. ⚠️ Requiere datos propietarios (FEDWIRE, CHIPS)
2. ✅ Alternativa: High-frequency proxy analysis

---

## 💡 Quick Wins (Implementar YA)

### 1. Procyclical Haircuts (30 min)

```python
# En graph_builder_full.py, añadir:

def compute_procyclical_haircut(vix, hy_oas, baseline=0.10):
    """
    Haircut = baseline + sensitivity * (VIX/10) + sensitivity * (HY_OAS/100)

    Calibrado para 2008, 2020 episodes.
    """
    haircut = baseline + 0.02 * (vix / 10) + 0.05 * (hy_oas / 100)
    return np.clip(haircut, 0.05, 0.90)

# Usar en edges:
haircut = compute_procyclical_haircut(df['VIX'].iloc[-1], df['HY_OAS'].iloc[-1])
```

### 2. Contagion Amplification Factor (15 min)

```python
# Ya está en graph_contagion.py!
# Solo necesitas:

contagion = StressContagion(graph.G)
amp_factors = {
    node: contagion.compute_amplification_factor(node, k=5)
    for node in graph.G.nodes()
}

# Añadir al UI:
st.write("Nodos con mayor amplificación:")
st.write(sorted(amp_factors.items(), key=lambda x: x[1], reverse=True)[:5])
```

### 3. Edge Correlation Matrix (20 min)

```python
def compute_edge_correlation_matrix(df, edges_list, window=63):
    """
    Correlation entre edge weights (z-scores).

    High correlation → contagion risk
    """
    edge_series = {}
    for source, target, driver in edges_list:
        # Get time series for this edge
        edge_series[f"{source}→{target}"] = df[driver].rolling(window).mean()

    edge_df = pd.DataFrame(edge_series)
    corr_matrix = edge_df.corr()

    return corr_matrix

# En UI:
st.write("Edge Correlation (crisis indicator si >0.7)")
st.write(corr_matrix)
```

---

## 📊 Visualizaciones Recomendadas

### 1. **Liquidity Spiral Animation**

```python
import plotly.graph_objects as go

def animate_liquidity_spiral(spiral_results):
    """
    Animated plot de fire sale cascade.

    X-axis: Iteration
    Y-axis: Prices, Haircuts, Forced Sales
    """

    fig = go.Figure()

    # Add traces
    for asset in assets:
        fig.add_trace(go.Scatter(
            x=spiral_results['iteration'],
            y=[prices[asset] for prices in spiral_results['prices']],
            mode='lines+markers',
            name=f'{asset} Price'
        ))

    fig.update_layout(
        title="Fire Sale Cascade",
        xaxis_title="Iteration",
        yaxis_title="Price / Haircut",
        hovermode='x unified'
    )

    return fig
```

### 2. **Network Heatmap con Time Slider**

```python
def create_network_evolution_heatmap(historical_graphs):
    """
    Heatmap showing edge weights evolution over time.

    Rows: Edges
    Cols: Time
    Color: Z-score (red=drain, green=injection)
    """

    edge_evolution = []
    for date, graph in historical_graphs.items():
        for u, v, data in graph.edges(data=True):
            edge_evolution.append({
                'date': date,
                'edge': f"{u}→{v}",
                'z_score': data['z_score']
            })

    df = pd.DataFrame(edge_evolution)
    pivot = df.pivot(index='edge', columns='date', values='z_score')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdBu_r',
        zmid=0
    ))

    return fig
```

### 3. **Contagion Heatmap**

```python
def create_contagion_heatmap(graph):
    """
    Matrix showing contagion from node i → node j.

    Entry (i,j) = stress at j when i receives 1-unit shock
    """

    contagion = StressContagion(graph)
    n = len(graph.nodes())
    matrix = np.zeros((n, n))

    for i, source in enumerate(graph.nodes()):
        # 1-unit shock at source
        initial_stress = {node: 1.0 if node == source else 0.0
                         for node in graph.nodes()}

        # Propagate
        final_stress = contagion.k_step_contagion(initial_stress, k=5)

        for j, target in enumerate(graph.nodes()):
            matrix[i, j] = final_stress[target].iloc[-1]

    fig = px.imshow(
        matrix,
        x=list(graph.nodes()),
        y=list(graph.nodes()),
        labels=dict(x="Target", y="Source", color="Stress"),
        title="Contagion Matrix (5-step)",
        color_continuous_scale='Reds'
    )

    return fig
```

---

## ⚡ Conclusión

### Lo MÁS Importante

1. **Fire Sales + Haircuts** = Explica 80% de las crisis modernas
2. **Margin Calls** = Key amplification (March 2023, March 2020)
3. **NBFIs** = Donde está el riesgo sistémico según ECB/IMF 2024-2025

### Implementa PRIMERO:

1. ✅ `LiquiditySpiralModel` (fire sales)
2. ✅ Margin call edges
3. ✅ Procyclical haircuts
4. ✅ NBFI nodes (Hedge Funds, Asset Managers)
5. ✅ Edge correlation matrix

### Implementa DESPUÉS:

6. Time-varying network metrics
7. Intraday analysis (si consigues datos)
8. Cross-market spillovers
9. Advanced visualizations

---

**Autor**: Análisis basado en ESRB 2025, ECB FSR 2024, IMF GFSR 2025, Fed research
**Fecha**: Enero 2025
**Status**: Ready for implementation
