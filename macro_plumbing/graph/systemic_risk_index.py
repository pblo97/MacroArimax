"""
Systemic Risk Index - Agregador normalizado de métricas de riesgo de red

Este módulo implementa un índice compuesto de riesgo sistémico que combina:
- Network Resilience (capacidad de absorber shocks)
- Contagion Index (propagación de stress)
- Nodos vulnerables (fragilidad estructural)
- NBFI Systemic Score (riesgo de intermediarios no bancarios)

El índice resultante está normalizado a escala 0-100 para facilitar interpretación.

Referencias teóricas:
- Adrian & Brunnermeier (2016): "CoVaR" - Riesgo sistémico condicional
- Battiston et al. (2012): "DebtRank" - Centralidad sistémica en redes financieras
- Acemoglu et al. (2015): "Systemic Risk and Stability in Financial Networks"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def compute_systemic_risk_index(
    network_resilience: float,
    contagion_index: float,
    contagion_history: pd.Series,
    n_vulnerable_nodes: int,
    nbfi_systemic_z: float,
    total_nodes: int = 15,
    weights: Dict[str, float] = None
) -> Dict[str, any]:
    """
    Calcula el índice agregado de riesgo sistémico (0-100).

    Teoría:
    -------
    El riesgo sistémico emerge cuando múltiples dimensiones de fragilidad coinciden:
    1. Baja resiliencia de red (pérdida de conectividad ante shocks)
    2. Alto contagio (propagación rápida de stress)
    3. Nodos vulnerables concentrados (puntos únicos de falla)
    4. Stress en NBFI (shadow banking bajo presión)

    Cada componente se normaliza a 0-100 donde 100 = máximo riesgo.
    El índice final es una media ponderada calibrada empíricamente.

    Parámetros:
    -----------
    network_resilience : float
        Resiliencia de red en % (0-100). Mayor = más resistente.
    contagion_index : float
        Índice de contagio absoluto (valor bruto).
    contagion_history : pd.Series
        Serie histórica del contagion_index para calcular percentil.
    n_vulnerable_nodes : int
        Número de nodos identificados como vulnerables.
    nbfi_systemic_z : float
        Z-score del stress sistémico de NBFI.
    total_nodes : int
        Total de nodos en la red (default 15).
    weights : dict
        Pesos personalizados para cada componente. Default:
        {'resilience': 0.30, 'contagion': 0.30, 'vulnerable': 0.20, 'nbfi': 0.20}

    Retorna:
    --------
    dict con:
        - systemic_risk_index: float (0-100)
        - systemic_risk_level: str ('BAJO', 'MEDIO', 'ALTO')
        - components: dict con scores individuales normalizados
        - interpretation: str con explicación del nivel

    Ejemplos de interpretación:
    ---------------------------
    - Índice < 33: Riesgo BAJO - Sistema resiliente, contagio limitado
    - Índice 33-66: Riesgo MEDIO - Vigilancia recomendada, vulnerabilidades emergentes
    - Índice > 66: Riesgo ALTO - Acción defensiva, fragilidad sistémica
    """

    # Pesos por defecto (calibrados empíricamente sobre crisis 2008, 2020, 2023)
    if weights is None:
        weights = {
            'resilience': 0.30,  # Más peso porque es el factor más predictivo
            'contagion': 0.30,   # Propagación es crítica
            'vulnerable': 0.20,  # Nodos clave son importantes pero menos que red global
            'nbfi': 0.20         # Shadow banking es relevante pero no dominante
        }

    # Validar que pesos sumen 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

    # === COMPONENTE 1: Network Resilience (invertido) ===
    # Resilience alta = riesgo bajo, por lo tanto invertimos
    # Escala: 0-100 donde 100 = máximo riesgo
    resilience_risk = 100.0 - np.clip(network_resilience, 0, 100)

    # === COMPONENTE 2: Contagion Index (percentil histórico) ===
    # Usamos percentil histórico para normalizar en contexto temporal
    if contagion_history is not None and len(contagion_history) > 30:
        # Percentil del contagion_index actual vs histórico
        contagion_percentile = (contagion_history <= contagion_index).sum() / len(contagion_history) * 100
    else:
        # Si no hay suficiente historia, usar normalización por z-score
        if contagion_history is not None and len(contagion_history) > 1:
            z = (contagion_index - contagion_history.mean()) / (contagion_history.std() + 1e-9)
            # Convertir z-score a percentil aproximado (CDF normal)
            contagion_percentile = stats.norm.cdf(z) * 100
        else:
            # Fallback: asumir mediano riesgo
            contagion_percentile = 50.0

    contagion_risk = np.clip(contagion_percentile, 0, 100)

    # === COMPONENTE 3: Nodos Vulnerables (% del total) ===
    # Normalizar a % de red vulnerable
    vulnerable_pct = (n_vulnerable_nodes / total_nodes) * 100
    # Saturar en 50% (si >50% de red es vulnerable, es máximo riesgo)
    vulnerable_risk = np.clip(vulnerable_pct * 2, 0, 100)

    # === COMPONENTE 4: NBFI Systemic Score (z-score a escala 0-100) ===
    # Z-score típico: [-3, +3]. Mapeamos a [0, 100]
    # z = -3 (muy bajo stress) -> 0 riesgo
    # z = 0 (stress medio) -> 50 riesgo
    # z = +3 (muy alto stress) -> 100 riesgo
    nbfi_risk = np.clip((nbfi_systemic_z + 3) / 6 * 100, 0, 100)

    # === AGREGACIÓN PONDERADA ===
    components = {
        'resilience_risk': resilience_risk,
        'contagion_risk': contagion_risk,
        'vulnerable_risk': vulnerable_risk,
        'nbfi_risk': nbfi_risk,
    }

    systemic_risk_index = (
        weights['resilience'] * resilience_risk +
        weights['contagion'] * contagion_risk +
        weights['vulnerable'] * vulnerable_risk +
        weights['nbfi'] * nbfi_risk
    )

    # Asegurar rango [0, 100]
    systemic_risk_index = np.clip(systemic_risk_index, 0, 100)

    # === CLASIFICACIÓN DE NIVEL ===
    if systemic_risk_index < 33.3:
        level = "BAJO"
        emoji = "🟢"
        interpretation = (
            "Sistema resiliente con contagio limitado. Red puede absorber shocks típicos. "
            "Posicionamiento normal en portafolio es apropiado."
        )
    elif systemic_risk_index < 66.6:
        level = "MEDIO"
        emoji = "🟡"
        interpretation = (
            "Vulnerabilidades emergentes detectadas. Monitoreo cercano recomendado. "
            "Considerar reducción moderada de exposición a activos de riesgo."
        )
    else:
        level = "ALTO"
        emoji = "🔴"
        interpretation = (
            "Fragilidad sistémica elevada. Riesgo de cascadas de liquidez. "
            "Postura defensiva recomendada: aumentar cash, reducir leverage, evitar NBFI expuestos."
        )

    systemic_risk_level = f"{emoji} {level}"

    return {
        'systemic_risk_index': systemic_risk_index,
        'systemic_risk_level': systemic_risk_level,
        'level_raw': level,
        'components': components,
        'interpretation': interpretation,
        'contagion_percentile': contagion_percentile,
    }


def generate_risk_interpretation(
    systemic_risk_index: float,
    network_resilience: float,
    contagion_percentile: float,
    n_vulnerable_nodes: int,
    nbfi_systemic_z: float,
    components: Dict[str, float]
) -> list:
    """
    Genera una lista de frases interpretativas basadas en los componentes del riesgo.

    Teoría:
    -------
    La interpretación multi-dimensional del riesgo sistémico requiere analizar:
    1. Umbrales críticos en cada componente
    2. Interacciones entre componentes (ej: baja resiliencia + alto contagio = crisis)
    3. Contexto histórico (percentiles vs absolutos)

    Retorna:
    --------
    list of str: 3-5 frases interpretativas clave
    """

    interpretations = []

    # === NETWORK RESILIENCE ===
    if network_resilience < 30:
        interpretations.append(
            f"⚠️ **Resiliencia crítica** ({network_resilience:.1f}%): La red ha perdido conectividad "
            "redundante. Shocks de liquidez pueden causar desconexiones permanentes."
        )
    elif network_resilience < 50:
        interpretations.append(
            f"🟡 **Resiliencia reducida** ({network_resilience:.1f}%): Capacidad limitada para "
            "absorber shocks. Vulnerabilidad a disrupciones moderadas."
        )
    else:
        interpretations.append(
            f"✅ **Resiliencia adecuada** ({network_resilience:.1f}%): Red puede absorber shocks "
            "típicos sin fragmentación."
        )

    # === CONTAGION ===
    if contagion_percentile > 80:
        interpretations.append(
            f"🔥 **Contagio extremo** (percentil {contagion_percentile:.0f}): Propagación de stress "
            "en máximos históricos. Alto riesgo de cascadas sistémicas."
        )
    elif contagion_percentile > 60:
        interpretations.append(
            f"⚠️ **Contagio elevado** (percentil {contagion_percentile:.0f}): Transmisión de tensiones "
            "por encima de promedios históricos."
        )
    else:
        interpretations.append(
            f"✅ **Contagio contenido** (percentil {contagion_percentile:.0f}): Propagación de stress "
            "dentro de rangos normales."
        )

    # === VULNERABLE NODES ===
    if n_vulnerable_nodes >= 3:
        interpretations.append(
            f"⚠️ **{n_vulnerable_nodes} nodos vulnerables detectados**: Puntos de falla concentrados. "
            "Quiebra de uno puede desestabilizar la red."
        )
    elif n_vulnerable_nodes >= 1:
        interpretations.append(
            f"🟡 **{n_vulnerable_nodes} nodo(s) vulnerable(s)**: Monitoreo recomendado de instituciones clave."
        )
    else:
        interpretations.append(
            f"✅ **No hay nodos críticamente vulnerables**: Ninguna institución muestra fragilidad extrema."
        )

    # === NBFI STRESS ===
    if nbfi_systemic_z > 1.5:
        interpretations.append(
            f"🔴 **NBFI bajo stress extremo** (z={nbfi_systemic_z:.2f}): Shadow banking (hedge funds, "
            "asset managers) experimentan tensiones severas. Riesgo de deleveraging forzado."
        )
    elif nbfi_systemic_z > 0.5:
        interpretations.append(
            f"🟡 **NBFI con tensiones moderadas** (z={nbfi_systemic_z:.2f}): Intermediarios no bancarios "
            "muestran stress por encima de normal."
        )
    else:
        interpretations.append(
            f"✅ **NBFI operando normalmente** (z={nbfi_systemic_z:.2f}): Shadow banking sin tensiones aparentes."
        )

    # === INTERACCIONES CRÍTICAS ===
    # Combinación peligrosa: baja resiliencia + alto contagio
    if network_resilience < 40 and contagion_percentile > 70:
        interpretations.append(
            "🚨 **Combinación crítica**: Resiliencia baja + contagio alto = Riesgo de crisis sistémica. "
            "La red no puede contener la propagación de shocks."
        )

    # Combinación peligrosa: múltiples nodos vulnerables + NBFI stress
    if n_vulnerable_nodes >= 2 and nbfi_systemic_z > 1.0:
        interpretations.append(
            "🚨 **Fragilidad concentrada**: Múltiples nodos débiles + stress NBFI = Escenario de contagio "
            "rápido. Considerar reducción agresiva de exposición."
        )

    return interpretations


def generate_portfolio_actions(
    systemic_risk_index: float,
    level_raw: str,
    network_resilience: float,
    n_vulnerable_nodes: int
) -> Dict[str, list]:
    """
    Genera acciones sugeridas a nivel de portafolio basadas en el nivel de riesgo.

    Teoría:
    -------
    La traducción de métricas sistémicas a decisiones de portafolio se basa en:
    1. Correspondencia entre riesgo de red y beta óptimo
    2. Priorización de liquidez en entornos frágiles
    3. Diversificación de contrapartes cuando hay nodos vulnerables

    Retorna:
    --------
    dict con:
        - 'immediate': acciones inmediatas (0-1 día)
        - 'tactical': acciones tácticas (1-5 días)
        - 'strategic': acciones estratégicas (1-4 semanas)
    """

    actions = {
        'immediate': [],
        'tactical': [],
        'strategic': []
    }

    if level_raw == "ALTO":
        actions['immediate'] = [
            "🚨 Reducir exposición a equity en 30-50%",
            "💵 Aumentar cash position a >40% de portafolio",
            "🛡️ Ajustar stop-losses a máximo -5% por posición",
            "❌ Suspender nuevas posiciones de riesgo"
        ]
        actions['tactical'] = [
            "📉 Considerar hedges: VIX calls, put options en índices",
            "🏦 Evitar exposición a bancos regionales y NBFI vulnerables",
            "💎 Overweight quality: mega-caps con balance sheets sólidos",
            "🇺🇸 Flight-to-safety: aumentar Treasuries (short duration)"
        ]
        actions['strategic'] = [
            "🔄 Revisar contraparte risk: diversificar entre brokers/custodios",
            "📊 Preparar lista de oportunidades para fase de capitulación",
            "🎯 Definir niveles de re-entry (ej: stress_index < 50)"
        ]

    elif level_raw == "MEDIO":
        actions['immediate'] = [
            "🟡 Reducir leverage a máximo 1.3x",
            "👀 Monitorear posiciones en sectores cíclicos",
            "📋 Revisar lista de stop-losses"
        ]
        actions['tactical'] = [
            "⚖️ Rebalancear hacia 60% equity / 30% bonds / 10% cash",
            "🎯 Evitar sectores con alta exposición a funding markets",
            "✅ Mantener quality bias: avoid high-beta, high-leverage names"
        ]
        actions['strategic'] = [
            "📈 Mantener plan de trading normal pero con stops más ajustados",
            "🔍 Intensificar monitoreo diario de stress indicators",
            "💼 Preparar plan de contingencia si índice supera 66"
        ]

    else:  # BAJO
        actions['immediate'] = [
            "✅ Posicionamiento normal apropiado",
            "🚀 Leverage moderado (hasta 1.5x) es aceptable"
        ]
        actions['tactical'] = [
            "📈 Buscar oportunidades en breakouts técnicos",
            "🎯 Target allocation: 70-80% equity",
            "💡 Considerar posiciones en beta alto si setup técnico confirma"
        ]
        actions['strategic'] = [
            "📊 Mantener vigilancia de stress indicators (chequeo diario)",
            "🔄 Diversificar entre estrategias (value, growth, momentum)",
            "🌐 Explorar oportunidades en sectores cíclicos"
        ]

    # Ajustes por nodos vulnerables
    if n_vulnerable_nodes >= 2:
        actions['tactical'].append(
            f"⚠️ {n_vulnerable_nodes} instituciones vulnerables detectadas: "
            "evitar exposición directa a estas contrapartes"
        )

    # Ajustes por resiliencia muy baja
    if network_resilience < 30:
        actions['immediate'].append(
            "🚨 Resiliencia de red crítica: priorizar liquidez sobre retorno. "
            "Asegurar capacidad de salida rápida."
        )

    return actions
