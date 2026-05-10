"""
wfa_sizing.py — Position Sizing Engine per WFA-Turbo-Pro
=========================================================
Modulo puro (nessuna dipendenza da Streamlit o wfa_core).
Calcola il numero di contratti da allocare per ogni trade OOS
in base a:
  - Capitale disponibile (fisso o con compounding)
  - Cap % del capitale per strategia
  - Stop-loss per contratto (USD o % del premio)
  - Peso WFA della strategia (full / bench)
  - RiskFactor dell'Equity Control (se attivo)

Regola bench + EC
-----------------
Una strategia viene esclusa dall'allocazione contratti (Contracts = 0)
se è in "panchina" (weight_wfa <= BENCH_WEIGHT_THRESHOLD) E il suo
RiskFactor EC è <= EC_RF_BENCH_CUTOFF.
In tutti gli altri casi il sizing si applica normalmente.

Budget giornaliero
------------------
Ogni giorno viene inizializzato con un budget massimo di rischio
pari a max_daily_loss_pct% del capitale corrente. I contratti di ogni
trade vengono clampati in modo che il rischio complessivo della giornata
non superi il budget residuo. Il budget si azzera a mezzanotte (data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ── Costanti di default ────────────────────────────────────────────────────

DEFAULT_SIZING_ENABLED        = False
DEFAULT_INITIAL_CAPITAL       = 100_000.0   # USD
DEFAULT_MAX_DAILY_LOSS_PCT    = 2.0         # % del capitale
DEFAULT_COMPOUNDING           = True
DEFAULT_CAP_PCT               = 5.0         # % capitale max per strategia
DEFAULT_SL_USD                = 200.0       # stop-loss per contratto (USD)
DEFAULT_SL_PCT                = 0.0         # stop-loss come % del premio (0 = disabilitato)

# Soglie regola bench + EC
BENCH_WEIGHT_THRESHOLD        = 0.79        # weight_wfa <= questa soglia = panchina
EC_RF_BENCH_CUTOFF            = 0.50        # RF <= questa soglia + panchina = skip


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class StrategySizingConfig:
    """Parametri di sizing specifici per una singola strategia."""
    cap_pct: float = DEFAULT_CAP_PCT      # % capitale max da allocare
    sl_usd:  float = DEFAULT_SL_USD       # stop-loss per contratto in USD
    sl_pct:  float = DEFAULT_SL_PCT       # stop-loss come % del premio (0 = usa sl_usd)

    def __post_init__(self):
        self.cap_pct = max(0.1, min(100.0, float(self.cap_pct)))
        self.sl_usd  = max(1.0, float(self.sl_usd))
        self.sl_pct  = max(0.0, float(self.sl_pct))


@dataclass
class PortfolioSizingParams:
    """Parametri globali di sizing del portafoglio."""
    initial_capital:       float = DEFAULT_INITIAL_CAPITAL
    max_daily_loss_pct:    float = DEFAULT_MAX_DAILY_LOSS_PCT
    compounding:           bool  = DEFAULT_COMPOUNDING

    def __post_init__(self):
        self.initial_capital    = max(1_000.0, float(self.initial_capital))
        self.max_daily_loss_pct = max(0.1, min(20.0, float(self.max_daily_loss_pct)))


# ── Funzioni di utilità ────────────────────────────────────────────────────

def clean_money(val) -> float:
    """Converte un valore monetario (es. '$1,234.56') in float."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).replace("$", "").replace(",", "").strip())


def should_skip_bench(weight_wfa: float, risk_factor: float) -> bool:
    """
    True se la strategia deve essere esclusa dall'allocazione contratti.

    Una strategia in panchina (weight_wfa <= BENCH_WEIGHT_THRESHOLD) con
    RiskFactor EC basso (risk_factor <= EC_RF_BENCH_CUTOFF) non riceve
    contratti perché il segnale composto di debolezza (basso peso + alto DD)
    rende il trade poco efficiente rispetto al rischio impegnato.
    """
    in_bench  = weight_wfa <= BENCH_WEIGHT_THRESHOLD
    low_rf    = risk_factor <= EC_RF_BENCH_CUTOFF
    return in_bench and low_rf


def _sl_per_contract(
    sizing_cfg: StrategySizingConfig,
    premium: float,
) -> float:
    """
    Calcola lo stop-loss unitario per contratto.

    Priorità:
    1. sl_pct > 0  →  sl_unit = premium * sl_pct / 100
    2. sl_usd > 0  →  sl_unit = sl_usd
    """
    if sizing_cfg.sl_pct > 0.0 and premium > 0.0:
        return premium * sizing_cfg.sl_pct / 100.0
    return max(1.0, sizing_cfg.sl_usd)


def compute_contracts(
    premium: float,
    sizing_cfg: StrategySizingConfig,
    capital: float,
    risk_factor: float,
    weight_wfa: float,
    remaining_daily_budget: float,
) -> tuple[int, float]:
    """
    Calcola il numero di contratti da allocare per un singolo trade.

    Formula
    -------
    max_risk      = capital × cap_pct%              (rischio massimo per la strategia)
    sl_unit       = stop-loss per contratto (USD)   (da _sl_per_contract)
    n_raw         = floor(max_risk / sl_unit × weight_wfa × risk_factor)
    n_budget_cap  = floor(remaining_daily_budget / sl_unit)
    contracts     = min(n_raw, n_budget_cap)        (clamp sul budget giornaliero)
    risk_used     = contracts × sl_unit             (rischio effettivo impegnato)

    Parametri
    ---------
    premium               : premio del trade (per il calcolo sl_pct)
    sizing_cfg            : StrategySizingConfig della strategia
    capital               : capitale corrente del portafoglio
    risk_factor           : RiskFactor EC (0.0 / 0.25 / 0.5 / 1.0)
    weight_wfa            : peso WFA del trade (es. 0.80 / 0.40)
    remaining_daily_budget: budget di rischio giornaliero ancora disponibile

    Ritorna
    -------
    (contracts: int, risk_used: float)
    """
    # Se skip bench+EC → nessun contratto
    if should_skip_bench(weight_wfa, risk_factor):
        return 0, 0.0

    # Se RF = 0 (EC stop totale) → nessun contratto
    if risk_factor == 0.0:
        return 0, 0.0

    sl_unit = _sl_per_contract(sizing_cfg, abs(premium))
    if sl_unit <= 0.0:
        return 0, 0.0

    max_risk = capital * (sizing_cfg.cap_pct / 100.0)
    n_raw    = math.floor(max_risk / sl_unit * weight_wfa * risk_factor)
    n_raw    = max(0, n_raw)

    # Clamp sul budget giornaliero residuo
    if remaining_daily_budget <= 0.0:
        return 0, 0.0
    n_budget_cap = math.floor(remaining_daily_budget / sl_unit)
    contracts    = min(n_raw, n_budget_cap)
    risk_used    = contracts * sl_unit
    return contracts, risk_used
