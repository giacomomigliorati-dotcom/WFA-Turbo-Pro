# ================================================================
#  MODULO OTTIMIZZATORE WFA — WFA-Turbo-Pro
#  File: wfa_optimizer_module.py
#  Versione: 2.0.1 — 2026-05-07
#  Changelog v2.0.1:
#    - Fix: UI robustezza duplicata al rerun di Streamlit.
#      La progress bar veniva mantenuta visibile dopo il calcolo
#      perché mancava prog.empty() al termine del loop.
#      La cache di robustezza ora usa una firma completa delle
#      top20 (Rank, parametri WFA) invece del solo len(top20),
#      così il ricalcolo viene triggerato correttamente quando
#      cambiano i risultati dell'ottimizzazione.
#      Chiave cache rinominata da wfa_robustness_n_rows a
#      wfa_robustness_top20_signature. Tutte le invalidazioni
#      (import JSON, run nuova) aggiornate di conseguenza.
#
#  Changelog v2.0.0:
#    - run_wfa_single_windowed: aggiunto campo "selected_strategies"
#      (List[str]) al dict di ogni finestra OOS — contiene le chiavi
#      del dict weights calcolato da _select_strategies.
#    - render_robustness_section: dopo il box plot PF e l'expander
#      dettaglio finestre, aggiunti due nuovi grafici per la combo
#      selezionata nella selectbox:
#        1. Bar chart verticale PnL per finestra OOS (verde se >0,
#           rosso se <=0), con linea di riferimento a y=0 e hover
#           completo (finestra, date, PnL, PF).
#        2. Bar chart orizzontale frequenza strategie: quante finestre
#           OOS ha coperto ogni strategia, con percentuale sul bar e
#           colore attenuato per frequenze < 50%.
#    - selected_strategies è già incluso in windows_detail →
#      va nel JSON esportato automaticamente senza modifiche a
#      _export_run_json.
#
#  Changelog v1.9.0:
#    - Import aggiuntivi: hashlib, json, datetime/timezone.
#    - Aggiunto helper _dataset_hash(df): hash MD5 riproducibile
#      sulle prime 500 righe del dataset.
#    - Aggiunto helper _export_run_json(): serializza i risultati
#      di una run in JSON scaricabile (version, exported_at,
#      target_metric, n_combos, dataset_hash, results).
#    - Aggiunto helper _import_run_json(): deserializza un file JSON
#      di run, con warning se l'hash del dataset non corrisponde.
#    - render_optimizer_tab: calcolo current_hash dopo il check
#      df_processed is None.
#    - render_optimizer_tab: expander "Importa risultati da run
#      precedente" aggiunto prima della griglia di ottimizzazione.
#    - render_optimizer_tab: bottone download JSON (reimportabile)
#      aggiunto accanto al CSV esistente.
#
#  Changelog v1.8.0:
#    - Aggiunto helper _time_under_water(arr): calcola il numero
#      massimo di giorni consecutivi sotto il picco equity per
#      ogni finestra OOS (intra-finestra).
#    - run_wfa_single_windowed: aggiunto campo "tuw" per finestra.
#    - compute_robustness_profile: aggiunto tuw_per_window e
#      tuw_median nel profilo restituito.
#    - render_robustness_section: aggiunta selectbox "Dimensione
#      cerchi" con 9 opzioni selezionabili dall'utente:
#        % finestre PF>1, N° finestre OOS, PF Mediana,
#        Max DD OOS ($), Total PnL OOS, Time Under Water (mediana gg),
#        Sharpe OOS, Sortino OOS, Calmar OOS.
#      Per metriche "meno è meglio" (Max DD, TUW) la dimensione
#      viene invertita: cerchi grandi = valori più bassi.
#
#  Changelog v1.7.1:
#    - Slider soglia Max Drawdown: max_value esteso da 2000 a 10000.
#    - Aggiunta spiegazione dettagliata nel tooltip (?) dello slider.
#
#  Changelog v1.7:
#    - FIX CRITICO: rolling OOS step corretto in run_wfa_single
#      e run_wfa_single_windowed.
#
#  Questo modulo fornisce:
#  - run_wfa_single(...)             : motore WFA aggregato (grid search)
#  - run_wfa_single_windowed(...)    : metriche per singola finestra OOS
#  - compute_robustness_profile(...) : statistiche robustezza da finestre
#  - render_optimizer_tab(...)       : UI Streamlit "⚙️ Ottimizzatore"
#  - render_robustness_section(...)  : scatter XY + box plot PF per finestra
# ================================================================

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

from wfa_core import (
    RANKING_METRICS,
    DEFAULT_RANKING_METRIC,
    compute_ranking_score,
    passes_roc_filter,
    compute_cusum_banned,
)


# ─────────────────────────────────────────────────────────────────
# DATA CLASS PARAMETRI
# ─────────────────────────────────────────────────────────────────

@dataclass
class WFAParams:
    top_n: int
    full_weight_count: int
    full_weight_pct: float
    bench_weight_pct: float
    max_per_group: int
    ranking_metric: str
    roc_filter_enabled: bool
    roc_filter_steps: int
    is_months: int = 12
    oos_days: int = 28


# ─────────────────────────────────────────────────────────────────
# HELPERS INTERNI
# ─────────────────────────────────────────────────────────────────

def _max_drawdown(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    mdd = float(np.min(dd))
    return abs(mdd) if mdd < 0 else 0.0


def _time_under_water(arr: np.ndarray) -> int:
    """Numero massimo di giorni consecutivi con equity sotto il picco precedente."""
    if arr.size == 0:
        return 0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    underwater = (cum < peak).astype(int)
    max_tuw = 0
    current = 0
    for v in underwater:
        if v:
            current += 1
            max_tuw = max(max_tuw, current)
        else:
            current = 0
    return max_tuw


def _profit_factor(pnl_list: List[float]) -> float:
    arr = np.asarray(pnl_list, dtype=float)
    wins = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    if losses < 1e-9:
        return wins if wins > 0 else 0.0
    return round(float(wins / losses), 2)


def _dataset_hash(df: pd.DataFrame) -> str:
    """Hash MD5 riproducibile sulle prime 500 righe del dataset."""
    sample = df.head(500).to_csv(index=False).encode("utf-8")
    return hashlib.md5(sample).hexdigest()


def _export_run_json(
    df_res: pd.DataFrame,
    target_metric: str,
    df_processed: Optional[pd.DataFrame] = None,
) -> bytes:
    """Serializza i risultati di una run in JSON scaricabile."""
    payload = {
        "version":       "1.0",
        "exported_at":   datetime.now(timezone.utc).isoformat(),
        "target_metric": target_metric,
        "n_combos":      len(df_res),
        "dataset_hash":  _dataset_hash(df_processed) if df_processed is not None else None,
        "results":       df_res.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _import_run_json(
    raw: bytes,
    df_processed: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Deserializza un file JSON di run.
    Restituisce (df_res, target_metric, warning_msg).
    warning_msg è None se tutto ok, stringa se hash non corrisponde.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return None, None, f"❌ File non valido: {e}"

    required = {"version", "target_metric", "results"}
    if not required.issubset(payload.keys()):
        return None, None, "❌ File JSON non riconosciuto (chiavi mancanti)."

    try:
        df_res = pd.DataFrame(payload["results"])
        if "Rank" not in df_res.columns:
            df_res.insert(0, "Rank", df_res.index + 1)
    except Exception as e:
        return None, None, f"❌ Errore nel deserializzare i risultati: {e}"

    target_metric = str(payload.get("target_metric", "calmar"))
    warn = None

    if df_processed is not None and payload.get("dataset_hash"):
        current_hash = _dataset_hash(df_processed)
        if current_hash != payload["dataset_hash"]:
            exported_at = payload.get("exported_at", "data sconosciuta")
            warn = (
                f"⚠️ Il dataset attuale **non corrisponde** a quello usato per questa run "
                f"(esportata il {exported_at[:10]}). "
                "I profili di robustezza verranno ricalcolati sul dataset corrente, "
                "ma i valori di Sharpe/Calmar/PnL potrebbero non essere replicabili."
            )

    return df_res, target_metric, warn


def _select_strategies(
    is_data: pd.DataFrame,
    params: WFAParams,
    group_map: Dict[str, str],
    strategy_col: str,
    date_col: str,
    pnl_col: str,
) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
    """Logica IS: ranking + CUSUM + selezione + pesi assoluti."""
    metrics_rows = []
    for strat, grp in is_data.groupby(strategy_col):
        if len(grp) < 5:
            continue

        strat_is = grp.rename(columns={date_col: "Date Closed", pnl_col: "Net PnL"})

        if params.roc_filter_enabled and not passes_roc_filter(strat_is, params.roc_filter_steps):
            continue

        score, higher_is_better = compute_ranking_score(
            strat_is, params.ranking_metric, params.roc_filter_steps
        )

        banned_days: List[int] = []
        if "weekday_open" in strat_is.columns:
            banned_days = compute_cusum_banned(strat_is)
            active_days = strat_is["weekday_open"].unique().tolist()
            remaining = [d for d in active_days if d not in banned_days]
            if not remaining:
                continue

        metrics_rows.append({
            "strategy": strat,
            "score": score,
            "family": group_map.get(strat, "Unknown"),
            "total_pnl_is": float(strat_is["Net PnL"].sum()),
            "higher_is_better": higher_is_better,
            "banned_days": banned_days,
        })

    if not metrics_rows:
        return {}, {}

    metrics_df = pd.DataFrame(metrics_rows)
    asc = not bool(metrics_df["higher_is_better"].all())
    metrics_df = metrics_df.sort_values(
        ["score", "total_pnl_is"], ascending=[asc, False]
    )

    selected: List[Dict] = []
    family_count: Dict[str, int] = {}
    for _, row in metrics_df.iterrows():
        fam = row["family"]
        if family_count.get(fam, 0) < params.max_per_group:
            selected.append(row.to_dict())
            family_count[fam] = family_count.get(fam, 0) + 1
        if len(selected) >= params.top_n:
            break

    if not selected:
        return {}, {}

    n_full = min(params.full_weight_count, len(selected))

    weights: Dict[str, float] = {}
    banned_per_strat: Dict[str, List[int]] = {}
    for idx, row in enumerate(selected):
        w = params.full_weight_pct if idx < n_full else params.bench_weight_pct
        strat_name = str(row["strategy"])
        weights[strat_name] = w / 100.0
        banned_per_strat[strat_name] = row.get("banned_days", [])

    return weights, banned_per_strat


# ─────────────────────────────────────────────────────────────────
# CORE: singolo loop WFA — aggregato (grid search veloce)
# ─────────────────────────────────────────────────────────────────

def run_wfa_single(
    df: pd.DataFrame,
    params: WFAParams,
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
) -> Optional[Dict[str, float]]:
    """Esegue un singolo loop Walk-Forward rolling e restituisce metriche OOS aggregate."""
    if df is None or df.empty:
        return None
    if any(c not in df.columns for c in (date_col, strategy_col, pnl_col)):
        return None

    df = df[[date_col, strategy_col, pnl_col] + (
        ["weekday_open"] if "weekday_open" in df.columns else []
    )].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    has_weekday = "weekday_open" in df.columns
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    all_oos_pnl: List[float] = []

    oos_step = pd.Timedelta(days=params.oos_days)
    start_oos = min_date + relativedelta(months=params.is_months)

    while True:
        start_is = start_oos - relativedelta(months=params.is_months)
        end_is = start_oos - pd.Timedelta(days=1)
        end_oos = start_oos + pd.Timedelta(days=params.oos_days - 1)

        if start_oos > max_date:
            break

        is_data = df[(df[date_col] >= start_is) & (df[date_col] <= end_is)]
        oos_data = df[(df[date_col] >= start_oos) & (df[date_col] <= end_oos)]

        if is_data.empty or oos_data.empty:
            start_oos += oos_step
            continue

        weights, banned_per_strat = _select_strategies(
            is_data, params, group_map, strategy_col, date_col, pnl_col
        )
        if not weights:
            start_oos += oos_step
            continue

        oos_sub = oos_data[oos_data[strategy_col].isin(weights.keys())].copy()
        if oos_sub.empty:
            start_oos += oos_step
            continue

        if has_weekday:
            mask = oos_sub.apply(
                lambda r: r["weekday_open"] in banned_per_strat.get(r[strategy_col], []),
                axis=1,
            )
            oos_sub = oos_sub[~mask]

        if oos_sub.empty:
            start_oos += oos_step
            continue

        oos_daily = (
            oos_sub.groupby([date_col, strategy_col])[pnl_col]
            .sum().reset_index()
        )
        oos_daily["weighted_pnl"] = oos_daily.apply(
            lambda r: r[pnl_col] * weights.get(r[strategy_col], 0.0), axis=1
        )
        daily_agg = oos_daily.groupby(date_col)["weighted_pnl"].sum()
        all_oos_pnl.extend(daily_agg.to_list())

        start_oos += oos_step

    if len(all_oos_pnl) < 5:
        return None

    pnl = np.asarray(all_oos_pnl, dtype=float)
    mean_p = float(np.mean(pnl))
    std_p = float(np.std(pnl))
    sharpe = (mean_p / (std_p or 1e-9)) * math.sqrt(252)

    neg = pnl[pnl < 0]
    dstd = float(np.std(neg)) if neg.size > 1 else 0.0
    sortino = (mean_p / (dstd or 1e-9)) * math.sqrt(252)

    total_pnl = float(np.sum(pnl))
    max_dd = _max_drawdown(pnl)
    calmar = (total_pnl / (max_dd or 1e-9)) if max_dd > 0 else 0.0
    recovery = (total_pnl / (max_dd or 1e-9)) if max_dd > 0 else 0.0

    return {
        "sharpe":          round(sharpe, 2),
        "sortino":         round(sortino, 2),
        "calmar":          round(calmar, 2),
        "total_pnl":       round(total_pnl, 2),
        "max_dd":          round(max_dd, 2),
        "recovery_factor": round(recovery, 2),
        "n_oos_days":      int(pnl.size),
    }


# ─────────────────────────────────────────────────────────────────
# CORE: singolo loop WFA — per finestra OOS (analisi robustezza)
# ─────────────────────────────────────────────────────────────────

def run_wfa_single_windowed(
    df: pd.DataFrame,
    params: WFAParams,
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
) -> List[Dict]:
    """Variante di run_wfa_single che restituisce metriche per ogni finestra OOS separata."""
    if df is None or df.empty:
        return []
    if any(c not in df.columns for c in (date_col, strategy_col, pnl_col)):
        return []

    df = df[[date_col, strategy_col, pnl_col] + (
        ["weekday_open"] if "weekday_open" in df.columns else []
    )].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    has_weekday = "weekday_open" in df.columns
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    windows: List[Dict] = []
    window_idx = 0

    oos_step = pd.Timedelta(days=params.oos_days)
    start_oos = min_date + relativedelta(months=params.is_months)

    while True:
        start_is = start_oos - relativedelta(months=params.is_months)
        end_is = start_oos - pd.Timedelta(days=1)
        end_oos = start_oos + pd.Timedelta(days=params.oos_days - 1)

        if start_oos > max_date:
            break

        is_data = df[(df[date_col] >= start_is) & (df[date_col] <= end_is)]
        oos_data = df[(df[date_col] >= start_oos) & (df[date_col] <= end_oos)]

        if is_data.empty or oos_data.empty:
            start_oos += oos_step
            continue

        weights, banned_per_strat = _select_strategies(
            is_data, params, group_map, strategy_col, date_col, pnl_col
        )
        if not weights:
            start_oos += oos_step
            continue

        oos_sub = oos_data[oos_data[strategy_col].isin(weights.keys())].copy()
        if oos_sub.empty:
            start_oos += oos_step
            continue

        if has_weekday:
            mask = oos_sub.apply(
                lambda r: r["weekday_open"] in banned_per_strat.get(r[strategy_col], []),
                axis=1,
            )
            oos_sub = oos_sub[~mask]

        if oos_sub.empty:
            start_oos += oos_step
            continue

        oos_sub = oos_sub.copy()
        oos_sub["weighted_pnl"] = oos_sub.apply(
            lambda r: r[pnl_col] * weights.get(r[strategy_col], 0.0), axis=1
        )
        trade_pnls = oos_sub["weighted_pnl"].tolist()

        oos_sub_daily = oos_sub.groupby(date_col)["weighted_pnl"].sum()
        daily_arr = np.asarray(oos_sub_daily.tolist(), dtype=float)

        window_idx += 1
        windows.append({
            "window":              window_idx,
            "start_oos":           str(start_oos.date()),
            "end_oos":             str(end_oos.date()),
            "pf":                  _profit_factor(trade_pnls),
            "max_dd":              round(_max_drawdown(daily_arr), 2),
            "tuw":                 _time_under_water(daily_arr),
            "n_trade":             len(trade_pnls),
            "total_pnl":           round(float(sum(trade_pnls)), 2),
            "selected_strategies": list(weights.keys()),
        })

        start_oos += oos_step

    return windows


# ─────────────────────────────────────────────────────────────────
# COMPUTE: profilo di robustezza da lista finestre OOS
# ─────────────────────────────────────────────────────────────────

def compute_robustness_profile(
    windows: List[Dict],
    dd_threshold: float = 0.15,
) -> Optional[Dict]:
    if not windows or len(windows) < 3:
        return None

    pf_vals = [w["pf"] for w in windows if w["n_trade"] >= 3]
    if len(pf_vals) < 3:
        return None

    pf_arr = np.asarray(pf_vals, dtype=float)
    dd_vals = [w["max_dd"] for w in windows]
    tuw_vals = [w["tuw"] for w in windows]

    return {
        "pf_median":           round(float(np.median(pf_arr)), 2),
        "pf_p25":              round(float(np.percentile(pf_arr, 25)), 2),
        "pf_pct_above_1":      round(float(np.mean(pf_arr > 1.0)), 2),
        "dd_pct_below_thresh": round(float(np.mean(np.asarray(dd_vals) < dd_threshold)), 2),
        "tuw_median":          int(round(float(np.median(tuw_vals)))),
        "n_windows":           len(windows),
        "n_windows_valid":     len(pf_vals),
        "pf_per_window":       [w["pf"] for w in windows],
        "dd_per_window":       dd_vals,
        "tuw_per_window":      tuw_vals,
        "windows_detail":      windows,
    }


# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE OPZIONI DIMENSIONE CERCHI SCATTER
# ─────────────────────────────────────────────────────────────────

# Ogni entry: (label UI, chiave nel dict scatter_row, fonte, invert)
# invert=True significa "meno è meglio" → dimensione invertita
_SIZE_OPTIONS: List[Tuple[str, str, str, bool]] = [
    ("% finestre PF > 1",          "pf_pct",    "profile", False),
    ("N° finestre OOS",             "n_windows", "profile", False),
    ("PF Mediana",                  "pf_median", "profile", False),
    ("Max DD OOS ($) — inv.",       "max_dd",    "df_res",  True),
    ("Total PnL OOS ($)",           "total_pnl", "df_res",  False),
    ("Time Under Water (mediana gg) — inv.", "tuw_median", "profile", True),
    ("Sharpe OOS",                  "sharpe",    "df_res",  False),
    ("Sortino OOS",                 "sortino",   "df_res",  False),
    ("Calmar OOS",                  "calmar",    "df_res",  False),
]
_SIZE_LABELS = [o[0] for o in _SIZE_OPTIONS]
_SIZE_DEFAULT = "% finestre PF > 1"


def _compute_size_vals(
    size_key: str,
    size_invert: bool,
    valid_scatter: List[Dict],
    min_px: float = 8.0,
    max_px: float = 40.0,
) -> List[float]:
    """Normalizza la variabile scelta in pixel [min_px, max_px] per la dimensione cerchi."""
    raw = [r[size_key] if r.get(size_key) is not None else 0.0 for r in valid_scatter]
    arr = np.asarray(raw, dtype=float)
    if size_invert:
        arr = -arr  # inverti: valori più bassi → cerchi più grandi
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return [max_px * 0.5] * len(arr)
    normalized = (arr - vmin) / (vmax - vmin)
    return [float(min_px + v * (max_px - min_px)) for v in normalized]


# ─────────────────────────────────────────────────────────────────
# UI: sezione robustezza (scatter XY + box plot on-demand)
# ─────────────────────────────────────────────────────────────────

def render_robustness_section(
    df_res: pd.DataFrame,
    df_processed: pd.DataFrame,
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
    target_metric: str = "calmar",
) -> None:
    st.markdown("---")
    st.markdown("### 🔬 Analisi di Robustezza — Top 20")
    st.caption(
        "Verifica la consistenza per finestra OOS: le combinazioni migliori "
        "sono quelle in alto a destra nello scatter (alta metrica target + alto PF P25)."
    )

    if df_processed is None or df_processed.empty or df_res.empty:
        st.warning("⚠️ Dati non disponibili per l'analisi di robustezza.")
        return

    _DD_SLIDER_HELP = (
        "Soglia di drawdown massimo accettabile per singola finestra OOS, espressa in dollari ($).\n\n"
        "**Come viene calcolato il Max DD**\n"
        "Il drawdown è calcolato in termini assoluti (monetari, non percentuali): "
        "è la caduta massima picco-a-valle dell'equity cumulata ponderata all'interno "
        "di ogni singola finestra OOS (tipicamente 28 giorni). "
        "Ogni finestra riparte da zero — quindi si tratta di un drawdown intra-finestra, "
        "non calcolato sull'intero storico.\n\n"
        "**A cosa serve questa soglia**\n"
        "Non filtra né elimina combinazioni: cambia il colore dei punti nello scatter. "
        "Per ogni combinazione nel Top 20, viene calcolata la percentuale di finestre OOS "
        "in cui il max_dd è rimasto sotto questa soglia. "
        "Quella percentuale determina il colore del punto (rosso = poche finestre sicure, "
        "verde = la maggior parte delle finestre rimane sotto soglia).\n\n"
        "**Perché il colore è in percentuale e non in $**\n"
        "Lo scatter vuole mostrare la consistenza del rischio nel tempo, non il valore "
        "puntuale di un singolo drawdown. Alzare la soglia rende più combinazioni 'sicure' "
        "(più verde); abbassarla è più selettivo e solo i portafogli con drawdown molto "
        "contenuti per finestra appaiono verdi."
    )

    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        dd_threshold = st.slider(
            "📉 Soglia Max Drawdown per colore scatter ($)",
            min_value=50, max_value=10000, value=150, step=50,
            help=_DD_SLIDER_HELP,
            key="robustness_dd_threshold",
        )
    with ctrl_col2:
        size_label = st.selectbox(
            "⚪ Dimensione cerchi",
            options=_SIZE_LABELS,
            index=_SIZE_LABELS.index(_SIZE_DEFAULT),
            key="robustness_size_metric",
            help=(
                "Scegli la variabile da codificare nella dimensione dei cerchi nello scatter.\n"
                "Le opzioni con '— inv.' usano scala invertita: cerchi grandi = valori più bassi "
                "(es. Max DD basso o TUW breve sono preferibili)."
            ),
        )
    dd_thresh_val = float(dd_threshold)
    size_cfg = next(o for o in _SIZE_OPTIONS if o[0] == size_label)
    size_key, size_src, size_invert = size_cfg[1], size_cfg[2], size_cfg[3]

    cache_key = "wfa_robustness_cache"
    top20 = df_res.head(20).copy()
    top20_signature = tuple(
        (
            int(row["Rank"]),
            int(row["top_n"]),
            int(row["full_weight_count"]),
            float(row["full_weight_pct"]),
            float(row["bench_weight_pct"]),
            int(row["max_per_group"]),
            str(row["ranking_metric"]),
            bool(row["roc_filter"]),
            int(row["roc_steps"]),
        )
        for _, row in top20.iterrows()
    )

    need_recompute = (
        cache_key not in st.session_state
        or st.session_state.get("wfa_robustness_top20_signature") != top20_signature
    )

    if need_recompute:
        profiles: List[Optional[Dict]] = []
        prog = st.progress(0.0, text="Analisi robustezza: finestre OOS in corso…")

        for i, (_, row) in enumerate(top20.iterrows()):
            prog.progress((i + 1) / len(top20), text=f"Robustezza combo {i+1}/{len(top20)}…")
            p = WFAParams(
                top_n=int(row["top_n"]),
                full_weight_count=int(row["full_weight_count"]),
                full_weight_pct=float(row["full_weight_pct"]),
                bench_weight_pct=float(row["bench_weight_pct"]),
                max_per_group=int(row["max_per_group"]),
                ranking_metric=str(row["ranking_metric"]),
                roc_filter_enabled=bool(row["roc_filter"]),
                roc_filter_steps=int(row["roc_steps"]),
            )
            windows = run_wfa_single_windowed(
                df=df_processed, params=p, group_map=group_map,
                date_col=date_col, strategy_col=strategy_col, pnl_col=pnl_col,
            )
            profiles.append(compute_robustness_profile(windows, dd_threshold=dd_thresh_val))

        prog.progress(1.0, text="✅ Analisi robustezza completata")
        prog.empty()
        st.session_state[cache_key] = profiles
        st.session_state["wfa_robustness_top20_signature"] = top20_signature
    else:
        profiles = st.session_state[cache_key]

    scatter_rows = []
    for i, (_, row) in enumerate(top20.iterrows()):
        prof = profiles[i] if i < len(profiles) else None
        pf_p25 = prof["pf_p25"] if prof else None
        pf_pct = prof["pf_pct_above_1"] if prof else 0.0
        dd_pct = prof["dd_pct_below_thresh"] if prof else 0.0

        if prof and "dd_per_window" in prof:
            dd_arr = np.asarray(prof["dd_per_window"], dtype=float)
            dd_pct = float(np.mean(dd_arr < dd_thresh_val))

        scatter_rows.append({
            "rank":        int(row["Rank"]),
            "target_val":  float(row.get(target_metric, 0.0)),
            "pf_p25":      pf_p25,
            "pf_pct":      round(pf_pct * 100, 1),
            "dd_pct":      round(dd_pct * 100, 1),
            # metriche da df_res
            "max_dd":      float(row.get("max_dd", 0.0)),
            "total_pnl":   float(row.get("total_pnl", 0.0)),
            "sharpe":      float(row.get("sharpe", 0.0)),
            "sortino":     float(row.get("sortino", 0.0)),
            "calmar":      float(row.get("calmar", 0.0)),
            # metriche da profilo robustezza
            "n_windows":   prof["n_windows"] if prof else 0,
            "pf_median":   prof["pf_median"] if prof else 0.0,
            "tuw_median":  prof["tuw_median"] if prof else 0,
            "label":       (
                f"Rank {int(row['Rank'])} | "
                f"top_n={int(row['top_n'])} | "
                f"{row['ranking_metric']} | "
                f"mpg={int(row['max_per_group'])}"
            ),
            "has_profile": prof is not None,
        })

    valid = [r for r in scatter_rows if r["has_profile"] and r["pf_p25"] is not None]
    invalid = [r for r in scatter_rows if not r["has_profile"] or r["pf_p25"] is None]

    if not valid:
        st.warning("Nessuna combinazione ha prodotto finestre OOS sufficienti per l'analisi.")
        return

    size_vals = _compute_size_vals(size_key, size_invert, valid)

    st.markdown(f"#### 📊 Scatter: {target_metric.capitalize()} vs PF P25 per finestra OOS")
    st.caption(
        f"**Asse X** = metrica target aggregata · "
        f"**Asse Y** = primo quartile del Profit Factor per finestra OOS · "
        f"**Dimensione** = {size_label} · "
        f"**Colore** = % finestre con Max DD < soglia (verde = più sicuro)"
    )

    target_vals = [r["target_val"] for r in valid]
    pf_p25_vals = [r["pf_p25"] for r in valid]
    color_vals  = [r["dd_pct"] for r in valid]
    labels      = [r["label"] for r in valid]
    text_labels = [str(r["rank"]) if r["rank"] <= 3 else "" for r in valid]
    size_raw    = [r.get(size_key, 0.0) for r in valid]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=target_vals,
        y=pf_p25_vals,
        mode="markers+text",
        text=text_labels,
        textposition="top center",
        textfont=dict(color="#e0eaf4", size=11),
        marker=dict(
            size=size_vals,
            color=color_vals,
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.4, "#b45309"],
                [0.7, "#065f46"],
                [1.0, "#6ee7b7"],
            ],
            cmin=0, cmax=100,
            colorbar=dict(
                title="% finestre<br>DD < soglia",
                ticksuffix="%",
                thickness=14,
                len=0.7,
            ),
            line=dict(color="#0d1117", width=1),
            opacity=0.92,
        ),
        customdata=list(zip(
            [r["rank"] for r in valid],
            [r["dd_pct"] for r in valid],
            size_raw,
        )),
        hovertemplate=(
            "%{meta}<br>"
            f"<b>{target_metric}:</b> %{{x:.2f}}<br>"
            "<b>PF P25:</b> %{y:.2f}<br>"
            "<b>% finestre DD<soglia:</b> %{customdata[1]:.0f}%<br>"
            f"<b>{size_label}:</b> %{{customdata[2]:.2f}}<br>"
            "<extra></extra>"
        ),
        meta=labels,
        name="Combinazioni",
    ))

    fig_scatter.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="PF = 1.0 (soglia sopravvivenza)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=10),
    )

    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#131920",
        font=dict(color="#e0eaf4"),
        xaxis=dict(
            title=f"{target_metric.capitalize()} OOS (aggregato)",
            gridcolor="#1e2a35",
            zeroline=False,
        ),
        yaxis=dict(
            title="Profit Factor P25 per finestra OOS",
            gridcolor="#1e2a35",
            zeroline=False,
        ),
        height=480,
        margin=dict(l=60, r=80, t=40, b=60),
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if invalid:
        st.caption(f"⚠️ {len(invalid)} combinazioni escluse dallo scatter per dati OOS insufficienti.")

    st.markdown("---")
    st.markdown("#### 🔎 Dettaglio combinazione — Box Plot PF per finestra OOS")

    combo_options = {}
    for i, (_, row) in enumerate(top20.iterrows()):
        lbl = (
            f"Rank {int(row['Rank'])} | "
            f"top_n={int(row['top_n'])} · "
            f"fwc={int(row['full_weight_count'])} · "
            f"fwp={int(row['full_weight_pct'])}% · "
            f"{row['ranking_metric']} · "
            f"mpg={int(row['max_per_group'])} · "
            f"roc={'\u2713' if row['roc_filter'] else '\u2717'}"
        )
        combo_options[lbl] = i

    selected_label = st.selectbox(
        "Seleziona combinazione da esaminare",
        options=list(combo_options.keys()),
        key="robustness_combo_select",
    )
    sel_idx = combo_options[selected_label]
    prof = profiles[sel_idx] if sel_idx < len(profiles) else None

    if prof is None:
        st.warning("Dati insufficienti per questa combinazione (finestre OOS < 3).")
        return

    pf_vals_sel = prof["pf_per_window"]
    n_win = prof["n_windows"]
    windows_detail = prof["windows_detail"]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("PF Mediana",           f"{prof['pf_median']:.2f}")
    mc2.metric("PF P25",               f"{prof['pf_p25']:.2f}")
    mc3.metric("% finestre PF>1",      f"{prof['pf_pct_above_1']*100:.0f}%")
    mc4.metric("% finestre DD<soglia", f"{prof['dd_pct_below_thresh']*100:.0f}%")
    mc5.metric("TUW mediana (gg)",     f"{prof['tuw_median']}")

    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.25, 0.25, size=len(pf_vals_sel)).tolist()
    point_colors = ["#6ee7b7" if v >= 1.0 else "#f87171" for v in pf_vals_sel]
    hover_texts = [f"Finestra {i+1}<br>PF: {v:.2f}" for i, v in enumerate(pf_vals_sel)]

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=pf_vals_sel,
        x=[0] * len(pf_vals_sel),
        name="PF per finestra",
        marker_color="#4f98a3",
        line_color="#4f98a3",
        fillcolor="rgba(79,152,163,0.25)",
        boxpoints=False,
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_box.add_trace(go.Scatter(
        x=jitter_x,
        y=pf_vals_sel,
        mode="markers",
        marker=dict(
            size=8,
            color=point_colors,
            line=dict(color="#0d1117", width=1),
            opacity=0.9,
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="Punti",
    ))
    fig_box.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="PF = 1.0",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=10),
    )
    fig_box.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#131920",
        font=dict(color="#e0eaf4"),
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False, range=[-1, 1]),
        yaxis=dict(title="Profit Factor", gridcolor="#1e2a35"),
        height=360,
        margin=dict(l=60, r=40, t=40, b=40),
        showlegend=False,
        title=dict(
            text=f"Distribuzione PF — {n_win} finestre OOS",
            font=dict(size=13, color="#e0eaf4"),
            x=0.02,
        ),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📋 Dettaglio finestre OOS (cronologico)", expanded=False):
        detail_df = pd.DataFrame(windows_detail)
        detail_df["semaforo"] = detail_df["pf"].apply(
            lambda v: "🟢" if v >= 1.2 else ("🟡" if v >= 1.0 else "🔴")
        )
        detail_df["pf"]        = detail_df["pf"].map(lambda v: f"{v:.2f}")
        detail_df["max_dd"]    = detail_df["max_dd"].map(lambda v: f"{v:,.2f}")
        detail_df["total_pnl"] = detail_df["total_pnl"].map(lambda v: f"{v:,.2f}")
        show_cols = ["window", "start_oos", "end_oos", "pf", "max_dd", "tuw", "n_trade", "total_pnl", "semaforo"]
        show_cols = [c for c in show_cols if c in detail_df.columns]
        st.dataframe(
            detail_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    # ── Visualizzazione 2: Bar chart PnL per finestra OOS ──────────────────
    st.markdown("#### 📅 PnL per finestra OOS")
    st.caption(
        "Verde = finestra profittevole · Rosso = finestra in perdita · "
        "Linea tratteggiata = break-even · Hover per dettaglio."
    )

    win_labels  = [f"W{w['window']}\n{w['start_oos'][:7]}" for w in windows_detail]
    win_pnl     = [w["total_pnl"] for w in windows_detail]
    win_pf      = [w["pf"] for w in windows_detail]
    win_start   = [w["start_oos"] for w in windows_detail]
    win_end     = [w["end_oos"] for w in windows_detail]
    bar_colors  = ["#6ee7b7" if v > 0 else "#f87171" for v in win_pnl]

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(
        x=win_labels,
        y=win_pnl,
        marker_color=bar_colors,
        marker_line=dict(color="#0d1117", width=0.8),
        customdata=list(zip(win_start, win_end, win_pf)),
        hovertemplate=(
            "<b>Finestra %{x}</b><br>"
            "Dal %{customdata[0]} al %{customdata[1]}<br>"
            "<b>PnL:</b> $%{y:,.2f}<br>"
            "<b>PF:</b> %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        ),
        name="PnL finestra",
        showlegend=False,
    ))
    fig_pnl.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9ca3af",
        line_width=1.2,
    )
    fig_pnl.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#131920",
        font=dict(color="#e0eaf4"),
        xaxis=dict(
            title="Finestra OOS",
            gridcolor="#1e2a35",
            tickangle=-45 if len(win_labels) > 8 else 0,
        ),
        yaxis=dict(
            title="Total PnL ($)",
            gridcolor="#1e2a35",
            tickprefix="$",
            separatethousands=True,
            zeroline=False,
        ),
        height=320,
        margin=dict(l=70, r=40, t=30, b=60),
        bargap=0.25,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Visualizzazione 1: Frequenza strategie nelle finestre OOS ──────────
    all_selected: List[str] = []
    for w in windows_detail:
        all_selected.extend(w.get("selected_strategies", []))

    if all_selected:
        st.markdown("#### 🧩 Frequenza strategie nelle finestre OOS")
        st.caption(
            "Quante finestre OOS ha coperto ogni strategia selezionata in IS. "
            "Barre più chiare = presente in meno del 50% delle finestre."
        )

        freq_counter = Counter(all_selected)
        n_total_windows = len(windows_detail)

        # Ordine discendente per frequenza
        strat_names = [s for s, _ in freq_counter.most_common()]
        strat_counts = [freq_counter[s] for s in strat_names]
        strat_pcts   = [freq_counter[s] / n_total_windows * 100 for s in strat_names]

        # Colore: teal pieno se >= 50%, teal attenuato se < 50%
        bar_colors_strat = [
            "#4f98a3" if (freq_counter[s] / n_total_windows) >= 0.5 else "#2a5560"
            for s in strat_names
        ]

        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=strat_counts,
            y=strat_names,
            orientation="h",
            marker_color=bar_colors_strat,
            marker_line=dict(color="#0d1117", width=0.6),
            text=[f"{p:.0f}%" for p in strat_pcts],
            textposition="outside",
            textfont=dict(color="#e0eaf4", size=11),
            customdata=strat_pcts,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Presente in <b>%{x}</b> finestre su " + str(n_total_windows) + "<br>"
                "Frequenza: <b>%{customdata:.1f}%</b><br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_freq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#131920",
            font=dict(color="#e0eaf4"),
            xaxis=dict(
                title="N° finestre OOS",
                gridcolor="#1e2a35",
                range=[0, n_total_windows * 1.18],  # spazio per le label %
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                gridcolor="#1e2a35",
                automargin=True,
            ),
            height=max(260, len(strat_names) * 28 + 80),
            margin=dict(l=20, r=80, t=30, b=50),
            bargap=0.3,
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.caption(
            "ℹ️ Dati `selected_strategies` non disponibili per questa combo "
            "(run calcolata prima della v2.0.0). Ricalcola la robustezza per visualizzare il grafico."
        )


# ─────────────────────────────────────────────────────────────────
# UI: render_optimizer_tab
# ─────────────────────────────────────────────────────────────────

def render_optimizer_tab(
    df_processed: Optional[pd.DataFrame],
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
) -> None:
    st.markdown("### ⚙️ Ottimizzatore Walk-Forward")
    st.caption(
        "Grid search su parametri WFA per massimizzare una metrica OOS "
        "(Sharpe, Calmar, PnL, Sortino, Recovery Factor)."
    )

    if df_processed is None or df_processed.empty:
        st.warning("⚠️ Carica e preprocessa prima un CSV nella tab principale.")
        return

    current_hash = _dataset_hash(df_processed)

    with st.expander("📂 Importa risultati da run precedente", expanded=False):
        uploaded = st.file_uploader(
            "Carica un file .json esportato da una run precedente",
            type=["json"],
            key="wfa_import_uploader",
            help="Evita di rilanciare la stessa ottimizzazione: carica il file JSON esportato in precedenza.",
        )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            df_imp, metric_imp, warn_imp = _import_run_json(raw_bytes, df_processed)
            if df_imp is None:
                st.error(warn_imp)
            else:
                if warn_imp:
                    st.warning(warn_imp)
                st.session_state["wfa_opt_results"] = df_imp
                st.session_state["wfa_opt_target"]  = metric_imp
                st.session_state.pop("wfa_robustness_cache", None)
                st.session_state.pop("wfa_robustness_top20_signature", None)
                payload_meta = json.loads(raw_bytes.decode())
                st.success(
                    f"✅ Run importata: **{len(df_imp)}** combinazioni · "
                    f"target `{metric_imp}` · "
                    f"esportata il **{payload_meta.get('exported_at','?')[:10]}**"
                )

    with st.expander("🎛️ Configura griglia di ottimizzazione", expanded=True):
        col_chk, col_range, col_target = st.columns([1.3, 1.7, 1.1])

        with col_chk:
            st.markdown("**Parametri da ottimizzare**")
            use_top_n     = st.checkbox("top_n", value=True)
            use_fwc       = st.checkbox("full_weight_count", value=True)
            use_fwp       = st.checkbox("full_weight_pct", value=True)
            use_bwp       = st.checkbox("bench_weight_pct", value=False)
            use_mpg       = st.checkbox("max_per_group", value=True)
            use_rank      = st.checkbox("ranking_metric", value=True)
            use_roc_en    = st.checkbox("roc_filter_enabled", value=True)
            use_roc_steps = st.checkbox("roc_filter_steps", value=False)

        with col_range:
            st.markdown("**Range / Valori**")
            top_n_min        = st.number_input("top_n min", 1, 20, 5)
            top_n_max        = st.number_input("top_n max", 1, 20, 10)
            fwc_min          = st.number_input("full_weight_count min", 1, 20, 3)
            fwp_values       = st.multiselect("full_weight_pct (%)", [70, 80, 90, 100], [70, 80, 90, 100])
            bwp_values       = st.multiselect("bench_weight_pct (%)", [30, 40, 50, 60], [30, 40])
            mpg_min          = st.number_input("max_per_group min", 1, 5, 1)
            mpg_max          = st.number_input("max_per_group max", 1, 5, 3)
            roc_steps_values = st.multiselect("roc_filter_steps", [1, 2, 3, 4], [1, 2, 3])

        with col_target:
            st.markdown("**Metrica target**")
            target_metric = st.selectbox(
                "Massimizza",
                ["sharpe", "calmar", "total_pnl", "sortino", "recovery_factor"],
                format_func=lambda x: {
                    "sharpe":          "📈 Sharpe OOS",
                    "calmar":          "📉 Calmar OOS",
                    "total_pnl":       "💰 Total PnL OOS",
                    "sortino":         "📊 Sortino OOS",
                    "recovery_factor": "🔄 Recovery Factor OOS",
                }[x],
            )
            st.markdown("---")
            st.markdown("**Finestra IS/OOS**")
            is_months = st.number_input("IS mesi", 3, 36, 12)
            oos_days  = st.number_input("OOS giorni", 7, 90, 28)

    def _fixed_session(key: str, default):
        return [st.session_state.get(key, default)]

    top_n_range     = list(range(int(top_n_min), int(top_n_max) + 1)) if use_top_n     else _fixed_session("top_n", 7)
    fwc_range       = list(range(int(fwc_min), int(top_n_max) + 1))   if use_fwc       else _fixed_session("full_weight_count", 3)
    fwp_range       = fwp_values or [80]                               if use_fwp       else _fixed_session("full_weight_pct", 80)
    bwp_range       = bwp_values or [40]                               if use_bwp       else _fixed_session("bench_weight_pct", 40)
    mpg_range       = list(range(int(mpg_min), int(mpg_max) + 1))     if use_mpg       else _fixed_session("max_per_group", 2)
    rank_range      = RANKING_METRICS                                   if use_rank      else _fixed_session("ranking_metric", DEFAULT_RANKING_METRIC)
    roc_en_range    = [True, False]                                     if use_roc_en    else _fixed_session("roc_filter_enabled", True)
    roc_steps_range = roc_steps_values or [2]                          if use_roc_steps else _fixed_session("roc_filter_steps", 2)

    raw_grid = list(itertools.product(
        top_n_range, fwc_range, fwp_range, bwp_range,
        mpg_range, rank_range, roc_en_range, roc_steps_range,
    ))
    grid = [g for g in raw_grid if g[1] <= g[0]]

    n_combos = len(grid)
    est_sec  = n_combos * 0.8
    est_str  = f"{est_sec:.0f}s" if est_sec < 120 else f"{est_sec/60:.1f} min"

    st.markdown(f"**Combinazioni totali:** `{n_combos:,}` &nbsp;·&nbsp; Stima tempo: `{est_str}`")

    if n_combos > 500:
        st.warning(f"⚠️ {n_combos:,} combinazioni potrebbero richiedere molto tempo. Riduci i range o disabilita alcuni parametri.")
    if n_combos > 2000:
        st.error("🛑 Più di 2000 combinazioni: alto rischio timeout. Limita la griglia.")

    run_btn = st.button("🚀 Avvia ottimizzazione", type="primary", disabled=(n_combos == 0))

    if run_btn:
        results: List[Dict] = []
        progress = st.progress(0.0, text="Inizializzazione…")
        t0 = time.time()
        update_every = max(1, n_combos // 200)

        for idx, (tn, fwc, fwp, bwp, mpg, rm, roc_en, roc_s) in enumerate(grid):
            if idx % update_every == 0 or idx == n_combos - 1:
                p_val = (idx + 1) / n_combos
                elapsed = time.time() - t0
                eta = (elapsed / (idx + 1)) * (n_combos - idx - 1) if idx > 0 else est_sec
                progress.progress(p_val, text=f"Combo {idx+1}/{n_combos} — ETA {eta:.0f}s")

            params = WFAParams(
                top_n=int(tn), full_weight_count=int(fwc),
                full_weight_pct=float(fwp), bench_weight_pct=float(bwp),
                max_per_group=int(mpg), ranking_metric=str(rm),
                roc_filter_enabled=bool(roc_en), roc_filter_steps=int(roc_s),
                is_months=int(is_months), oos_days=int(oos_days),
            )
            metrics = run_wfa_single(
                df=df_processed, params=params, group_map=group_map,
                date_col=date_col, strategy_col=strategy_col, pnl_col=pnl_col,
            )
            if metrics is None:
                continue

            row = {
                "top_n": params.top_n, "full_weight_count": params.full_weight_count,
                "full_weight_pct": params.full_weight_pct, "bench_weight_pct": params.bench_weight_pct,
                "max_per_group": params.max_per_group, "ranking_metric": params.ranking_metric,
                "roc_filter": params.roc_filter_enabled, "roc_steps": params.roc_filter_steps,
            }
            row.update(metrics)
            results.append(row)

        progress.progress(1.0, text="✅ Ottimizzazione completata")
        elapsed_total = time.time() - t0
        st.success(f"Completato in {elapsed_total:.1f}s — {len(results)} combinazioni valide su {n_combos}.")

        if not results:
            st.error("Nessun risultato valido. Verifica il dataset e riprova.")
            return

        df_res = (
            pd.DataFrame(results)
            .sort_values(target_metric, ascending=False)
            .reset_index(drop=True)
        )
        df_res.insert(0, "Rank", df_res.index + 1)
        st.session_state["wfa_opt_results"] = df_res
        st.session_state["wfa_opt_target"]  = target_metric
        st.session_state.pop("wfa_robustness_cache", None)
        st.session_state.pop("wfa_robustness_top20_signature", None)

    if "wfa_opt_results" not in st.session_state:
        return

    df_res = st.session_state["wfa_opt_results"]
    target_metric_used = st.session_state.get("wfa_opt_target", target_metric)
    top20  = df_res.head(20)

    st.markdown("---")
    st.markdown("#### 🏆 Top 20 combinazioni")

    show_cols = [
        "Rank", "top_n", "full_weight_count", "full_weight_pct", "bench_weight_pct",
        "max_per_group", "ranking_metric", "roc_filter", "roc_steps",
        "sharpe", "sortino", "calmar", "total_pnl", "max_dd", "recovery_factor", "n_oos_days",
    ]
    show_cols = [c for c in show_cols if c in top20.columns]

    def _style(row):
        if row["Rank"] == 1:
            return ["background-color:#1a3a2a;color:#7fffb2;font-weight:bold"] * len(row)
        return [""] * len(row)

    float_cols = ["sharpe", "sortino", "calmar", "recovery_factor"]
    money_cols = ["total_pnl", "max_dd"]
    display_df = top20[show_cols].copy()
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")
    for col in money_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda v: f"{v:,.2f}")

    st.dataframe(display_df.style.apply(_style, axis=1), use_container_width=True, height=480)

    best = df_res.iloc[0]
    st.markdown("---")
    st.markdown("#### 🥇 Combinazione ottimale")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📈 Sharpe OOS",      f"{best['sharpe']:.2f}")
    c2.metric("📊 Sortino OOS",     f"{best['sortino']:.2f}")
    c3.metric("📉 Calmar OOS",      f"{best['calmar']:.2f}")
    c4.metric("💰 Total PnL OOS",   f"${best['total_pnl']:,.2f}")
    c5.metric("🔄 Recovery Factor", f"{best['recovery_factor']:.2f}")

    st.markdown(
        f"`top_n={int(best['top_n'])}` · "
        f"`full_weight_count={int(best['full_weight_count'])}` · "
        f"`full_weight_pct={int(best['full_weight_pct'])}%` · "
        f"`bench_weight_pct={int(best['bench_weight_pct'])}%` · "
        f"`max_per_group={int(best['max_per_group'])}` · "
        f"`ranking_metric={best['ranking_metric']}` · "
        f"`roc_filter={bool(best['roc_filter'])}` · "
        f"`roc_steps={int(best['roc_steps'])}`"
    )

    if st.button("⚡ Applica questa configurazione", type="primary"):
        st.session_state["top_n"]              = int(best["top_n"])
        st.session_state["full_weight_count"]  = int(best["full_weight_count"])
        st.session_state["full_weight_pct"]    = int(best["full_weight_pct"])
        st.session_state["bench_weight_pct"]   = int(best["bench_weight_pct"])
        st.session_state["max_per_group"]      = int(best["max_per_group"])
        st.session_state["ranking_metric"]     = str(best["ranking_metric"])
        st.session_state["roc_filter_enabled"] = bool(best["roc_filter"])
        st.session_state["roc_filter_steps"]   = int(best["roc_steps"])
        st.success("Configurazione salvata in session_state — torna alla tab principale e riesegui il WFA.")
        st.rerun()

    render_robustness_section(
        df_res=top20,
        df_processed=df_processed,
        group_map=group_map,
        date_col=date_col,
        strategy_col=strategy_col,
        pnl_col=pnl_col,
        target_metric=target_metric_used,
    )

    st.markdown("---")
    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Esporta risultati completi (CSV)",
        data=csv_bytes,
        file_name="wfa_optimizer_results.csv",
        mime="text/csv",
    )

    json_bytes = _export_run_json(df_res, target_metric_used, df_processed)
    st.download_button(
        "💾 Esporta run (JSON — reimportabile)",
        data=json_bytes,
        file_name=f"wfa_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        help="Salva i risultati in un file JSON che puoi ricaricare in futuro senza rilanciare l'ottimizzazione.",
    )
