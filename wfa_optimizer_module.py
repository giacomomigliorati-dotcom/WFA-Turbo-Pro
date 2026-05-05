# ================================================================
#  MODULO OTTIMIZZATORE WFA — WFA-Turbo-Pro
#  File: wfa_optimizer_module.py
#  Versione: 1.3.1 — 2026-05-05  (fix box plot marker color list)
#
#  Questo modulo fornisce:
#  - run_wfa_single(...): motore Walk-Forward aggregato (grid search veloce)
#  - run_wfa_single_windowed(...): variante per-finestra OOS (analisi robustezza)
#  - compute_robustness_profile(...): metriche robustezza da finestre OOS
#  - render_optimizer_tab(...): UI Streamlit "⚙️ Ottimizzatore" per grid search
#  - render_robustness_section(...): scatter XY + box plot PF per finestra OOS
# ================================================================

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
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


def _profit_factor(pnl_list: List[float]) -> float:
    """Profit Factor = somma wins / somma abs(losses). Restituisce 0.0 se no losses."""
    arr = np.asarray(pnl_list, dtype=float)
    wins = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    if losses < 1e-9:
        return wins if wins > 0 else 0.0
    return round(float(wins / losses), 2)


def _select_strategies(
    is_data: pd.DataFrame,
    params: WFAParams,
    group_map: Dict[str, str],
    strategy_col: str,
    date_col: str,
    pnl_col: str,
) -> Dict[str, float]:
    """Logica IS comune: ranking + selezione + pesi. Restituisce dict {strategia: peso}."""
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
        metrics_rows.append({
            "strategy": strat,
            "score": score,
            "family": group_map.get(strat, "Unknown"),
            "total_pnl_is": float(strat_is["Net PnL"].sum()),
            "higher_is_better": higher_is_better,
        })

    if not metrics_rows:
        return {}

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
        return {}

    n_full = min(params.full_weight_count, len(selected))
    raw_weights: Dict[str, float] = {}
    for idx, row in enumerate(selected):
        w = params.full_weight_pct if idx < n_full else params.bench_weight_pct
        raw_weights[str(row["strategy"])] = w / 100.0

    tot_w = sum(raw_weights.values())
    if tot_w <= 0:
        return {}
    return {k: v / tot_w for k, v in raw_weights.items()}


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

    df = df[[date_col, strategy_col, pnl_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    all_oos_pnl: List[float] = []

    start_is = min_date
    while True:
        end_is = start_is + relativedelta(months=params.is_months) - pd.Timedelta(days=1)
        start_oos = end_is + pd.Timedelta(days=1)
        end_oos = start_oos + pd.Timedelta(days=params.oos_days - 1)

        if start_oos > max_date:
            break

        is_data = df[(df[date_col] >= start_is) & (df[date_col] <= end_is)]
        oos_data = df[(df[date_col] >= start_oos) & (df[date_col] <= end_oos)]

        if is_data.empty or oos_data.empty:
            start_is = start_oos
            continue

        weights = _select_strategies(is_data, params, group_map, strategy_col, date_col, pnl_col)
        if not weights:
            start_is = start_oos
            continue

        oos_sub = oos_data[oos_data[strategy_col].isin(weights.keys())]
        if oos_sub.empty:
            start_is = start_oos
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
        start_is = start_oos

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

    recent_n = max(5, min(126, pnl.size // 2))
    recent = pnl[-recent_n:]
    r_mean = float(np.mean(recent))
    r_std = float(np.std(recent))
    recent_sharpe = (r_mean / (r_std or 1e-9)) * math.sqrt(252)
    recent_pnl = float(np.sum(recent))
    recent_dd = _max_drawdown(recent)

    return {
        "sharpe":          round(sharpe, 2),
        "sortino":         round(sortino, 2),
        "calmar":          round(calmar, 2),
        "total_pnl":       round(total_pnl, 2),
        "max_dd":          round(max_dd, 2),
        "recovery_factor": round(recovery, 2),
        "recent_sharpe":   round(recent_sharpe, 2),
        "recent_pnl":      round(recent_pnl, 2),
        "recent_max_dd":   round(recent_dd, 2),
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
    """Variante di run_wfa_single che restituisce metriche per ogni finestra OOS separata.

    Ogni elemento della lista corrisponde a una finestra OOS e contiene:
    - window (int): indice progressivo
    - start_oos, end_oos (str): date della finestra
    - pf (float): Profit Factor della finestra
    - max_dd (float): max drawdown intra-finestra
    - n_trade (int): numero trade nella finestra
    - total_pnl (float): PnL totale pesato della finestra
    """
    if df is None or df.empty:
        return []
    if any(c not in df.columns for c in (date_col, strategy_col, pnl_col)):
        return []

    df = df[[date_col, strategy_col, pnl_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    windows: List[Dict] = []
    window_idx = 0

    start_is = min_date
    while True:
        end_is = start_is + relativedelta(months=params.is_months) - pd.Timedelta(days=1)
        start_oos = end_is + pd.Timedelta(days=1)
        end_oos = start_oos + pd.Timedelta(days=params.oos_days - 1)

        if start_oos > max_date:
            break

        is_data = df[(df[date_col] >= start_is) & (df[date_col] <= end_is)]
        oos_data = df[(df[date_col] >= start_oos) & (df[date_col] <= end_oos)]

        if is_data.empty or oos_data.empty:
            start_is = start_oos
            continue

        weights = _select_strategies(is_data, params, group_map, strategy_col, date_col, pnl_col)
        if not weights:
            start_is = start_oos
            continue

        oos_sub = oos_data[oos_data[strategy_col].isin(weights.keys())].copy()
        if oos_sub.empty:
            start_is = start_oos
            continue

        # PnL pesato per singolo trade (non aggregato per giorno → serve per PF)
        oos_sub["weighted_pnl"] = oos_sub.apply(
            lambda r: r[pnl_col] * weights.get(r[strategy_col], 0.0), axis=1
        )
        trade_pnls = oos_sub["weighted_pnl"].tolist()

        # Aggrega per giorno per max_dd
        oos_sub_daily = oos_sub.groupby(date_col)["weighted_pnl"].sum()
        daily_arr = np.asarray(oos_sub_daily.tolist(), dtype=float)

        window_idx += 1
        windows.append({
            "window":    window_idx,
            "start_oos": str(start_oos.date()),
            "end_oos":   str(end_oos.date()),
            "pf":        _profit_factor(trade_pnls),
            "max_dd":    round(_max_drawdown(daily_arr), 2),
            "n_trade":   len(trade_pnls),
            "total_pnl": round(float(sum(trade_pnls)), 2),
        })

        start_is = start_oos

    return windows


# ─────────────────────────────────────────────────────────────────
# COMPUTE: profilo di robustezza da lista finestre OOS
# ─────────────────────────────────────────────────────────────────

def compute_robustness_profile(
    windows: List[Dict],
    dd_threshold: float = 0.15,
) -> Optional[Dict]:
    """Calcola il profilo di robustezza aggregato da una lista di finestre OOS.

    dd_threshold: soglia max_dd assoluta (es. 0.15 = $150 su conto $1000).
    Restituisce None se windows è vuoto o ha meno di 3 finestre valide.
    """
    if not windows or len(windows) < 3:
        return None

    pf_vals = [w["pf"] for w in windows if w["n_trade"] >= 3]
    if len(pf_vals) < 3:
        return None

    pf_arr = np.asarray(pf_vals, dtype=float)
    dd_vals = [w["max_dd"] for w in windows]

    return {
        "pf_median":          round(float(np.median(pf_arr)), 2),
        "pf_p25":             round(float(np.percentile(pf_arr, 25)), 2),
        "pf_pct_above_1":     round(float(np.mean(pf_arr > 1.0)), 2),
        "dd_pct_below_thresh": round(float(np.mean(np.asarray(dd_vals) < dd_threshold)), 2),
        "n_windows":          len(windows),
        "n_windows_valid":    len(pf_vals),
        "pf_per_window":      [w["pf"] for w in windows],
        "dd_per_window":      dd_vals,
        "windows_detail":     windows,
    }


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
    """Sezione analisi di robustezza post-ottimizzazione.

    Riesegue i 20 run windowed sulle Top-20 combinazioni, poi mostra:
    1. Scatter XY: metrica target (asse X) vs PF P25 (asse Y)
    2. Box plot PF per finestra OOS della combinazione selezionata
    """
    st.markdown("---")
    st.markdown("### 🔬 Analisi di Robustezza — Top 20")
    st.caption(
        "Verifica la consistenza per finestra OOS: le combinazioni migliori "
        "sono quelle in alto a destra nello scatter (alta metrica target + alto PF P25)."
    )

    if df_processed is None or df_processed.empty or df_res.empty:
        st.warning("⚠️ Dati non disponibili per l'analisi di robustezza.")
        return

    # ── Controllo soglia drawdown ──────────────────────────────────────
    dd_threshold = st.slider(
        "📉 Soglia Max Drawdown per colore scatter ($)",
        min_value=50, max_value=2000, value=150, step=50,
        help="Finestre OOS con max_dd < questa soglia vengono considerate 'sicure' per il colore dei punti.",
        key="robustness_dd_threshold",
    )
    dd_thresh_val = float(dd_threshold)

    # ── Riesecuzione windowed sulle Top-20 ────────────────────────────
    cache_key = "wfa_robustness_cache"
    top20 = df_res.head(20).copy()

    need_recompute = (
        cache_key not in st.session_state
        or st.session_state.get("wfa_robustness_n_rows", 0) != len(top20)
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
        st.session_state[cache_key] = profiles
        st.session_state["wfa_robustness_n_rows"] = len(top20)
    else:
        profiles = st.session_state[cache_key]

    # ── Costruzione dati scatter ───────────────────────────────────────
    scatter_rows = []
    for i, (_, row) in enumerate(top20.iterrows()):
        prof = profiles[i] if i < len(profiles) else None
        pf_p25 = prof["pf_p25"] if prof else None
        pf_pct = prof["pf_pct_above_1"] if prof else 0.0
        dd_pct = prof["dd_pct_below_thresh"] if prof else 0.0

        # Ricalcola dd_pct_below_thresh con la soglia slider corrente
        if prof and "dd_per_window" in prof:
            dd_arr = np.asarray(prof["dd_per_window"], dtype=float)
            dd_pct = float(np.mean(dd_arr < dd_thresh_val))

        scatter_rows.append({
            "rank":        int(row["Rank"]),
            "target_val":  float(row.get(target_metric, 0.0)),
            "pf_p25":      pf_p25,
            "pf_pct":      round(pf_pct * 100, 1),   # % finestre PF>1
            "dd_pct":      round(dd_pct * 100, 1),    # % finestre dd<soglia
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

    # ── Scatter XY ────────────────────────────────────────────────────
    st.markdown(f"#### 📊 Scatter: {target_metric.capitalize()} vs PF P25 per finestra OOS")
    st.caption(
        "**Asse X** = metrica target aggregata · "
        "**Asse Y** = primo quartile del Profit Factor per finestra OOS · "
        "**Dimensione** = % finestre con PF > 1.0 · "
        "**Colore** = % finestre con Max DD < soglia (verde = più sicuro)"
    )

    target_vals = [r["target_val"] for r in valid]
    pf_p25_vals = [r["pf_p25"] for r in valid]
    size_vals   = [max(8.0, r["pf_pct"] * 0.4) for r in valid]
    color_vals  = [r["dd_pct"] for r in valid]
    labels      = [r["label"] for r in valid]
    ranks       = [r["rank"] for r in valid]

    # Testo visibile solo per rank 1-3
    text_labels = [str(r["rank"]) if r["rank"] <= 3 else "" for r in valid]

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
                [0.0, "#7f1d1d"],   # rosso scuro — DD frequente
                [0.4, "#b45309"],   # arancio
                [0.7, "#065f46"],   # verde scuro
                [1.0, "#6ee7b7"],   # verde chiaro — DD raro
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
            ranks,
            [r["pf_pct"] for r in valid],
            [r["dd_pct"] for r in valid],
        )),
        hovertemplate=(
            "%{meta}<br>"
            f"<b>{target_metric}:</b> %{{x:.2f}}<br>"
            "<b>PF P25:</b> %{y:.2f}<br>"
            "<b>% finestre PF>1:</b> %{customdata[1]:.0f}%<br>"
            "<b>% finestre DD<soglia:</b> %{customdata[2]:.0f}%<br>"
            "<extra></extra>"
        ),
        meta=labels,
        name="Combinazioni",
    ))

    # Linea orizzontale PF = 1.0
    fig_scatter.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="PF = 1.0 (soglia sopravvivenza)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=10),
    )

    # Annotazioni per rank 1-3
    for r in valid:
        if r["rank"] <= 3:
            fig_scatter.add_annotation(
                x=r["target_val"],
                y=r["pf_p25"],
                text=f"  #{r['rank']}",
                showarrow=False,
                font=dict(color="#e0eaf4", size=10),
                xanchor="left",
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
        st.caption(
            f"⚠️ {len(invalid)} combinazioni escluse dallo scatter per dati OOS insufficienti."
        )

    # ── Selectbox + Box plot on-demand ────────────────────────────────
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
            f"roc={'✓' if row['roc_filter'] else '✗'}"
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

    # Metriche rapide
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("PF Mediana",      f"{prof['pf_median']:.2f}")
    mc2.metric("PF P25",          f"{prof['pf_p25']:.2f}")
    mc3.metric("% finestre PF>1", f"{prof['pf_pct_above_1']*100:.0f}%")
    mc4.metric("% finestre DD<soglia", f"{prof['dd_pct_below_thresh']*100:.0f}%")

    # ── Box plot — FIX: go.Box senza boxpoints + go.Scatter overlay per colori per-punto
    # Plotly non supporta marker.color come lista dentro go.Box con boxpoints="all".
    # Soluzione: box puro per la forma statistica, poi Scatter overlay per i jitter points
    # con colori individuali.
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.25, 0.25, size=len(pf_vals_sel)).tolist()
    point_colors = ["#6ee7b7" if v >= 1.0 else "#f87171" for v in pf_vals_sel]
    hover_texts = [
        f"Finestra {i+1}<br>PF: {v:.2f}" for i, v in enumerate(pf_vals_sel)
    ]

    fig_box = go.Figure()

    # Trace 1: box statistico (no punti individuali)
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

    # Trace 2: scatter overlay con colori per-punto (verde PF≥1, rosso PF<1)
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

    # Linea PF = 1.0
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
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
            showgrid=False,
            range=[-1, 1],
        ),
        yaxis=dict(
            title="Profit Factor",
            gridcolor="#1e2a35",
        ),
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

    # Mini-tabella finestre cronologica
    with st.expander("📋 Dettaglio finestre OOS (cronologico)", expanded=False):
        detail_df = pd.DataFrame(windows_detail)
        detail_df["semaforo"] = detail_df["pf"].apply(
            lambda v: "🟢" if v >= 1.2 else ("🟡" if v >= 1.0 else "🔴")
        )
        detail_df["pf"]       = detail_df["pf"].map(lambda v: f"{v:.2f}")
        detail_df["max_dd"]   = detail_df["max_dd"].map(lambda v: f"{v:,.2f}")
        detail_df["total_pnl"]= detail_df["total_pnl"].map(lambda v: f"{v:,.2f}")
        st.dataframe(
            detail_df[["window", "start_oos", "end_oos", "pf", "max_dd", "n_trade", "total_pnl", "semaforo"]],
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────
# UI: render_optimizer_tab  (invariata + chiamata robustezza in coda)
# ─────────────────────────────────────────────────────────────────

def render_optimizer_tab(
    df_processed: Optional[pd.DataFrame],
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
) -> None:
    """Renderizza la tab ⚙️ Ottimizzatore in una app Streamlit esistente."""
    st.markdown("### ⚙️ Ottimizzatore Walk-Forward")
    st.caption(
        "Grid search su parametri WFA per massimizzare una metrica OOS "
        "(Sharpe, Calmar, PnL, Sortino, Recovery Factor)."
    )

    if df_processed is None or df_processed.empty:
        st.warning("⚠️ Carica e preprocessa prima un CSV nella tab principale.")
        return

    with st.expander("🎛️ Configura griglia di ottimizzazione", expanded=True):
        col_chk, col_range, col_target = st.columns([1.3, 1.7, 1.1])

        with col_chk:
            st.markdown("**Parametri da ottimizzare**")
            use_top_n    = st.checkbox("top_n", value=True)
            use_fwc      = st.checkbox("full_weight_count", value=True)
            use_fwp      = st.checkbox("full_weight_pct", value=True)
            use_bwp      = st.checkbox("bench_weight_pct", value=False)
            use_mpg      = st.checkbox("max_per_group", value=True)
            use_rank     = st.checkbox("ranking_metric", value=True)
            use_roc_en   = st.checkbox("roc_filter_enabled", value=True)
            use_roc_steps= st.checkbox("roc_filter_steps", value=False)

        with col_range:
            st.markdown("**Range / Valori**")
            top_n_min       = st.number_input("top_n min", 1, 20, 5)
            top_n_max       = st.number_input("top_n max", 1, 20, 10)
            fwc_min         = st.number_input("full_weight_count min", 1, 20, 3)
            fwp_values      = st.multiselect("full_weight_pct (%)", [70, 80, 90, 100], [70, 80, 90, 100])
            bwp_values      = st.multiselect("bench_weight_pct (%)", [30, 40, 50, 60], [30, 40])
            mpg_min         = st.number_input("max_per_group min", 1, 5, 1)
            mpg_max         = st.number_input("max_per_group max", 1, 5, 3)
            roc_steps_values= st.multiselect("roc_filter_steps", [1, 2, 3, 4], [1, 2, 3])

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

    top_n_range   = list(range(int(top_n_min), int(top_n_max) + 1)) if use_top_n   else _fixed_session("top_n", 7)
    fwc_range     = list(range(int(fwc_min), int(top_n_max) + 1))   if use_fwc     else _fixed_session("full_weight_count", 3)
    fwp_range     = fwp_values or [80]                                if use_fwp     else _fixed_session("full_weight_pct", 80)
    bwp_range     = bwp_values or [40]                                if use_bwp     else _fixed_session("bench_weight_pct", 40)
    mpg_range     = list(range(int(mpg_min), int(mpg_max) + 1))      if use_mpg     else _fixed_session("max_per_group", 2)
    rank_range    = RANKING_METRICS                                    if use_rank    else _fixed_session("ranking_metric", DEFAULT_RANKING_METRIC)
    roc_en_range  = [True, False]                                      if use_roc_en  else _fixed_session("roc_filter_enabled", True)
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
                p = (idx + 1) / n_combos
                elapsed = time.time() - t0
                eta = (elapsed / (idx + 1)) * (n_combos - idx - 1) if idx > 0 else est_sec
                progress.progress(p, text=f"Combo {idx+1}/{n_combos} — ETA {eta:.0f}s")

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

        df_res = pd.DataFrame(results).sort_values(target_metric, ascending=False).reset_index(drop=True)
        df_res.insert(0, "Rank", df_res.index + 1)
        df_res["overfit_flag"] = df_res.apply(
            lambda r: "🔴 Sì" if (r["recent_sharpe"] < r["sharpe"] * 0.5 and r["sharpe"] > 0) else "🟢 No",
            axis=1,
        )
        st.session_state["wfa_opt_results"] = df_res
        st.session_state["wfa_opt_target"]  = target_metric
        # Invalida cache robustezza al nuovo run
        st.session_state.pop("wfa_robustness_cache", None)

    if "wfa_opt_results" not in st.session_state:
        return

    df_res = st.session_state["wfa_opt_results"]
    top20  = df_res.head(20)

    st.markdown("---")
    st.markdown("#### 🏆 Top 20 combinazioni")

    show_cols = [
        "Rank", "top_n", "full_weight_count", "full_weight_pct", "bench_weight_pct",
        "max_per_group", "ranking_metric", "roc_filter", "roc_steps",
        "sharpe", "sortino", "calmar", "total_pnl", "recovery_factor",
        "recent_sharpe", "recent_pnl", "n_oos_days", "overfit_flag",
    ]
    show_cols = [c for c in show_cols if c in top20.columns]

    def _style(row):
        if row["Rank"] == 1:
            return ["background-color:#1a3a2a;color:#7fffb2;font-weight:bold"] * len(row)
        if row.get("overfit_flag", "🟢 No") == "🔴 Sì":
            return ["background-color:#2a1a1a;color:#fca5a5"] * len(row)
        return [""] * len(row)

    float_cols = ["sharpe", "sortino", "calmar", "recovery_factor", "recent_sharpe"]
    money_cols = ["total_pnl", "recent_pnl", "max_dd", "recent_max_dd"]
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

    if best.get("overfit_flag", "🟢 No") == "🔴 Sì":
        st.warning(
            f"⚠️ Anti-overfitting: Sharpe recente {best['recent_sharpe']:.2f} "
            f"< 50% di Sharpe storico {best['sharpe']:.2f}."
        )
    else:
        st.success(
            f"✅ Stabilità buona: Sharpe recente {best['recent_sharpe']:.2f} "
            f"vs storico {best['sharpe']:.2f}."
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

    # Grafico Sharpe storico vs recente
    st.markdown("---")
    st.markdown("#### 🔍 Sharpe storico vs recente (Top 20)")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top20["Rank"].astype(str), y=top20["sharpe"],
        name="Sharpe OOS storico", marker_color="#4f98a3",
    ))
    fig_bar.add_trace(go.Bar(
        x=top20["Rank"].astype(str), y=top20["recent_sharpe"],
        name="Sharpe OOS recente", marker_color="#e8af34",
    ))
    fig_bar.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e0eaf4"),
        xaxis_title="Rank", yaxis_title="Sharpe Ratio",
        height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sezione robustezza (scatter XY + box plot) ─────────────────
    render_robustness_section(
        df_res=top20,
        df_processed=df_processed,
        group_map=group_map,
        date_col=date_col,
        strategy_col=strategy_col,
        pnl_col=pnl_col,
        target_metric=st.session_state.get("wfa_opt_target", target_metric),
    )

    # Export CSV completo
    st.markdown("---")
    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Esporta risultati completi (CSV)",
        data=csv_bytes,
        file_name="wfa_optimizer_results.csv",
        mime="text/csv",
    )
