# ================================================================
#  MODULO OTTIMIZZATORE WFA — WFA-Turbo-Pro
#  File: wfa_optimizer_module.py
#  Versione: 1.0.0 — 2026-05-04
#
#  Questo modulo fornisce:
#  - run_wfa_single(...): motore Walk-Forward parametrico per una singola combinazione
#  - render_optimizer_tab(...): UI Streamlit "⚙️ Ottimizzatore" per grid search dei parametri
#
#  Integrazione tipica (da fare in streamlit_wfa_app.py):
#      from wfa_optimizer_module import run_wfa_single, render_optimizer_tab
#      ...
#      with tab_opt:
#          render_optimizer_tab(df_processed=df_preprocessed, group_map=GROUP_MAP,
#                               date_col="Date Closed", strategy_col="Strategy",
#                               pnl_col="Net PnL")
# ================================================================

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta


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
    ranking_metric: str  # "Sharpe", "Sortino", "Omega", "Ulcer", "ROC"
    roc_filter_enabled: bool
    roc_filter_steps: int
    is_months: int = 12
    oos_days: int = 28


# ─────────────────────────────────────────────────────────────────
# CORE: singolo loop WFA
# ─────────────────────────────────────────────────────────────────

def _compute_rank_score(returns: np.ndarray, metric: str, roc_steps: int) -> float:
    """Calcola lo score IS per una serie di PnL giornalieri."""
    if returns.size < 2:
        return -999.0

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))

    if metric == "Sharpe":
        return (mean_r / (std_r or 1e-9)) * math.sqrt(252)

    if metric == "Sortino":
        neg = returns[returns < 0]
        dstd = float(np.std(neg)) if neg.size > 1 else 0.0
        return (mean_r / (dstd or 1e-9)) * math.sqrt(252)

    if metric == "Omega":
        gains = float(np.sum(np.maximum(returns, 0)))
        losses = float(np.sum(np.maximum(-returns, 0)))
        return gains / (losses or 1e-9)

    if metric == "Ulcer":
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        ui = math.sqrt(float(np.mean(dd**2))) if dd.size > 0 else 0.0
        return mean_r / (ui or 1e-9)

    if metric == "ROC":
        # Rate of Change sul PnL cumulativo nella finestra IS
        cum = np.cumsum(returns)
        return (cum[-1] - cum[0]) / (abs(cum[0]) + 1e-9)

    # Default: Sharpe
    return (mean_r / (std_r or 1e-9)) * math.sqrt(252)


def _passes_roc_filter(returns: np.ndarray, steps: int) -> bool:
    """True se il ROC recente su 'steps' periodi è >= 0."""
    if steps <= 0 or returns.size <= steps:
        return True
    cum = np.cumsum(returns)
    base = cum[-steps - 1]
    recent_roc = (cum[-1] - base) / (abs(base) + 1e-9)
    return recent_roc >= 0


def _max_drawdown(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    mdd = float(np.min(dd))
    return abs(mdd) if mdd < 0 else 0.0


def run_wfa_single(
    df: pd.DataFrame,
    params: WFAParams,
    group_map: Dict[str, str],
    *,
    date_col: str = "Date",
    strategy_col: str = "Strategy",
    pnl_col: str = "Net_PnL",
) -> Optional[Dict[str, float]]:
    """Esegue un singolo loop Walk-Forward rolling e restituisce metriche OOS.

    Il DataFrame deve contenere colonne:
    - date_col: data del trade / giorno (sarà convertita in datetime)
    - strategy_col: nome strategia
    - pnl_col: PnL (già netto commissioni)

    Non modifica df originale.
    """
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

        # ── Ranking IS per strategia ────────────────────────────
        metrics_rows = []
        for strat, grp in is_data.groupby(strategy_col):
            pnl = grp[pnl_col].to_numpy(dtype=float)
            if pnl.size < 5:
                continue

            # ROC filter opzionale
            if params.roc_filter_enabled and not _passes_roc_filter(pnl, params.roc_filter_steps):
                continue

            score = _compute_rank_score(pnl, params.ranking_metric, params.roc_filter_steps)
            metrics_rows.append(
                {
                    "strategy": strat,
                    "score": score,
                    "family": group_map.get(strat, "Unknown"),
                    "total_pnl_is": float(pnl.sum()),
                }
            )

        if not metrics_rows:
            start_is = start_oos
            continue

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df = metrics_df.sort_values(["score", "total_pnl_is"], ascending=[False, False])

        # ── Selezione top_n con vincolo max_per_group ───────────
        selected: List[Dict[str, object]] = []
        family_count: Dict[str, int] = {}
        for _, row in metrics_df.iterrows():
            strat = row["strategy"]
            fam = row["family"]
            if family_count.get(fam, 0) < params.max_per_group:
                selected.append(row.to_dict())
                family_count[fam] = family_count.get(fam, 0) + 1
            if len(selected) >= params.top_n:
                break

        if not selected:
            start_is = start_oos
            continue

        # ── Pesi (pieno / panchina) ─────────────────────────────
        n_full = min(params.full_weight_count, len(selected))
        raw_weights: Dict[str, float] = {}
        for idx, row in enumerate(selected):
            w = params.full_weight_pct if idx < n_full else params.bench_weight_pct
            raw_weights[str(row["strategy"])] = w / 100.0

        tot_w = sum(raw_weights.values())
        if tot_w <= 0:
            start_is = start_oos
            continue

        weights = {k: v / tot_w for k, v in raw_weights.items()}

        # ── PnL OOS pesato ──────────────────────────────────────
        oos_sub = oos_data[oos_data[strategy_col].isin(weights.keys())]
        if oos_sub.empty:
            start_is = start_oos
            continue

        oos_daily = (
            oos_sub.groupby([date_col, strategy_col])[pnl_col]
            .sum()
            .reset_index()
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

    # Sub-periodo recente (~ultimi 6 mesi trading ≈ 126 giorni)
    recent_n = max(5, min(126, pnl.size // 2))
    recent = pnl[-recent_n:]
    r_mean = float(np.mean(recent))
    r_std = float(np.std(recent))
    recent_sharpe = (r_mean / (r_std or 1e-9)) * math.sqrt(252)
    recent_pnl = float(np.sum(recent))
    recent_dd = _max_drawdown(recent)

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
        "recovery_factor": round(recovery, 4),
        "recent_sharpe": round(recent_sharpe, 4),
        "recent_pnl": round(recent_pnl, 2),
        "recent_max_dd": round(recent_dd, 2),
        "n_oos_days": int(pnl.size),
    }


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
    """Renderizza la tab ⚙️ Ottimizzatore in una app Streamlit esistente.

    - df_processed deve essere il DataFrame preprocessato (dopo filtri Legendary, Net PnL, ecc.).
    - group_map mappa Strategy → famiglia (famiglie usate per max_per_group).
    """
    st.markdown("### ⚙️ Ottimizzatore Walk-Forward")
    st.caption(
        "Grid search su parametri WFA per massimizzare una metrica OOS "
        "(Sharpe, Calmar, PnL, Sortino, Recovery Factor)."
    )

    if df_processed is None or df_processed.empty:
        st.warning("⚠️ Carica e preprocessa prima un CSV nella tab principale.")
        return

    # ── Configurazione griglia ──────────────────────────────────
    with st.expander("🎛️ Configura griglia di ottimizzazione", expanded=True):
        col_chk, col_range, col_target = st.columns([1.3, 1.7, 1.1])

        with col_chk:
            st.markdown("**Parametri da ottimizzare**")
            use_top_n = st.checkbox("top_n", value=True)
            use_fwc = st.checkbox("full_weight_count", value=True)
            use_fwp = st.checkbox("full_weight_pct", value=True)
            use_bwp = st.checkbox("bench_weight_pct", value=False)
            use_mpg = st.checkbox("max_per_group", value=True)
            use_rank = st.checkbox("ranking_metric", value=True)
            use_roc_en = st.checkbox("roc_filter_enabled", value=True)
            use_roc_steps = st.checkbox("roc_filter_steps", value=False)

        with col_range:
            st.markdown("**Range / Valori**")
            top_n_min = st.number_input("top_n min", 1, 20, 5)
            top_n_max = st.number_input("top_n max", 1, 20, 10)
            fwc_min = st.number_input("full_weight_count min", 1, 20, 3)
            fwp_values = st.multiselect("full_weight_pct (%)", [70, 80, 90, 100], [70, 80, 90, 100])
            bwp_values = st.multiselect("bench_weight_pct (%)", [30, 40, 50, 60], [30, 40])
            mpg_min = st.number_input("max_per_group min", 1, 5, 1)
            mpg_max = st.number_input("max_per_group max", 1, 5, 3)
            roc_steps_values = st.multiselect("roc_filter_steps", [1, 2, 3, 4], [1, 2, 3])

        with col_target:
            st.markdown("**Metrica target**")
            target_metric = st.selectbox(
                "Massimizza",
                ["sharpe", "calmar", "total_pnl", "sortino", "recovery_factor"],
                format_func=lambda x: {
                    "sharpe": "📈 Sharpe OOS",
                    "calmar": "📉 Calmar OOS",
                    "total_pnl": "💰 Total PnL OOS",
                    "sortino": "📊 Sortino OOS",
                    "recovery_factor": "🔄 Recovery Factor OOS",
                }[x],
            )
            st.markdown("---")
            st.markdown("**Finestra IS/OOS**")
            is_months = st.number_input("IS mesi", 3, 36, 12)
            oos_days = st.number_input("OOS giorni", 7, 90, 28)

    # ── Costruzione griglia di combinazioni ─────────────────────
    ALL_RANKING = ["Sharpe", "Sortino", "Omega", "Ulcer", "ROC"]

    def _fixed_session(key: str, default):
        return [st.session_state.get(key, default)]

    top_n_range = list(range(int(top_n_min), int(top_n_max) + 1)) if use_top_n else _fixed_session("top_n", 7)
    fwc_range = list(range(int(fwc_min), int(top_n_max) + 1)) if use_fwc else _fixed_session("full_weight_count", 3)
    fwp_range = fwp_values or [80] if use_fwp else _fixed_session("full_weight_pct", 80)
    bwp_range = bwp_values or [40] if use_bwp else _fixed_session("bench_weight_pct", 40)
    mpg_range = list(range(int(mpg_min), int(mpg_max) + 1)) if use_mpg else _fixed_session("max_per_group", 2)
    rank_range = ALL_RANKING if use_rank else _fixed_session("ranking_metric", "Omega")
    roc_en_range = [True, False] if use_roc_en else _fixed_session("roc_filter_enabled", True)
    roc_steps_range = roc_steps_values or [2] if use_roc_steps else _fixed_session("roc_filter_steps", 2)

    raw_grid = list(
        itertools.product(
            top_n_range,
            fwc_range,
            fwp_range,
            bwp_range,
            mpg_range,
            rank_range,
            roc_en_range,
            roc_steps_range,
        )
    )
    grid = [g for g in raw_grid if g[1] <= g[0]]  # fwc <= top_n

    n_combos = len(grid)
    est_sec = n_combos * 0.8
    est_str = f"{est_sec:.0f}s" if est_sec < 120 else f"{est_sec/60:.1f} min"

    st.markdown(f"**Combinazioni totali:** `{n_combos:,}` &nbsp;·&nbsp; Stima tempo: `{est_str}`")

    if n_combos > 500:
        st.warning(
            f"⚠️ {n_combos:,} combinazioni potrebbero richiedere molto tempo. "
            "Riduci i range o disabilita alcuni parametri."
        )
    if n_combos > 2000:
        st.error("🛑 Più di 2000 combinazioni: alto rischio timeout. Limita la griglia.")

    run_btn = st.button("🚀 Avvia ottimizzazione", type="primary", disabled=(n_combos == 0))

    if run_btn:
        results: List[Dict[str, float]] = []
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
                top_n=int(tn),
                full_weight_count=int(fwc),
                full_weight_pct=float(fwp),
                bench_weight_pct=float(bwp),
                max_per_group=int(mpg),
                ranking_metric=str(rm),
                roc_filter_enabled=bool(roc_en),
                roc_filter_steps=int(roc_s),
                is_months=int(is_months),
                oos_days=int(oos_days),
            )

            metrics = run_wfa_single(
                df=df_processed,
                params=params,
                group_map=group_map,
                date_col=date_col,
                strategy_col=strategy_col,
                pnl_col=pnl_col,
            )
            if metrics is None:
                continue

            row = {
                "top_n": params.top_n,
                "full_weight_count": params.full_weight_count,
                "full_weight_pct": params.full_weight_pct,
                "bench_weight_pct": params.bench_weight_pct,
                "max_per_group": params.max_per_group,
                "ranking_metric": params.ranking_metric,
                "roc_filter": params.roc_filter_enabled,
                "roc_steps": params.roc_filter_steps,
            }
            row.update(metrics)
            results.append(row)

        progress.progress(1.0, text="✅ Ottimizzazione completata")
        elapsed_total = time.time() - t0
        st.success(
            f"Completato in {elapsed_total:.1f}s — {len(results)} combinazioni valide su {n_combos}."
        )

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
        st.session_state["wfa_opt_target"] = target_metric

    # ── Visualizzazione risultati ────────────────────────────────
    if "wfa_opt_results" not in st.session_state:
        return

    df_res = st.session_state["wfa_opt_results"]
    top20 = df_res.head(20)

    st.markdown("---")
    st.markdown("#### 🏆 Top 20 combinazioni")

    show_cols = [
        "Rank",
        "top_n",
        "full_weight_count",
        "full_weight_pct",
        "bench_weight_pct",
        "max_per_group",
        "ranking_metric",
        "roc_filter",
        "roc_steps",
        "sharpe",
        "sortino",
        "calmar",
        "total_pnl",
        "recovery_factor",
        "recent_sharpe",
        "recent_pnl",
        "n_oos_days",
        "overfit_flag",
    ]
    show_cols = [c for c in show_cols if c in top20.columns]

    def _style(row):
        if row["Rank"] == 1:
            return ["background-color:#1a3a2a;color:#7fffb2;font-weight:bold"] * len(row)
        if row.get("overfit_flag", "🟢 No") == "🔴 Sì":
            return ["background-color:#2a1a1a;color:#fca5a5"] * len(row)
        return [""] * len(row)

    st.dataframe(
        top20[show_cols].style.apply(_style, axis=1),
        use_container_width=True,
        height=480,
    )

    # Dettaglio combinazione migliore
    best = df_res.iloc[0]
    st.markdown("---")
    st.markdown("#### 🥇 Combinazione ottimale")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📈 Sharpe OOS", f"{best.sharpe:.3f}")
    c2.metric("📊 Sortino OOS", f"{best.sortino:.3f}")
    c3.metric("📉 Calmar OOS", f"{best.calmar:.3f}")
    c4.metric("💰 Total PnL OOS", f"{best.total_pnl:,.0f}")
    c5.metric("🔄 Recovery Factor", f"{best.recovery_factor:.3f}")

    st.markdown(
        f"`top_n={int(best.top_n)}` · "
        f"`full_weight_count={int(best.full_weight_count)}` · "
        f"`full_weight_pct={int(best.full_weight_pct)}%` · "
        f"`bench_weight_pct={int(best.bench_weight_pct)}%` · "
        f"`max_per_group={int(best.max_per_group)}` · "
        f"`ranking_metric={best.ranking_metric}` · "
        f"`roc_filter={bool(best.roc_filter)}` · "
        f"`roc_steps={int(best.roc_steps)}`"
    )

    if best.get("overfit_flag", "🟢 No") == "🔴 Sì":
        st.warning(
            f"⚠️ Anti-overfitting: Sharpe recente {best.recent_sharpe:.3f} < 50% di Sharpe storico {best.sharpe:.3f}."
        )
    else:
        st.success(
            f"✅ Stabilità buona: Sharpe recente {best.recent_sharpe:.3f} vs storico {best.sharpe:.3f}."
        )

    # Pulsante applica
    if st.button("⚡ Applica questa configurazione", type="primary"):
        st.session_state["top_n"] = int(best.top_n)
        st.session_state["full_weight_count"] = int(best.full_weight_count)
        st.session_state["full_weight_pct"] = int(best.full_weight_pct)
        st.session_state["bench_weight_pct"] = int(best.bench_weight_pct)
        st.session_state["max_per_group"] = int(best.max_per_group)
        st.session_state["ranking_metric"] = str(best.ranking_metric)
        st.session_state["roc_filter_enabled"] = bool(best.roc_filter)
        st.session_state["roc_filter_steps"] = int(best.roc_steps)
        st.success("Configurazione salvata in session_state — torna alla tab principale e riesegui il WFA.")
        st.rerun()

    # Grafico anti-overfitting
    st.markdown("---")
    st.markdown("#### 🔍 Sharpe storico vs recente (Top 20)")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top20["Rank"].astype(str),
            y=top20["sharpe"],
            name="Sharpe OOS storico",
            marker_color="#4f98a3",
        )
    )
    fig.add_trace(
        go.Bar(
            x=top20["Rank"].astype(str),
            y=top20["recent_sharpe"],
            name="Sharpe OOS recente",
            marker_color="#e8af34",
        )
    )
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e0eaf4"),
        xaxis_title="Rank",
        yaxis_title="Sharpe Ratio",
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Export CSV completo
    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Esporta risultati completi (CSV)",
        data=csv_bytes,
        file_name="wfa_optimizer_results.csv",
        mime="text/csv",
    )
