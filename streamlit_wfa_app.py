import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import json
from collections import Counter

from wfa_core import (
    GIORNI_SETTIMANA,
    OOS_STEP_DAYS,
    DEFAULT_RANKING_METRIC,
    RANKING_METRICS,
    DEFAULT_MAX_PER_GROUP,
    DEFAULT_TOP_N,
    DEFAULT_FULL_WEIGHT_COUNT,
    DEFAULT_FULL_WEIGHT_PCT,
    DEFAULT_BENCH_WEIGHT_PCT,
    DEFAULT_ROC_STEPS,
    DEFAULT_ROC_FILTER_ENABLED,
    DEFAULT_ROC_FILTER_STEPS,
    DEFAULT_EC_ENABLED,
    DEFAULT_EC_CAPITAL_MODE,
    DEFAULT_EC_P80,
    DEFAULT_EC_P90,
    DEFAULT_EC_P95,
    compute_ranking_score,
    passes_roc_filter,
    compute_cusum_banned,
    format_banned_days,
    compute_dd_percentiles_from_is,
    apply_equity_control,
    apply_portfolio_sizing,
)
from wfa_sizing import (
    StrategySizingConfig,
    PortfolioSizingParams,
    clean_money,
    DEFAULT_SIZING_ENABLED,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_DAILY_LOSS_PCT,
    DEFAULT_COMPOUNDING,
    DEFAULT_CAP_PCT,
    DEFAULT_SL_USD,
    DEFAULT_SL_PCT,
    BENCH_WEIGHT_THRESHOLD,
)
from wfa_optimizer_module import render_optimizer_tab, run_wfa_single_windowed, WFAParams

st.set_page_config(page_title="Texano's Walk Forward", layout="wide")

# ─── OZONE THEME CSS ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
  --oz-bg:#0d1117; --oz-panel:#1a2332; --oz-panel2:#1e2d42; --oz-border:#2a3f5f;
  --oz-cyan:#00d4ff; --oz-cyan-dim:#0099bb; --oz-blue:#3b82f6;
  --oz-amber:#f59e0b; --oz-red:#ef4444; --oz-green:#22c55e;
  --oz-text:#e0eaf4; --oz-muted:#7a9abf;
}
html,body,[class*="css"]{ font-family:'Inter','Segoe UI',sans-serif!important; background-color:var(--oz-bg)!important; color:var(--oz-text)!important; }
.stApp,.main .block-container{ background-color:var(--oz-bg)!important; }
h1{ font-size:2rem!important; font-weight:700!important; letter-spacing:-0.5px;
  background:linear-gradient(90deg,#00d4ff 0%,#3b82f6 60%,#818cf8 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; padding-bottom:4px; }
h2{ color:var(--oz-cyan)!important; font-weight:600!important; border-bottom:1px solid var(--oz-border); padding-bottom:6px; margin-bottom:12px; }
h3{ color:#7dd3fc!important; font-weight:500!important; }
[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{ background-color:#101820!important; border-right:1px solid var(--oz-border)!important; }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{ color:var(--oz-cyan)!important; font-size:0.85rem!important; text-transform:uppercase; letter-spacing:0.08em; }
[data-testid="stMetric"]{ background:var(--oz-panel)!important; border:1px solid var(--oz-border)!important; border-radius:8px!important; padding:14px 16px!important; }
[data-testid="stMetricLabel"]{ color:var(--oz-muted)!important; font-size:0.72rem!important; text-transform:uppercase; letter-spacing:0.07em; }
[data-testid="stMetricValue"]{ color:var(--oz-cyan)!important; font-size:1.35rem!important; font-weight:700!important; }
[data-testid="stDataFrame"]{ border:1px solid var(--oz-border)!important; border-radius:8px!important; overflow:hidden; }
.stButton>button,[data-testid="baseButton-secondary"]{
  background:linear-gradient(135deg,#0d2a3f 0%,#0d1f30 100%)!important;
  border:1px solid var(--oz-cyan-dim)!important; color:var(--oz-cyan)!important;
  border-radius:6px!important; font-weight:500!important; font-family:'Inter',sans-serif!important;
  letter-spacing:0.03em; transition:all 0.15s ease; }
.stButton>button:hover{ background:linear-gradient(135deg,#0d3a55 0%,#0d2d44 100%)!important;
  border-color:var(--oz-cyan)!important; box-shadow:0 0 10px rgba(0,212,255,0.25)!important; }
[data-testid="baseButton-primary"]>button,button[kind="primary"]{
  background:linear-gradient(135deg,#005f7a 0%,#007a99 100%)!important;
  border:1.5px solid var(--oz-cyan)!important; color:#ffffff!important;
  font-size:1rem!important; font-weight:700!important; box-shadow:0 0 14px rgba(0,212,255,0.35)!important; }
[data-testid="stDownloadButton"]>button{ background:linear-gradient(135deg,#1a3a52 0%,#122840 100%)!important;
  border:1px solid var(--oz-blue)!important; color:#93c5fd!important; border-radius:6px!important; font-weight:500!important; }
[data-testid="stDownloadButton"]>button:hover{ box-shadow:0 0 10px rgba(59,130,246,0.3)!important; }
[data-testid="stFileUploader"]{ border:1px dashed var(--oz-border)!important; border-radius:8px!important; background:var(--oz-panel)!important; padding:6px; }
[data-testid="stExpander"]{ border:1px solid var(--oz-border)!important; border-radius:8px!important; background:var(--oz-panel)!important; }
[data-testid="stAlert"][data-baseweb="notification"]{ border-radius:8px!important; }
.stInfo{ background:rgba(0,212,255,0.08)!important; border-left:3px solid var(--oz-cyan)!important; }
.stWarning{ background:rgba(245,158,11,0.10)!important; border-left:3px solid var(--oz-amber)!important; }
.stError{ background:rgba(239,68,68,0.10)!important; border-left:3px solid var(--oz-red)!important; }
.stSuccess{ background:rgba(34,197,94,0.08)!important; border-left:3px solid var(--oz-green)!important; }
[data-testid="stSelectbox"]>div>div,
[data-testid="stNumberInput"]>div>div>input,
[data-testid="stTextInput"]>div>div>input{
  background-color:var(--oz-panel2)!important; border:1px solid var(--oz-border)!important;
  color:var(--oz-text)!important; border-radius:6px!important; }
[data-testid="stTabs"] [data-baseweb="tab-list"]{ border-bottom:1px solid var(--oz-border)!important; background:transparent!important; }
[data-testid="stTabs"] [data-baseweb="tab"]{ color:var(--oz-muted)!important; font-weight:500!important; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]{ color:var(--oz-cyan)!important; border-bottom:2px solid var(--oz-cyan)!important; }
[data-testid="stSpinner"]>div{ border-top-color:var(--oz-cyan)!important; }
::-webkit-scrollbar{ width:6px; height:6px; }
::-webkit-scrollbar-track{ background:var(--oz-bg); }
::-webkit-scrollbar-thumb{ background:var(--oz-border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover{ background:var(--oz-cyan-dim); }
</style>
""", unsafe_allow_html=True)

# ─── TITOLO ────────────────────────────────────────────────────────────────────────────
col_title, col_help = st.columns([10, 1])
with col_title:
    st.title("Texano's Walk Forward")
with col_help:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("❓", help="Informazioni sul programma"):
        st.session_state["show_help"] = not st.session_state.get("show_help", False)

if st.session_state.get("show_help", False):
    with st.expander("ℹ️ Come funziona Texano's Walk Forward", expanded=True):
        st.markdown("""
### 📌 Obiettivo
Portafoglio algoritmico **rotazionale** con motore **Walk-Forward Analysis (WFA)** rolling 12 mesi.

---

### ⚙️ Workflow

**1. Preprocessing** — Rimozione strategie *Legendary*, calcolo Net PnL netto commissioni, estrazione weekday.

**2. Mappatura gruppi** — Ogni strategia deve appartenere a un gruppo (configurabile via sidebar).
Esporta e reimporta la configurazione JSON per non perderla dopo il refresh.

**3. Metrica di Ranking IS (configurabile)** — Scegli la metrica per ordinare le strategie nella finestra IS:
- **Omega Ratio** — somma rendimenti positivi / |negativi| (default)
- **Sharpe Ratio** — rendimento medio / deviazione standard, annualizzato
- **Sortino Ratio** — come Sharpe ma usa solo la downside deviation
- **Ulcer Index** — misura la profondità e durata dei drawdown (minore = migliore, ranking invertito)
- **ROC (Rate of Change)** — variazione percentuale del Net PnL cumulativo su N finestre OOS (es. 2×28gg = 56 giorni)

**4. Filtro ROC < 0 (opzionale)** — Esclude le strategie il cui Net PnL cumulativo su una finestra
configurabile (multipli di OOS step) è negativo. Utile per bloccare strategie in trend discendente.

**5. CUSUM Weekday Killer** — Blacklist weekday per combinazioni strategia/giorno anomale.
Se una strategia trada *esclusivamente* nei giorni bannati, viene **scartata completamente** da quella finestra OOS.

**6. Anti-overlap + rotazione frazionata** — Top-N strategie, max per gruppo, pesi pieno/panchina.

**7. OOS filtrato** — Trade filtrati da blacklist CUSUM e ponderati.

**8. Equity Control (opzionale)** — Modulation del rischio per strategia basata sui percentili del DD IS.
Schema a 4 livelli: DD < P80 → size piena; P80–P90 → 50%; P90–P95 → 25%; DD >= P95 → stop totale.
Modalità Trailing (equity cumulativa tra finestre) o Fisso (reset a ogni finestra OOS).

**9. Position Sizing (opzionale)** — Calcola il numero di contratti da allocare per ogni trade OOS.
Basato su capitale, cap% per strategia, stop-loss per contratto e budget giornaliero massimo.
Le strategie in panchina con RiskFactor EC ≤ 0.50 vengono escluse dall'allocazione contratti.

---

### 💾 Backup configurazione
1. **📤 Esporta configurazione JSON** → salva `texano_config.json`
2. Al prossimo accesso: **📥 Importa** → **✅ Applica**
        """)

# ─── COSTANTI LOCALI ──────────────────────────────────────────────────────────────────────────────────
DEFAULT_GROUP_MAPPING = {
    'Condor ZM+': 'Iron Condor',
    'TEST - Iron Condor delle 20:00': 'Iron Condor',
    'Monday Condor': 'Iron Condor',
    'IC Friday - Croccante (optimized)': 'Iron Condor',
    'TEST - DC 2-5': 'RIC / DCS / DC',
    "Giusti's DCS": 'RIC / DCS / DC',
    "Giusti's 1DSC": 'RIC / DCS / DC',
    'TEST - RIC by TradingMonk': 'RIC / DCS / DC',
    "1DSC - Jack's": 'RIC / DCS / DC',
    'TEST - RIC from Hell': 'RIC / DCS / DC',
    'RS Rain orario uscita delta': 'Short Put / Rain',
    'TEST - Sell put 16:00': 'Short Put / Rain',
    'TEST - Sell put 20:00': 'Short Put / Rain',
    'Bull call VIX - Morning': 'Long Call / Butterfly',
    'TEST - Call butterfly': 'Long Call / Butterfly',
    'TEST-Boost Up for RIC': 'Long Call / Butterfly',
    'PDown w/VIX': 'Bearish (PDown)',
    'TEST - PDown': 'Bearish (PDown)'
}

OZONE_LAYOUT = dict(
    plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
    font=dict(family='Inter, Segoe UI, sans-serif', color='#e0eaf4', size=12),
    xaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    yaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    margin=dict(l=50, r=30, t=60, b=50),
)

# ─── FUNZIONI CORE GENERALI ──────────────────────────────────────────────────────────────────────────────────
def calculate_metrics(df, pnl_col='Weighted Net PnL'):
    if df.empty: return {}
    pnl = df[pnl_col]
    total_pnl  = pnl.sum()
    total_trades = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win  = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    daily_pnl = df.groupby(df['Date Closed'].dt.date)[pnl_col].sum()
    cum_pnl   = daily_pnl.cumsum()
    max_dd    = (cum_pnl.cummax() - cum_pnl).max()
    mean_d, std_d = daily_pnl.mean(), daily_pnl.std()
    downside  = daily_pnl[daily_pnl < 0].std()
    sharpe    = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0
    sortino   = (mean_d / downside) * np.sqrt(252) if pd.notna(downside) and downside > 0 else 0
    days = (daily_pnl.index.max() - daily_pnl.index.min()).days
    calmar    = (total_pnl * (365.25 / days)) / max_dd if max_dd > 0 and days > 0 else 0
    recovery_factor = total_pnl / max_dd if max_dd > 0 else np.inf
    return {
        'Total Weighted PnL': total_pnl, 'Total Trades': total_trades, 'Win Rate': win_rate,
        'Profit Factor': profit_factor, 'Avg Win ($)': avg_win, 'Avg Loss ($)': avg_loss,
        'Expectancy ($)': expectancy, 'Max Drawdown ($)': max_dd, 'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino, 'Calmar Ratio': calmar, 'Recovery Factor': recovery_factor,
        'Avg Daily PnL ($)': daily_pnl.mean(), 'Best Day ($)': daily_pnl.max(), 'Worst Day ($)': daily_pnl.min(),
    }

def render_metrics(m, label):
    st.subheader(label)
    if not m:
        st.info("Dati insufficienti.")
        return
    r1 = st.columns(4)
    r1[0].metric("Weighted Net PnL", f"${m['Total Weighted PnL']:,.2f}")
    r1[1].metric("Totale Trade",      f"{int(m['Total Trades'])}")
    r1[2].metric("Win Rate",           f"{m['Win Rate']*100:.1f}%")
    r1[3].metric("Profit Factor",      f"{m['Profit Factor']:.2f}" if m['Profit Factor'] != np.inf else "\u221e")
    r2 = st.columns(4)
    r2[0].metric("Avg Win",      f"${m['Avg Win ($)']:,.2f}")
    r2[1].metric("Avg Loss",     f"${m['Avg Loss ($)']:,.2f}")
    r2[2].metric("Expectancy",   f"${m['Expectancy ($)']:,.2f}")
    r2[3].metric("Max Drawdown", f"${m['Max Drawdown ($)']:,.2f}")
    r3 = st.columns(4)
    r3[0].metric("Sharpe Ratio",    f"{m['Sharpe Ratio']:.2f}")
    r3[1].metric("Sortino Ratio",   f"{m['Sortino Ratio']:.2f}")
    r3[2].metric("Calmar Ratio",    f"{m['Calmar Ratio']:.2f}")
    r3[3].metric("Recovery Factor", f"{m['Recovery Factor']:.2f}" if m['Recovery Factor'] != np.inf else "\u221e")
    r4 = st.columns(3)
    r4[0].metric("Avg Daily PnL", f"${m['Avg Daily PnL ($)']:,.2f}")
    r4[1].metric("Best Day",      f"${m['Best Day ($)']:,.2f}")
    r4[2].metric("Worst Day",     f"${m['Worst Day ($)']:,.2f}")

# ─── FUNZIONI JSON CONFIG ─────────────────────────────────────────────────────────────────────────────────
def build_config_payload():
    return {
        "version": 5,
        "group_names":          st.session_state.get("group_names", []),
        "strategy_mapping":     st.session_state.get("strategy_mapping", {}),
        "max_per_group":        int(st.session_state.get("max_per_group",        DEFAULT_MAX_PER_GROUP)),
        "top_n":                int(st.session_state.get("top_n",                DEFAULT_TOP_N)),
        "full_weight_count":    int(st.session_state.get("full_weight_count",    DEFAULT_FULL_WEIGHT_COUNT)),
        "full_weight_pct":      int(st.session_state.get("full_weight_pct",      DEFAULT_FULL_WEIGHT_PCT)),
        "bench_weight_pct":     int(st.session_state.get("bench_weight_pct",     DEFAULT_BENCH_WEIGHT_PCT)),
        "ranking_metric":       st.session_state.get("ranking_metric",           DEFAULT_RANKING_METRIC),
        "roc_steps":            int(st.session_state.get("roc_steps",            DEFAULT_ROC_STEPS)),
        "roc_filter_enabled":   bool(st.session_state.get("roc_filter_enabled",  DEFAULT_ROC_FILTER_ENABLED)),
        "roc_filter_steps":     int(st.session_state.get("roc_filter_steps",     DEFAULT_ROC_FILTER_STEPS)),
        "ec_enabled":           bool(st.session_state.get("ec_enabled",          DEFAULT_EC_ENABLED)),
        "ec_capital_mode":      st.session_state.get("ec_capital_mode",          DEFAULT_EC_CAPITAL_MODE),
        "ec_p80":               int(st.session_state.get("ec_p80",               DEFAULT_EC_P80)),
        "ec_p90":               int(st.session_state.get("ec_p90",               DEFAULT_EC_P90)),
        "ec_p95":               int(st.session_state.get("ec_p95",               DEFAULT_EC_P95)),
        "sizing_enabled":            bool(st.session_state.get("sizing_enabled",            DEFAULT_SIZING_ENABLED)),
        "sizing_initial_capital":    float(st.session_state.get("sizing_initial_capital",   DEFAULT_INITIAL_CAPITAL)),
        "sizing_max_daily_loss_pct": float(st.session_state.get("sizing_max_daily_loss_pct",DEFAULT_MAX_DAILY_LOSS_PCT)),
        "sizing_compounding":        bool(st.session_state.get("sizing_compounding",         DEFAULT_COMPOUNDING)),
        "sizing_strategy_configs":   st.session_state.get("sizing_strategy_configs",         {}),
    }

def apply_config_payload(payload, all_strategies):
    if not isinstance(payload, dict): raise ValueError("JSON non valido.")
    group_names      = payload.get("group_names", [])
    strategy_mapping = payload.get("strategy_mapping", {})
    if not isinstance(group_names, list):      raise ValueError("group_names deve essere una lista.")
    if not isinstance(strategy_mapping, dict): raise ValueError("strategy_mapping deve essere un dizionario.")
    clean_groups  = sorted({str(g).strip() for g in group_names if str(g).strip()})
    clean_mapping = {str(s).strip(): str(g).strip() for s, g in strategy_mapping.items() if str(s).strip()}
    merged_groups = sorted(set(clean_groups) | {g for g in clean_mapping.values() if g})
    current_mapping = st.session_state.get("strategy_mapping", {}).copy()
    current_mapping.update(clean_mapping)
    for strat in all_strategies: current_mapping.setdefault(strat, "")
    st.session_state["group_names"]       = merged_groups
    st.session_state["strategy_mapping"]  = current_mapping
    st.session_state["max_per_group"]     = max(1, int(payload.get("max_per_group",     DEFAULT_MAX_PER_GROUP)))
    st.session_state["top_n"]             = max(1, int(payload.get("top_n",             DEFAULT_TOP_N)))
    st.session_state["full_weight_count"] = max(0, int(payload.get("full_weight_count", DEFAULT_FULL_WEIGHT_COUNT)))
    st.session_state["full_weight_pct"]   = max(1, min(100, int(payload.get("full_weight_pct",  DEFAULT_FULL_WEIGHT_PCT))))
    st.session_state["bench_weight_pct"]  = max(1, min(100, int(payload.get("bench_weight_pct", DEFAULT_BENCH_WEIGHT_PCT))))
    rm = payload.get("ranking_metric", DEFAULT_RANKING_METRIC)
    st.session_state["ranking_metric"]    = rm if rm in RANKING_METRICS else DEFAULT_RANKING_METRIC
    st.session_state["roc_steps"]         = max(1, int(payload.get("roc_steps",         DEFAULT_ROC_STEPS)))
    st.session_state["roc_filter_enabled"]= bool(payload.get("roc_filter_enabled",       DEFAULT_ROC_FILTER_ENABLED))
    st.session_state["roc_filter_steps"]  = max(1, int(payload.get("roc_filter_steps",   DEFAULT_ROC_FILTER_STEPS)))
    st.session_state["ec_enabled"]        = bool(payload.get("ec_enabled",       DEFAULT_EC_ENABLED))
    ec_mode = payload.get("ec_capital_mode", DEFAULT_EC_CAPITAL_MODE)
    st.session_state["ec_capital_mode"]   = ec_mode if ec_mode in ("Trailing", "Fisso") else DEFAULT_EC_CAPITAL_MODE
    st.session_state["ec_p80"]            = max(1, min(99, int(payload.get("ec_p80", DEFAULT_EC_P80))))
    st.session_state["ec_p90"]            = max(1, min(99, int(payload.get("ec_p90", DEFAULT_EC_P90))))
    st.session_state["ec_p95"]            = max(1, min(99, int(payload.get("ec_p95", DEFAULT_EC_P95))))
    st.session_state["sizing_enabled"]            = bool(payload.get("sizing_enabled",            DEFAULT_SIZING_ENABLED))
    st.session_state["sizing_initial_capital"]    = max(1000.0, float(payload.get("sizing_initial_capital",    DEFAULT_INITIAL_CAPITAL)))
    st.session_state["sizing_max_daily_loss_pct"] = max(0.1, min(20.0, float(payload.get("sizing_max_daily_loss_pct", DEFAULT_MAX_DAILY_LOSS_PCT))))
    st.session_state["sizing_compounding"]        = bool(payload.get("sizing_compounding",         DEFAULT_COMPOUNDING))
    raw_cfgs = payload.get("sizing_strategy_configs", {})
    st.session_state["sizing_strategy_configs"] = raw_cfgs if isinstance(raw_cfgs, dict) else {}

# ─── INIT SESSION STATE ────────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "top_n":              DEFAULT_TOP_N,
    "full_weight_count":  DEFAULT_FULL_WEIGHT_COUNT,
    "full_weight_pct":    DEFAULT_FULL_WEIGHT_PCT,
    "bench_weight_pct":   DEFAULT_BENCH_WEIGHT_PCT,
    "ranking_metric":     DEFAULT_RANKING_METRIC,
    "roc_steps":          DEFAULT_ROC_STEPS,
    "roc_filter_enabled": DEFAULT_ROC_FILTER_ENABLED,
    "roc_filter_steps":   DEFAULT_ROC_FILTER_STEPS,
    "max_per_group":      DEFAULT_MAX_PER_GROUP,
    "ec_enabled":         DEFAULT_EC_ENABLED,
    "ec_capital_mode":    DEFAULT_EC_CAPITAL_MODE,
    "ec_p80":             DEFAULT_EC_P80,
    "ec_p90":             DEFAULT_EC_P90,
    "ec_p95":             DEFAULT_EC_P95,
    "sizing_enabled":            DEFAULT_SIZING_ENABLED,
    "sizing_initial_capital":    DEFAULT_INITIAL_CAPITAL,
    "sizing_max_daily_loss_pct": DEFAULT_MAX_DAILY_LOSS_PCT,
    "sizing_compounding":        DEFAULT_COMPOUNDING,
    "sizing_strategy_configs":   {},
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── UPLOAD CSV ───────────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

# ─── NORMALIZZAZIONE COLONNA Net PnL ────────────────────────────────────────────────────────────────────────────────
_NET_PNL_ALIASES = {
    'net pnl': 'Net PnL', 'netpnl': 'Net PnL', 'net p&l': 'Net PnL',
    'netp&l': 'Net PnL', 'net pl': 'Net PnL', 'netpl': 'Net PnL',
    'net profit': 'Net PnL', 'net profit/loss': 'Net PnL',
    'pnl': 'Net PnL', 'p&l': 'Net PnL',
}
_COMMISSION_COLS = [
    'Opening Commissions + Fees', 'Closing Commissions + Fees',
    'Commissions + Fees', 'Commission', 'Fees',
]
_PL_RAW_COLS = ['P/L', 'P/L $', 'Gross P/L', 'Gross PnL', 'Profit/Loss']

def _normalize_net_pnl_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'Net PnL' in df.columns:
        return df
    col_lower_map  = {c.strip().lower(): c for c in df.columns}
    col_nospace_map = {c.strip().lower().replace(' ', ''): c for c in df.columns}
    for alias, canonical in _NET_PNL_ALIASES.items():
        if alias in col_lower_map:
            return df.rename(columns={col_lower_map[alias]: canonical})
        alias_nospace = alias.replace(' ', '')
        if alias_nospace in col_nospace_map:
            return df.rename(columns={col_nospace_map[alias_nospace]: canonical})
    pl_col = next((c for c in _PL_RAW_COLS if c in df.columns), None)
    if pl_col is not None:
        df = df.copy()
        net = df[pl_col].apply(clean_money)
        for comm_col in _COMMISSION_COLS:
            if comm_col in df.columns:
                net = net - df[comm_col].apply(clean_money).abs()
        df['Net PnL'] = net
        return df
    return df

@st.cache_data(show_spinner=False)
def preprocess_df(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    df = _normalize_net_pnl_column(df)
    if 'Net PnL' not in df.columns:
        raise ValueError(
            f"Colonna 'Net PnL' non trovata nel CSV. "
            f"Colonne disponibili: {list(df.columns)}"
        )
    df = df[~df['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()
    return df


def _format_ban_freq(banned_freq: Counter) -> str:
    """Formatta il Counter dei ban storici come 'Mon(3x) Wed(1x)' ecc."""
    if not banned_freq:
        return "Nessuno"
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    parts = []
    for day_idx in sorted(banned_freq.keys()):
        name = day_names[day_idx] if day_idx < len(day_names) else str(day_idx)
        parts.append(f"{name}({banned_freq[day_idx]}x)")
    return " ".join(parts)


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    try:
        df_raw = preprocess_df(file_bytes)
    except ValueError as _e:
        st.error(f"❌ Errore nel CSV caricato: {_e}")
        st.stop()
    all_strategies_in_file = sorted(df_raw['Strategy'].dropna().unique().tolist())

    if 'strategy_mapping' not in st.session_state:
        st.session_state['strategy_mapping'] = DEFAULT_GROUP_MAPPING.copy()
    if 'group_names' not in st.session_state:
        st.session_state['group_names'] = sorted(set(DEFAULT_GROUP_MAPPING.values()))
    for strat in all_strategies_in_file:
        if strat not in st.session_state['strategy_mapping']:
            st.session_state['strategy_mapping'][strat] = ''

    # ─── SIDEBAR ─────────────────────────────────────────────────────────────────────────────────────
    st.sidebar.header("\u2699\ufe0f Configurazione")

    st.sidebar.subheader("\U0001f4be Backup configurazione")
    st.sidebar.caption("Esporta per salvare tutti i parametri. Reimporta dopo ogni refresh.")
    config_json_str = json.dumps(build_config_payload(), ensure_ascii=False, indent=2)
    st.sidebar.download_button(label="\U0001f4e4 Esporta configurazione JSON",
        data=config_json_str.encode("utf-8"), file_name="texano_config.json", mime="application/json")
    uploaded_config = st.sidebar.file_uploader("\U0001f4e5 Importa configurazione JSON", type=["json"], key="config_json_uploader")
    if uploaded_config is not None:
        try:
            parsed = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.session_state["pending_config_payload"] = parsed
            st.sidebar.success("\u2705 JSON caricato. Premi 'Applica' per attivarlo.")
        except Exception as e:
            st.sidebar.error(f"JSON non valido: {e}")
    if st.sidebar.button("\u2705 Applica configurazione JSON"):
        if "pending_config_payload" not in st.session_state:
            st.sidebar.warning("Carica prima un file JSON.")
        else:
            try:
                apply_config_payload(st.session_state["pending_config_payload"], all_strategies_in_file)
                del st.session_state["pending_config_payload"]
                st.sidebar.success("Configurazione applicata!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Errore: {e}")
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f3db\ufe0f Parametri WFA")
    st.sidebar.number_input("Numero strategie selezionate (Top-N)", min_value=1, max_value=20, step=1, help="Default: 7", key="top_n")
    st.sidebar.number_input("Strategie con peso pieno (rank 1\u2026N)", min_value=0, max_value=int(st.session_state["top_n"]), step=1, key="full_weight_count")
    st.sidebar.number_input("Peso pieno (%)", min_value=1, max_value=100, step=5, key="full_weight_pct")
    st.sidebar.number_input("Peso panchina (%)", min_value=1, max_value=100, step=5, key="bench_weight_pct")
    _bench_count = int(st.session_state["top_n"]) - int(st.session_state["full_weight_count"])
    st.sidebar.markdown(
        f"**Schema:** {int(st.session_state['full_weight_count'])}\u00d7"
        f"{int(st.session_state['full_weight_pct'])}% + "
        f"{_bench_count}\u00d7{int(st.session_state['bench_weight_pct'])}%"
    )
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f3c6 Metrica di Ranking IS")
    ranking_metric_idx = RANKING_METRICS.index(st.session_state["ranking_metric"]) if st.session_state["ranking_metric"] in RANKING_METRICS else 0
    st.sidebar.selectbox("Metrica per ordinare le strategie nella finestra IS", options=RANKING_METRICS, index=ranking_metric_idx,
        help="Omega Ratio: default. Ulcer Index: minore = migliore. ROC: PnL netto su N finestre OOS.", key="ranking_metric")
    if st.session_state["ranking_metric"] == "ROC":
        st.sidebar.number_input(f"Finestra ROC (multipli di {OOS_STEP_DAYS}gg)", min_value=1, max_value=24, step=1,
            help=f"Es. 2 = finestra di {2*OOS_STEP_DAYS} giorni.", key="roc_steps")
        st.sidebar.caption(f"\U0001f4c5 Finestra ROC attiva: **{int(st.session_state['roc_steps']) * OOS_STEP_DAYS} giorni**")
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f6ab Filtro ROC < 0")
    st.sidebar.toggle("Attiva filtro ROC < 0",
        help="Se attivo, esclude le strategie con Net PnL negativo sulla finestra configurata.", key="roc_filter_enabled")
    if st.session_state["roc_filter_enabled"]:
        st.sidebar.number_input(f"Finestra filtro ROC (multipli di {OOS_STEP_DAYS}gg)", min_value=1, max_value=24, step=1, key="roc_filter_steps")
        st.sidebar.caption(f"\U0001f4c5 Finestra filtro ROC: **{int(st.session_state['roc_filter_steps']) * OOS_STEP_DAYS} giorni**")
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f4c9 Equity Control")
    st.sidebar.toggle("Attiva Equity Control",
        help="Modula il rischio per strategia in base al drawdown OOS corrente rispetto ai percentili IS.", key="ec_enabled")
    if st.session_state["ec_enabled"]:
        ec_mode_options = ["Trailing", "Fisso"]
        ec_mode_idx = ec_mode_options.index(st.session_state["ec_capital_mode"]) if st.session_state["ec_capital_mode"] in ec_mode_options else 0
        st.sidebar.selectbox("Modalita capitale", options=ec_mode_options, index=ec_mode_idx,
            help="Trailing: equity cumulativa. Fisso: reset a ogni finestra OOS.", key="ec_capital_mode")
        st.sidebar.number_input("Percentile DD soglia P80 (half size)", min_value=50, max_value=95, step=5,
            help="Sotto questa soglia DD: RiskFactor=1.0.", key="ec_p80")
        st.sidebar.number_input("Percentile DD soglia P90 (quarter size)",
            min_value=int(st.session_state["ec_p80"]) + 1, max_value=98, step=1,
            help="Tra P80 e P90: RiskFactor=0.5.", key="ec_p90")
        st.sidebar.number_input("Percentile DD soglia P95 (stop totale)",
            min_value=int(st.session_state["ec_p90"]) + 1, max_value=99, step=1,
            help="Tra P90 e P95: RiskFactor=0.25. Oltre P95: RiskFactor=0.0.", key="ec_p95")
        st.sidebar.caption(
            f"**Schema attivo:** DD<P{int(st.session_state['ec_p80'])}=100% "
            f"| P{int(st.session_state['ec_p80'])}-P{int(st.session_state['ec_p90'])}=50% "
            f"| P{int(st.session_state['ec_p90'])}-P{int(st.session_state['ec_p95'])}=25% "
            f"| DD>=P{int(st.session_state['ec_p95'])}=0%"
        )
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f4b0 Position Sizing")
    st.sidebar.toggle("Attiva Position Sizing",
        help="Calcola il numero di contratti per ogni trade OOS. Richiede la colonna 'Premium' nel CSV.", key="sizing_enabled")
    if st.session_state["sizing_enabled"]:
        st.sidebar.number_input("Capitale iniziale ($)", min_value=1_000, max_value=10_000_000, step=1_000,
            help=f"Default: ${DEFAULT_INITIAL_CAPITAL:,.0f}", key="sizing_initial_capital")
        st.sidebar.number_input("Max perdita giornaliera (%)", min_value=0.1, max_value=20.0, step=0.1,
            help=f"Budget di rischio giornaliero come % del capitale. Default: {DEFAULT_MAX_DAILY_LOSS_PCT}%", key="sizing_max_daily_loss_pct")
        st.sidebar.toggle("Compounding (reinvestimento profitti)",
            help="Se attivo, il capitale si aggiorna con i profitti/perdite finestra per finestra.", key="sizing_compounding")
        with st.sidebar.expander("⚙️ Parametri per strategia", expanded=False):
            st.caption("Cap %: % del capitale per questa strategia. SL USD: stop-loss per contratto. SL %: stop come % del premio.")
            current_cfgs = st.session_state.get("sizing_strategy_configs", {})
            updated_cfgs = {}
            for strat in all_strategies_in_file:
                cfg = current_cfgs.get(strat, {})
                st.markdown(f"**{strat}**")
                c1, c2, c3 = st.columns(3)
                cap    = c1.number_input("Cap %",  min_value=0.1, max_value=100.0, step=0.5, value=float(cfg.get("cap_pct", DEFAULT_CAP_PCT)), key=f"sz_cap_{strat}")
                sl_usd = c2.number_input("SL $",   min_value=1.0, max_value=50_000.0, step=10.0, value=float(cfg.get("sl_usd", DEFAULT_SL_USD)), key=f"sz_slusd_{strat}")
                sl_pct = c3.number_input("SL %",   min_value=0.0, max_value=100.0, step=0.5, value=float(cfg.get("sl_pct", DEFAULT_SL_PCT)), key=f"sz_slpct_{strat}")
                updated_cfgs[strat] = {"cap_pct": cap, "sl_usd": sl_usd, "sl_pct": sl_pct}
            if st.button("\u2705 Salva parametri sizing"):
                st.session_state["sizing_strategy_configs"] = updated_cfgs
                st.sidebar.success("Parametri sizing salvati — esporta il JSON!")
                st.rerun()
    st.sidebar.divider()

    st.sidebar.subheader("Limite per gruppo")
    st.sidebar.number_input("Max strategie stesso gruppo", min_value=1, max_value=int(st.session_state["top_n"]), step=1, help="Default 2", key="max_per_group")
    st.sidebar.divider()

    st.sidebar.subheader("Gestione gruppi")
    new_group_name = st.sidebar.text_input("Nome nuovo gruppo", key="new_group_name")
    if st.sidebar.button("\u2795 Crea gruppo"):
        clean_name = new_group_name.strip()
        if clean_name:
            if clean_name not in st.session_state['group_names']:
                st.session_state['group_names'].append(clean_name)
                st.session_state['group_names'] = sorted(st.session_state['group_names'])
                st.sidebar.success(f"Gruppo '{clean_name}' creato")
                st.rerun()
            else:
                st.sidebar.warning("Gruppo gi\u00e0 esistente")
        else:
            st.sidebar.warning("Inserisci un nome valido")
    if st.session_state['group_names']:
        group_to_rename = st.sidebar.selectbox("Gruppo da rinominare", options=st.session_state['group_names'], key="group_to_rename")
        renamed_group = st.sidebar.text_input("Nuovo nome", key="renamed_group")
        if st.sidebar.button("\u270f\ufe0f Rinomina gruppo"):
            old_name, new_name = group_to_rename, renamed_group.strip()
            if not new_name:
                st.sidebar.warning("Inserisci un nome valido")
            elif new_name in st.session_state['group_names'] and new_name != old_name:
                st.sidebar.warning("Nome gi\u00e0 esistente")
            else:
                st.session_state['group_names'] = [new_name if g == old_name else g for g in st.session_state['group_names']]
                st.session_state['strategy_mapping'] = {s: (new_name if g == old_name else g) for s, g in st.session_state['strategy_mapping'].items()}
                st.sidebar.success(f"'{old_name}' \u2192 '{new_name}'")
                st.rerun()
    st.sidebar.divider()

    st.sidebar.subheader("Assegnazione strategie")
    group_options = [''] + st.session_state['group_names']
    with st.sidebar.expander("\U0001f4cb Modifica gruppo di ogni strategia", expanded=True):
        updated_mapping = {}
        for strat in all_strategies_in_file:
            cur = st.session_state['strategy_mapping'].get(strat, '')
            idx = group_options.index(cur) if cur in group_options else 0
            sel = st.selectbox(label=strat, options=group_options, index=idx, key=f"group_{strat}")
            updated_mapping[strat] = sel
        if st.button("\u2705 Salva assegnazioni"):
            st.session_state['strategy_mapping'] = updated_mapping
            st.sidebar.success("Assegnazioni aggiornate \u2014 esporta il JSON!")
            st.rerun()

    strategy_mapping = st.session_state['strategy_mapping']
    still_unmapped = [s for s in all_strategies_in_file if not strategy_mapping.get(s, '').strip()]
    if still_unmapped:
        st.error(f"\U0001f6d1 BLOCCO DI SICUREZZA: Strategie senza gruppo: {', '.join(still_unmapped)}")
        st.info("Apri la sidebar (\u25b6), assegna un gruppo a ogni strategia e salva.")
        st.stop()

    top_n             = int(st.session_state["top_n"])
    full_weight_count = int(st.session_state["full_weight_count"])
    full_weight_pct   = int(st.session_state["full_weight_pct"])
    bench_weight_pct  = int(st.session_state["bench_weight_pct"])
    max_per_group     = int(st.session_state["max_per_group"]