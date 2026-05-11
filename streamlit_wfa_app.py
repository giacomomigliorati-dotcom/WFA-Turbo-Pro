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
from wfa_optimizer_module import render_optimizer_tab, run_wfa_single_windowed, WFAParams
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

st.set_page_config(page_title="Texano's Walk Forward", layout="wide")

# ─── OZONE THEME CSS ─────────────────────────────────────────────────────────
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

# ─── TITOLO ────────────────────────────────────────────────────────────────
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

**6. Anti-overlap + rotazione frazionata** — Top-N strategie, max per gruppo, pesi pieno/panchina.

**7. OOS filtrato** — Trade filtrati da blacklist CUSUM e ponderati.

---

### 💾 Backup configurazione
1. **📤 Esporta configurazione JSON** → salva `texano_config.json`
2. Al prossimo accesso: **📥 Importa** → **✅ Applica**
        """)

# ─── COSTANTI LOCALI (solo quelle non in wfa_core) ───────────────────────────
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

# ─── FUNZIONI CORE GENERALI ───────────────────────────────────────────────────
def calculate_metrics(df):
    if df.empty: return {}
    pnl = df['Weighted Net PnL']
    total_pnl  = pnl.sum()
    total_trades = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win  = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    daily_pnl = df.groupby(df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
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

# ─── FUNZIONI JSON CONFIG ─────────────────────────────────────────────────────
def build_config_payload():
    return {
        "version": 3,
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
    st.session_state["ec_enabled"] = bool(payload.get("ec_enabled", DEFAULT_EC_ENABLED))
    ec_mode = payload.get("ec_capital_mode", DEFAULT_EC_CAPITAL_MODE)
    st.session_state["ec_capital_mode"] = ec_mode if ec_mode in ("Trailing", "Fisso") else DEFAULT_EC_CAPITAL_MODE
    st.session_state["ec_p80"] = max(1, min(99, int(payload.get("ec_p80", DEFAULT_EC_P80))))
    st.session_state["ec_p90"] = max(1, min(99, int(payload.get("ec_p90", DEFAULT_EC_P90))))
    st.session_state["ec_p95"] = max(1, min(99, int(payload.get("ec_p95", DEFAULT_EC_P95))))
    st.session_state["sizing_enabled"]            = bool(payload.get("sizing_enabled",            DEFAULT_SIZING_ENABLED))
    st.session_state["sizing_initial_capital"]    = max(1000.0, float(payload.get("sizing_initial_capital",    DEFAULT_INITIAL_CAPITAL)))
    st.session_state["sizing_max_daily_loss_pct"] = max(0.1,   min(20.0, float(payload.get("sizing_max_daily_loss_pct", DEFAULT_MAX_DAILY_LOSS_PCT))))
    st.session_state["sizing_compounding"]        = bool(payload.get("sizing_compounding",         DEFAULT_COMPOUNDING))
    raw_cfgs = payload.get("sizing_strategy_configs", {})
    st.session_state["sizing_strategy_configs"] = raw_cfgs if isinstance(raw_cfgs, dict) else {}

# ─── INIT SESSION STATE ───────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "top_n":              DEFAULT_TOP_N,
    "full_weight_count":  DEFAULT_FULL_WEIGHT_COUNT,
    "full_weight_pct":    DEFAULT_FULL_WEIGHT_PCT,
    "bench_weight_pct":   DEFAULT_BENCH_WEIGHT_PCT,
    "ranking_metric":     DEFAULT_RANKING_METRIC,
    "roc_steps":          DEFAULT_ROC_STEPS,
    "roc_filter_enabled": DEFAULT_ROC_FILTER_ENABLED,
    "roc_filter_steps":   DEFAULT_ROC_FILTER_STEPS,
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
    "max_per_group":      DEFAULT_MAX_PER_GROUP,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── UPLOAD CSV ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

@st.cache_data(show_spinner=False)
def preprocess_df(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    df = df[~df['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()
    return df

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df_raw = preprocess_df(file_bytes)
    all_strategies_in_file = sorted(df_raw['Strategy'].dropna().unique().tolist())

    if 'strategy_mapping' not in st.session_state:
        st.session_state['strategy_mapping'] = DEFAULT_GROUP_MAPPING.copy()
    if 'group_names' not in st.session_state:
        st.session_state['group_names'] = sorted(set(DEFAULT_GROUP_MAPPING.values()))
    for strat in all_strategies_in_file:
        if strat not in st.session_state['strategy_mapping']:
            st.session_state['strategy_mapping'][strat] = ''

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────
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

    # ── Parametri WFA — FIX: key= nativo, nessuna scrittura manuale su session_state ──
    st.sidebar.subheader("\U0001f3db\ufe0f Parametri WFA")

    st.sidebar.number_input(
        "Numero strategie selezionate (Top-N)",
        min_value=1, max_value=20, step=1,
        help="Default: 7",
        key="top_n",
    )

    st.sidebar.number_input(
        "Strategie con peso pieno (rank 1\u2026N)",
        min_value=0, max_value=int(st.session_state["top_n"]),
        step=1,
        key="full_weight_count",
    )

    st.sidebar.number_input(
        "Peso pieno (%)",
        min_value=1, max_value=100, step=5,
        key="full_weight_pct",
    )

    st.sidebar.number_input(
        "Peso panchina (%)",
        min_value=1, max_value=100, step=5,
        key="bench_weight_pct",
    )

    _bench_count = int(st.session_state["top_n"]) - int(st.session_state["full_weight_count"])
    st.sidebar.markdown(
        f"**Schema:** {int(st.session_state['full_weight_count'])}\u00d7"
        f"{int(st.session_state['full_weight_pct'])}% + "
        f"{_bench_count}\u00d7{int(st.session_state['bench_weight_pct'])}%"
    )
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f3c6 Metrica di Ranking IS")
    ranking_metric_idx = RANKING_METRICS.index(st.session_state["ranking_metric"]) \
        if st.session_state["ranking_metric"] in RANKING_METRICS else 0
    st.sidebar.selectbox(
        "Metrica per ordinare le strategie nella finestra IS",
        options=RANKING_METRICS, index=ranking_metric_idx,
        help="Omega Ratio: default. Ulcer Index: minore = migliore. ROC: PnL netto su N finestre OOS.",
        key="ranking_metric",
    )

    if st.session_state["ranking_metric"] == "ROC":
        st.sidebar.number_input(
            f"Finestra ROC (multipli di {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, step=1,
            help=f"Es. 2 = finestra di {2*OOS_STEP_DAYS} giorni.",
            key="roc_steps",
        )
        st.sidebar.caption(f"\U0001f4c5 Finestra ROC attiva: **{int(st.session_state['roc_steps']) * OOS_STEP_DAYS} giorni**")
    st.sidebar.divider()

    st.sidebar.subheader("\U0001f6ab Filtro ROC < 0")
    st.sidebar.toggle(
        "Attiva filtro ROC < 0",
        help="Se attivo, esclude le strategie con Net PnL negativo sulla finestra configurata.",
        key="roc_filter_enabled",
    )

    if st.session_state["roc_filter_enabled"]:
        st.sidebar.number_input(
            f"Finestra filtro ROC (multipli di {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, step=1,
            key="roc_filter_steps",
        )
        st.sidebar.caption(
            f"\U0001f4c5 Finestra filtro ROC: **{int(st.session_state['roc_filter_steps']) * OOS_STEP_DAYS} giorni**"
        )

    st.sidebar.subheader("📉 Equity Control")
    st.sidebar.toggle(
        "Attiva Equity Control",
        help="Modula il rischio per strategia in base al drawdown OOS corrente rispetto ai percentili IS.",
        key="ec_enabled",
    )
    if st.session_state["ec_enabled"]:
        ec_mode_options = ["Trailing", "Fisso"]
        ec_mode_idx = ec_mode_options.index(st.session_state["ec_capital_mode"]) \
            if st.session_state["ec_capital_mode"] in ec_mode_options else 0
        st.sidebar.selectbox(
            "Modalita capitale",
            options=ec_mode_options,
            index=ec_mode_idx,
            help="Trailing: equity cumulativa. Fisso: reset a ogni finestra OOS.",
            key="ec_capital_mode",
        )
        st.sidebar.number_input(
            "Percentile DD soglia P80 (half size)",
            min_value=50, max_value=95, step=5,
            help="Sotto questa soglia DD: RiskFactor=1.0.",
            key="ec_p80",
        )
        st.sidebar.number_input(
            "Percentile DD soglia P90 (quarter size)",
            min_value=int(st.session_state["ec_p80"]) + 1, max_value=98, step=1,
            help="Tra P80 e P90: RiskFactor=0.5.",
            key="ec_p90",
        )
        st.sidebar.number_input(
            "Percentile DD soglia P95 (stop totale)",
            min_value=int(st.session_state["ec_p90"]) + 1, max_value=99, step=1,
            help="Tra P90 e P95: RiskFactor=0.25. Oltre P95: RiskFactor=0.0.",
            key="ec_p95",
        )
        st.sidebar.caption(
            f"**Schema attivo:** DD<P{int(st.session_state['ec_p80'])}=100% "
            f"| P{int(st.session_state['ec_p80'])}-P{int(st.session_state['ec_p90'])}=50% "
            f"| P{int(st.session_state['ec_p90'])}-P{int(st.session_state['ec_p95'])}=25% "
            f"| DD>=P{int(st.session_state['ec_p95'])}=0%"
        )

    st.sidebar.subheader("💰 Position Sizing")
    st.sidebar.toggle(
        "Attiva Position Sizing",
        help="Calcola il numero di contratti per ogni trade OOS. Richiede la colonna 'Premium' nel CSV.",
        key="sizing_enabled",
    )
    if st.session_state["sizing_enabled"]:
        st.sidebar.number_input(
            "Capitale iniziale ($)",
            min_value=1_000, max_value=10_000_000, step=1_000,
            help=f"Default: ${DEFAULT_INITIAL_CAPITAL:,.0f}",
            key="sizing_initial_capital",
        )
        st.sidebar.number_input(
            "Max perdita giornaliera (%)",
            min_value=0.1, max_value=20.0, step=0.1,
            help=f"Budget di rischio giornaliero come % del capitale. Default: {DEFAULT_MAX_DAILY_LOSS_PCT}%",
            key="sizing_max_daily_loss_pct",
        )
        st.sidebar.toggle(
            "Compounding (reinvestimento profitti)",
            help="Se attivo, il capitale si aggiorna con i profitti/perdite finestra per finestra.",
            key="sizing_compounding",
        )
        with st.sidebar.expander("⚙️ Parametri per strategia", expanded=False):
            st.caption("Cap %: % del capitale per questa strategia. SL USD: stop-loss per contratto. SL %: stop come % del premio.")
            current_cfgs = st.session_state.get("sizing_strategy_configs", {})
            updated_cfgs = {}
            for strat in all_strategies_in_file:
                cfg = current_cfgs.get(strat, {})
                st.markdown(f"**{strat}**")
                c1, c2, c3 = st.columns(3)
                cap    = c1.number_input("Cap %",  min_value=0.1, max_value=100.0, step=0.5,
                                         value=float(cfg.get("cap_pct", DEFAULT_CAP_PCT)), key=f"sz_cap_{strat}")
                sl_usd = c2.number_input("SL $",   min_value=1.0,  max_value=50_000.0, step=10.0,
                                         value=float(cfg.get("sl_usd", DEFAULT_SL_USD)), key=f"sz_slusd_{strat}")
                sl_pct = c3.number_input("SL %",   min_value=0.0,  max_value=100.0, step=0.5,
                                         value=float(cfg.get("sl_pct", DEFAULT_SL_PCT)), key=f"sz_slpct_{strat}")
                updated_cfgs[strat] = {"cap_pct": cap, "sl_usd": sl_usd, "sl_pct": sl_pct}
            if st.button("✅ Salva parametri sizing"):
                st.session_state["sizing_strategy_configs"] = updated_cfgs
                st.sidebar.success("Parametri sizing salvati — esporta il JSON!")
                st.rerun()
    st.sidebar.divider()

    st.sidebar.subheader("Limite per gruppo")
    st.sidebar.number_input(
        "Max strategie stesso gruppo",
        min_value=1, max_value=int(st.session_state["top_n"]),
        step=1,
        help="Default 2",
        key="max_per_group",
    )
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
        group_to_rename = st.sidebar.selectbox("Gruppo da rinominare",
            options=st.session_state['group_names'], key="group_to_rename")
        renamed_group = st.sidebar.text_input("Nuovo nome", key="renamed_group")
        if st.sidebar.button("\u270f\ufe0f Rinomina gruppo"):
            old_name, new_name = group_to_rename, renamed_group.strip()
            if not new_name:
                st.sidebar.warning("Inserisci un nome valido")
            elif new_name in st.session_state['group_names'] and new_name != old_name:
                st.sidebar.warning("Nome gi\u00e0 esistente")
            else:
                st.session_state['group_names'] = [new_name if g == old_name else g
                    for g in st.session_state['group_names']]
                st.session_state['strategy_mapping'] = {
                    s: (new_name if g == old_name else g)
                    for s, g in st.session_state['strategy_mapping'].items()}
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

    # Alias leggibili per il resto del codice
    top_n            = int(st.session_state["top_n"])
    full_weight_count= int(st.session_state["full_weight_count"])
    full_weight_pct  = int(st.session_state["full_weight_pct"])
    bench_weight_pct = int(st.session_state["bench_weight_pct"])
    max_per_group    = int(st.session_state["max_per_group"])
    ranking_metric   = st.session_state["ranking_metric"]
    roc_steps        = int(st.session_state["roc_steps"])
    roc_filter_enabled = bool(st.session_state["roc_filter_enabled"])
    roc_filter_steps = int(st.session_state["roc_filter_steps"])
    ec_enabled       = bool(st.session_state["ec_enabled"])
    ec_capital_mode  = st.session_state["ec_capital_mode"]
    ec_p80           = int(st.session_state["ec_p80"])
    ec_p90           = int(st.session_state["ec_p90"])
    ec_p95           = int(st.session_state["ec_p95"])
    sizing_enabled            = bool(st.session_state["sizing_enabled"])
    sizing_initial_capital    = float(st.session_state["sizing_initial_capital"])
    sizing_max_daily_loss_pct = float(st.session_state["sizing_max_daily_loss_pct"])
    sizing_compounding        = bool(st.session_state["sizing_compounding"])
    sizing_strategy_configs   = st.session_state.get("sizing_strategy_configs", {})
    bench_count      = top_n - full_weight_count

    # ─── TAB PRINCIPALI ───────────────────────────────────────────────────────
    tab_wfa, tab_opt = st.tabs(["\U0001f4ca Walk-Forward", "\u2699\ufe0f Ottimizzatore"])

    with tab_wfa:
        st.markdown("---")
        col_launch, col_info = st.columns([3, 7])
        with col_launch:
            launch = st.button("\u25b6 Lancia simulazione WFA", type="primary", use_container_width=True)
        with col_info:
            roc_filter_label = (
                f" &nbsp;|&nbsp; \U0001f6ab ROC filter: <b style='color:#ef4444'>"
                f"{roc_filter_steps*OOS_STEP_DAYS}gg</b>"
            ) if roc_filter_enabled else ""
            ec_label = (
                f" &nbsp;|&nbsp; 📉 EC: <b style='color:#22c55e'>{ec_capital_mode}</b>"
            ) if ec_enabled else ""
            sizing_label = (
                f" &nbsp;|&nbsp; 💰 Sizing: <b style='color:#f59e0b'>ON</b>"
                f" | Capital: <b style='color:#f59e0b'>${sizing_initial_capital:,.0f}</b>"
                f" | MaxDD: <b style='color:#f59e0b'>{sizing_max_daily_loss_pct:.1f}%</b>"
            ) if sizing_enabled else ""
            ranking_label_color = "#f59e0b" if ranking_metric != "Omega Ratio" else "#00d4ff"
            st.markdown(
                f"<div style='padding-top:10px; color:#7a9abf; font-size:0.88rem;'>"
                f"Ranking: <b style='color:{ranking_label_color}'>{ranking_metric}</b>"
                + (f" ({roc_steps*OOS_STEP_DAYS}gg)" if ranking_metric=="ROC" else "")
                + f" &nbsp;|&nbsp; Top-N: <b style='color:#00d4ff'>{top_n}</b>"
                f" &nbsp;|&nbsp; Pesi: <b style='color:#00d4ff'>{full_weight_count}\u00d7{full_weight_pct}%</b>"
                f" + <b style='color:#f59e0b'>{bench_count}\u00d7{bench_weight_pct}%</b>"
                f" &nbsp;|&nbsp; Max/gr: <b style='color:#00d4ff'>{max_per_group}</b>"
                + roc_filter_label + ec_label + sizing_label + "</div>",
                unsafe_allow_html=True
            )

        if launch:
            _top_n    = top_n
            _fwc      = full_weight_count
            _fwp      = full_weight_pct / 100.0
            _bwp      = bench_weight_pct / 100.0
            _mpg      = max_per_group
            _sm       = dict(strategy_mapping)
            _metric   = ranking_metric
            _roc_s    = roc_steps
            _roc_fe   = roc_filter_enabled
            _roc_fs   = roc_filter_steps

            with st.spinner("Elaborazione Walk-Forward in corso..."):
                df_filtered = df_raw.copy()
                df_filtered['Date Opened'] = pd.to_datetime(df_filtered['Date Opened'])
                df_filtered['Date Closed'] = pd.to_datetime(df_filtered['Date Closed'])
                df_filtered['weekday_open'] = df_filtered['Date Opened'].dt.weekday
                df_filtered['P/L'] = df_filtered['P/L'].apply(clean_money)
                df_filtered['P/L %'] = df_filtered['P/L %'].apply(
                    lambda x: float(str(x).replace('%','').replace(',','')) if pd.notna(x) else np.nan)
                df_filtered['Opening Commissions + Fees'] = df_filtered['Opening Commissions + Fees'].fillna(0).apply(clean_money)
                df_filtered['Closing Commissions + Fees'] = df_filtered['Closing Commissions + Fees'].fillna(0).apply(clean_money)
                df_filtered['Net PnL'] = (
                    df_filtered['P/L']
                    - df_filtered['Opening Commissions + Fees']
                    - df_filtered['Closing Commissions + Fees']
                )
                df_filtered = df_filtered.sort_values('Date Closed').reset_index(drop=True)

                if ec_enabled:
                    dd_thresholds = {}
                    for strat, strat_hist in df_filtered.groupby("Strategy"):
                        dd_thresholds[strat] = compute_dd_percentiles_from_is(
                            strat_hist,
                            p80=ec_p80,
                            p90=ec_p90,
                            p95=ec_p95,
                        )
                else:
                    dd_thresholds = {}

                min_date  = df_filtered['Date Closed'].min()
                max_date  = df_filtered['Date Closed'].max()
                current_oos_start = min_date + pd.DateOffset(months=12)
                oos_results, historical_allocations, cusum_exclusions_log, roc_filter_log = [], [], [], []
                equity_state = {}
                sizing_params = PortfolioSizingParams(
                    initial_capital=sizing_initial_capital,
                    max_daily_loss_pct=sizing_max_daily_loss_pct,
                    compounding=sizing_compounding,
                )
                current_capital = sizing_params.initial_capital

                while current_oos_start <= max_date:
                    current_oos_end  = current_oos_start + pd.Timedelta(days=OOS_STEP_DAYS)
                    current_is_start = current_oos_start - pd.DateOffset(months=12)
                    is_data  = df_filtered[(df_filtered['Date Closed'] >= current_is_start)
                                          & (df_filtered['Date Closed'] < current_oos_start)].copy()
                    oos_data = df_filtered[(df_filtered['Date Closed'] >= current_oos_start)
                                          & (df_filtered['Date Closed'] < current_oos_end)].copy()
                    if is_data.empty:
                        current_oos_start = current_oos_end
                        continue

                    valid_strats = is_data['Strategy'].value_counts()
                    valid_strats = valid_strats[valid_strats >= 5].index.tolist()
                    is_metrics = []

                    for strat in valid_strats:
                        if strat not in _sm: continue
                        strat_is = is_data[is_data['Strategy'] == strat].copy()
                        if _roc_fe and not passes_roc_filter(strat_is, _roc_fs):
                            roc_filter_log.append({'OOS Start': current_oos_start, 'Strategy': strat,
                                'Motivo': f"ROC < 0 su {_roc_fs*OOS_STEP_DAYS}gg"})
                            continue
                        score, higher_is_better = compute_ranking_score(strat_is, _metric, _roc_s)
                        banned_days = compute_cusum_banned(strat_is)
                        active_days = strat_is['weekday_open'].unique().tolist()
                        remaining   = [d for d in active_days if d not in banned_days]
                        if not remaining:
                            cusum_exclusions_log.append({'OOS Start': current_oos_start, 'Strategy': strat,
                                'Motivo': f"Tutti i giorni bannati: {format_banned_days(banned_days)}"})
                            continue
                        is_metrics.append({'Strategy': strat, 'Group': _sm[strat],
                            'Score': score, 'Higher_is_better': higher_is_better,
                            'Net PnL IS': strat_is['Net PnL'].sum(), 'Banned Days': banned_days})

                    is_metrics_df = pd.DataFrame(is_metrics)
                    if not is_metrics_df.empty:
                        asc = False if is_metrics_df['Higher_is_better'].all() else True
                        is_metrics_df = is_metrics_df.sort_values(['Score','Net PnL IS'], ascending=[asc, False])
                        selected_strategies, group_counts = [], {}
                        window_realized = 0.0
                        for _, row in is_metrics_df.iterrows():
                            if len(selected_strategies) >= _top_n: break
                            grp = row['Group']
                            if group_counts.get(grp, 0) < _mpg:
                                selected_strategies.append(row)
                                group_counts[grp] = group_counts.get(grp, 0) + 1
                        for rank, strat_row in enumerate(selected_strategies):
                            weight     = _fwp if rank < _fwc else _bwp
                            strat_name = strat_row['Strategy']
                            banned     = strat_row['Banned Days']
                            historical_allocations.append({
                                'OOS Start': current_oos_start, 'OOS End': current_oos_end,
                                'Rank': rank + 1, 'Strategy': strat_name, 'Group': strat_row['Group'],
                                'Score IS': round(strat_row['Score'], 4), 'Metric': _metric,
                                'Weight': weight, 'Banned Days': banned,
                                'Top-N': _top_n, 'Max Per Group': _mpg,
                            })
                            strat_oos = oos_data[oos_data['Strategy'] == strat_name].copy()
                            if not strat_oos.empty:
                                strat_oos = strat_oos[~strat_oos['weekday_open'].isin(banned)]
                                start_eq = 0.0
                                if ec_enabled and ec_capital_mode == "Trailing":
                                    start_eq = equity_state.get(strat_name, 0.0)
                                strat_oos['Weighted Net PnL'] = strat_oos['Net PnL'] * weight
                                strat_oos['Weight'] = weight

                                if ec_enabled:
                                    thresholds = dd_thresholds.get(strat_name)
                                    if thresholds is not None:
                                        strat_oos_ec = apply_equity_control(
                                            strat_oos,
                                            thresholds=thresholds,
                                            capital_mode=ec_capital_mode,
                                            start_equity=start_eq,
                                        )
                                        strat_oos_ec["Weighted Net PnL Raw"] = strat_oos_ec["Weighted Net PnL"]
                                        strat_oos_ec["Weighted Net PnL"] = strat_oos_ec["EC Weighted Net PnL"]
                                        if ec_capital_mode == "Trailing":
                                            if "EC Tracking Equity" in strat_oos_ec.columns and not strat_oos_ec.empty:
                                                equity_state[strat_name] = float(strat_oos_ec["EC Tracking Equity"].iloc[-1])
                                        strat_oos = strat_oos_ec

                                if sizing_enabled:
                                    cfg_dict = sizing_strategy_configs.get(strat_name, {})
                                    sizing_cfg = StrategySizingConfig(
                                        cap_pct=cfg_dict.get("cap_pct", DEFAULT_CAP_PCT),
                                        sl_usd=cfg_dict.get("sl_usd", DEFAULT_SL_USD),
                                        sl_pct=cfg_dict.get("sl_pct", DEFAULT_SL_PCT),
                                    )

                                    strat_oos = apply_portfolio_sizing(
                                        strat_oos,
                                        sizing_cfg=sizing_cfg,
                                        capital=current_capital,
                                        max_daily_loss_pct=sizing_params.max_daily_loss_pct,
                                    )
                                    strat_oos["Weighted Net PnL PreSizing"] = strat_oos["Weighted Net PnL"]
                                    strat_oos["Weighted Net PnL"] = strat_oos["Realized PnL $"]
                                    window_realized += float(strat_oos["Realized PnL $"].sum())

                                oos_results.append(strat_oos)

                        if sizing_enabled and sizing_params.compounding:
                            current_capital = max(0.0, current_capital + window_realized)
                    current_oos_start = current_oos_end

            if not oos_results:
                st.error("Nessun trade OOS generato. Il dataset deve coprire almeno 13 mesi di storia.")
            else:
                st.session_state["wfa_results"] = {
                    "final_oos_df":   pd.concat(oos_results).sort_values('Date Closed').reset_index(drop=True),
                    "hist_alloc_df":  pd.DataFrame(historical_allocations),
                    "exclusions_df":  pd.DataFrame(cusum_exclusions_log),
                    "roc_filter_df":  pd.DataFrame(roc_filter_log),
                    "params": {
                        "top_n": _top_n, "full_weight_count": _fwc,
                        "full_weight_pct": int(_fwp*100), "bench_weight_pct": int(_bwp*100),
                        "max_per_group": _mpg, "ranking_metric": _metric,
                        "roc_steps": _roc_s, "roc_filter_enabled": _roc_fe, "roc_filter_steps": _roc_fs,
                    }
                }
                # Invalida cache dettaglio finestre al cambio run
                st.session_state.pop("wfa_detail_windows", None)
                st.session_state.pop("wfa_detail_sig", None)
                st.success("\u2705 Elaborazione completata!")

        if st.session_state.get("wfa_results") is not None:
            res           = st.session_state["wfa_results"]
            final_oos_df  = res["final_oos_df"]
            hist_alloc_df = res["hist_alloc_df"]
            exclusions_df = res["exclusions_df"]
            roc_filter_df = res.get("roc_filter_df", pd.DataFrame())
            run_params    = res["params"]

            st.header("1. Allocazione Corrente")
            latest_start = hist_alloc_df['OOS Start'].max()
            latest_end   = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start]['OOS End'].iloc[0]
            today        = pd.Timestamp.today().normalize()
            days_left    = (latest_end - today).days

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.info(f"\U0001f4c5 **Dal:** {latest_start.strftime('%d %b %Y')}")
            c2.info(f"\U0001f51a **Fino al:** {latest_end.strftime('%d %b %Y')}")
            if days_left > 0: c3.warning(f"\u23f3 **Rotazione tra:** {days_left}gg")
            else:             c3.error("\U0001f504 **Rotazione scaduta**")
            c4.info(f"\U0001f3c6 **Ranking:** {run_params['ranking_metric']}")
            c5.info(f"\U0001f3db\ufe0f Top-N: **{run_params['top_n']}** | Max/gr: **{run_params['max_per_group']}**")
            c6.info(f"\u2696\ufe0f **{run_params['full_weight_count']}\u00d7{run_params['full_weight_pct']}%** + **{run_params['top_n']-run_params['full_weight_count']}\u00d7{run_params['bench_weight_pct']}%**")

            latest_alloc = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start].copy()
            disp = latest_alloc[['Rank','Strategy','Group','Score IS','Metric','Weight','Banned Days']].copy()
            disp.rename(columns={'Score IS': f"Score IS ({run_params['ranking_metric']})", 'Metric': 'Metrica'}, inplace=True)
            disp['Weight'] = (disp['Weight'] * 100).round(0).astype(int).astype(str) + '%'
            disp['Giorni Spenti (CUSUM)'] = disp['Banned Days'].apply(format_banned_days)
            disp.drop(columns=['Banned Days', 'Metrica'], inplace=True)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            _show_logs = []
            if not exclusions_df.empty:
                latest_excl = exclusions_df[exclusions_df['OOS Start'] == latest_start]
                if not latest_excl.empty: _show_logs.append(("CUSUM", latest_excl))
            if not roc_filter_df.empty:
                latest_roc_excl = roc_filter_df[roc_filter_df['OOS Start'] == latest_start]
                if not latest_roc_excl.empty: _show_logs.append(("ROC filter", latest_roc_excl))
            for log_label, log_df in _show_logs:
                st.warning(f"\u26a0\ufe0f Strategie escluse ({log_label}):")
                st.dataframe(log_df[['Strategy','Motivo']], use_container_width=True, hide_index=True)

            st.header("2. Metriche Out-Of-Sample")
            tab_global, tab_recent = st.tabs(["\U0001f4ca Storico Completo OOS", "\U0001f4c8 Dal 1\u00b0 Settembre 2025"])
            with tab_global:
                render_metrics(calculate_metrics(final_oos_df), "Metriche \u2014 Storico Completo OOS")
            with tab_recent:
                recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
                render_metrics(calculate_metrics(recent_df), "Metriche \u2014 Dal 1\u00b0 Settembre 2025")

            st.header("3. Curva Equity \u2014 Storico Completo OOS")
            all_daily = final_oos_df.groupby(final_oos_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
            all_cum   = all_daily.cumsum()
            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(
                x=all_cum.index, y=all_cum.values, mode='lines', fill='tozeroy',
                line=dict(color='#00d4ff', width=2), fillcolor='rgba(0,212,255,0.08)'
            ))
            fig_all.update_layout(title="Equity Cumulativa WFA \u2014 Storico Completo",
                xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
            st.plotly_chart(fig_all, use_container_width=True)

            st.header("4. Curva Equity \u2014 Dal 1\u00b0 Settembre 2025")
            recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
            if not recent_df.empty:
                rec_daily = recent_df.groupby(recent_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
                rec_cum   = rec_daily.cumsum()
                fig_rec = go.Figure()
                fig_rec.add_trace(go.Scatter(
                    x=rec_cum.index, y=rec_cum.values, mode='lines', fill='tozeroy',
                    line=dict(color='#f59e0b', width=2), fillcolor='rgba(245,158,11,0.08)'
                ))
                fig_rec.update_layout(title="Equity Cumulativa WFA \u2014 Dal 1\u00b0 Settembre 2025",
                    xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
                st.plotly_chart(fig_rec, use_container_width=True)
            else:
                st.info("Nessun dato OOS disponibile dal 1\u00b0 settembre 2025.")

            st.header("5. Esporta Dati")
            hist_export = hist_alloc_df.copy()
            hist_export['Banned Days'] = hist_export['Banned Days'].apply(
                lambda x: format_banned_days(x) if isinstance(x, list) else x)
            equity_csv_df = pd.DataFrame({
                'Date': all_cum.index, 'Daily PnL': all_daily.values, 'Cumulative PnL': all_cum.values})

            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            with col_b1:
                st.download_button("\U0001f4e5 Allocazioni Storiche",
                    data=hist_export.to_csv(index=False).encode('utf-8'), file_name="wfa_allocations.csv", mime="text/csv")
            with col_b2:
                st.download_button("\U0001f4e5 Trade OOS Filtrati",
                    data=final_oos_df.to_csv(index=False).encode('utf-8'), file_name="wfa_oos_trades.csv", mime="text/csv")
            with col_b3:
                st.download_button("\U0001f4e5 Equity Line Completa",
                    data=equity_csv_df.to_csv(index=False).encode('utf-8'), file_name="equity_full.csv", mime="text/csv")
            with col_b4:
                if not recent_df.empty:
                    rec_equity_csv = pd.DataFrame({
                        'Date': rec_cum.index, 'Daily PnL': rec_daily.values, 'Cumulative PnL': rec_cum.values})
                    st.download_button("\U0001f4e5 Equity Line Set 2025+",
                        data=rec_equity_csv.to_csv(index=False).encode('utf-8'), file_name="equity_sep2025.csv", mime="text/csv")
            with col_b5:
                combined_excl = pd.concat([exclusions_df, roc_filter_df]).reset_index(drop=True) \
                    if not roc_filter_df.empty else exclusions_df
                if not combined_excl.empty:
                    combined_excl['OOS Start'] = combined_excl['OOS Start'].astype(str)
                    st.download_button("\U0001f4e5 Log Esclusioni",
                        data=combined_excl.to_csv(index=False).encode('utf-8'), file_name="wfa_exclusions.csv", mime="text/csv")

            # ─── SEZIONE 6: DETTAGLIO COMBINAZIONE WFA ────────────────────────
            st.header("6. Dettaglio Combinazione WFA")
            st.caption(
                "Analisi per singola finestra OOS dei parametri attualmente in uso: "
                "distribuzione del Profit Factor, PnL finestra per finestra e frequenza delle strategie selezionate."
            )

            _wfa_params_detail = WFAParams(
                top_n=run_params["top_n"],
                full_weight_count=run_params["full_weight_count"],
                full_weight_pct=float(run_params["full_weight_pct"]),
                bench_weight_pct=float(run_params["bench_weight_pct"]),
                max_per_group=run_params["max_per_group"],
                ranking_metric=run_params["ranking_metric"],
                roc_filter_enabled=run_params["roc_filter_enabled"],
                roc_filter_steps=run_params["roc_filter_steps"],
            )

            _detail_cache_key = "wfa_detail_windows"
            _detail_sig_key   = "wfa_detail_sig"
            _current_sig = (
                run_params["top_n"], run_params["full_weight_count"],
                run_params["full_weight_pct"], run_params["bench_weight_pct"],
                run_params["max_per_group"], run_params["ranking_metric"],
                run_params["roc_filter_enabled"], run_params["roc_filter_steps"],
            )

            if (
                _detail_cache_key not in st.session_state
                or st.session_state.get(_detail_sig_key) != _current_sig
            ):
                with st.spinner("Calcolo finestre OOS per il dettaglio..."):
                    _df_for_detail = df_raw.copy()
                    _df_for_detail["Date Opened"] = pd.to_datetime(_df_for_detail["Date Opened"])
                    _df_for_detail["Date Closed"] = pd.to_datetime(_df_for_detail["Date Closed"])
                    _df_for_detail["weekday_open"] = _df_for_detail["Date Opened"].dt.weekday
                    _df_for_detail["P/L"] = _df_for_detail["P/L"].apply(clean_money)
                    _df_for_detail["Opening Commissions + Fees"] = _df_for_detail["Opening Commissions + Fees"].fillna(0).apply(clean_money)
                    _df_for_detail["Closing Commissions + Fees"] = _df_for_detail["Closing Commissions + Fees"].fillna(0).apply(clean_money)
                    _df_for_detail["Net PnL"] = (
                        _df_for_detail["P/L"]
                        - _df_for_detail["Opening Commissions + Fees"]
                        - _df_for_detail["Closing Commissions + Fees"]
                    )
                    _df_for_detail = _df_for_detail.sort_values("Date Closed").reset_index(drop=True)

                    _windows_detail = run_wfa_single_windowed(
                        df=_df_for_detail,
                        params=_wfa_params_detail,
                        group_map=strategy_mapping,
                        date_col="Date Closed",
                        strategy_col="Strategy",
                        pnl_col="Net PnL",
                    )
                st.session_state[_detail_cache_key] = _windows_detail
                st.session_state[_detail_sig_key]   = _current_sig
            else:
                _windows_detail = st.session_state[_detail_cache_key]

            if not _windows_detail:
                st.warning("Nessuna finestra OOS disponibile per il dettaglio.")
            else:
                _n_win = len(_windows_detail)

                # ── Box Plot PF per finestra OOS ──────────────────────────────
                st.markdown("#### \U0001f3af Box Plot PF per finestra OOS")
                _pf_vals = [w["pf"] for w in _windows_detail]
                _rng = np.random.default_rng(42)
                _jitter_x     = _rng.uniform(-0.25, 0.25, size=len(_pf_vals)).tolist()
                _point_colors = ["#6ee7b7" if v >= 1.0 else "#f87171" for v in _pf_vals]
                _hover_texts  = [f"Finestra {i+1}<br>PF: {v:.2f}" for i, v in enumerate(_pf_vals)]

                fig_wfa_box = go.Figure()
                fig_wfa_box.add_trace(go.Box(
                    y=_pf_vals, x=[0] * len(_pf_vals),
                    name="PF per finestra",
                    marker_color="#4f98a3", line_color="#4f98a3",
                    fillcolor="rgba(79,152,163,0.25)",
                    boxpoints=False, showlegend=False, hoverinfo="skip",
                ))
                fig_wfa_box.add_trace(go.Scatter(
                    x=_jitter_x, y=_pf_vals, mode="markers",
                    marker=dict(size=8, color=_point_colors,
                                line=dict(color="#0d1117", width=1), opacity=0.9),
                    text=_hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False, name="Punti",
                ))
                fig_wfa_box.add_hline(
                    y=1.0, line_dash="dash", line_color="#ef4444", line_width=1.5,
                    annotation_text="PF = 1.0",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#ef4444", size=10),
                )
                fig_wfa_box.update_layout(
                    template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#131920",
                    font=dict(color="#e0eaf4"),
                    xaxis=dict(showticklabels=False, zeroline=False, showgrid=False, range=[-1, 1]),
                    yaxis=dict(title="Profit Factor", gridcolor="#1e2a35"),
                    height=360, margin=dict(l=60, r=40, t=40, b=40), showlegend=False,
                    title=dict(
                        text=f"Distribuzione PF \u2014 {_n_win} finestre OOS",
                        font=dict(size=13, color="#e0eaf4"), x=0.02,
                    ),
                )
                st.plotly_chart(fig_wfa_box, use_container_width=True)

                # ── PnL per finestra OOS ──────────────────────────────────────
                st.markdown("#### \U0001f4c5 PnL per finestra OOS")
                st.caption("Verde = finestra profittevole \u00b7 Rosso = finestra in perdita \u00b7 Linea tratteggiata = break-even.")

                _win_labels = [f"W{w['window']}\n{w['start_oos'][:7]}" for w in _windows_detail]
                _win_pnl    = [w["total_pnl"] for w in _windows_detail]
                _win_pf     = [w["pf"] for w in _windows_detail]
                _win_start  = [w["start_oos"] for w in _windows_detail]
                _win_end    = [w["end_oos"] for w in _windows_detail]
                _bar_colors = ["#6ee7b7" if v > 0 else "#f87171" for v in _win_pnl]

                fig_wfa_pnl = go.Figure()
                fig_wfa_pnl.add_trace(go.Bar(
                    x=_win_labels, y=_win_pnl,
                    marker_color=_bar_colors,
                    marker_line=dict(color="#0d1117", width=0.8),
                    customdata=list(zip(_win_start, _win_end, _win_pf)),
                    hovertemplate=(
                        "<b>Finestra %{x}</b><br>"
                        "Dal %{customdata[0]} al %{customdata[1]}<br>"
                        "<b>PnL:</b> $%{y:,.2f}<br>"
                        "<b>PF:</b> %{customdata[2]:.2f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))
                fig_wfa_pnl.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1.2)
                fig_wfa_pnl.update_layout(
                    template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#131920",
                    font=dict(color="#e0eaf4"),
                    xaxis=dict(title="Finestra OOS", gridcolor="#1e2a35",
                               tickangle=-45 if len(_win_labels) > 8 else 0),
                    yaxis=dict(title="Total PnL ($)", gridcolor="#1e2a35",
                               tickprefix="$", separatethousands=True, zeroline=False),
                    height=320, margin=dict(l=70, r=40, t=30, b=60), bargap=0.25,
                )
                st.plotly_chart(fig_wfa_pnl, use_container_width=True)

                # ── Frequenza strategie nelle finestre OOS ────────────────────
                _all_selected = []
                for _w in _windows_detail:
                    _all_selected.extend(_w.get("selected_strategies", []))

                if _all_selected:
                    st.markdown("#### \U0001f9e9 Frequenza strategie nelle finestre OOS")
                    st.caption(
                        "Quante finestre OOS ha coperto ogni strategia selezionata in IS. "
                        "Barre pi\u00f9 chiare = presente in meno del 50% delle finestre."
                    )
                    _freq         = Counter(_all_selected)
                    _strat_names  = [s for s, _ in _freq.most_common()]
                    _strat_counts = [_freq[s] for s in _strat_names]
                    _strat_pcts   = [_freq[s] / _n_win * 100 for s in _strat_names]
                    _bar_colors_s = [
                        "#4f98a3" if (_freq[s] / _n_win) >= 0.5 else "#2a5560"
                        for s in _strat_names
                    ]

                    fig_wfa_freq = go.Figure()
                    fig_wfa_freq.add_trace(go.Bar(
                        x=_strat_counts, y=_strat_names, orientation="h",
                        marker_color=_bar_colors_s,
                        marker_line=dict(color="#0d1117", width=0.6),
                        text=[f"{p:.0f}%" for p in _strat_pcts],
                        textposition="outside",
                        textfont=dict(color="#e0eaf4", size=11),
                        customdata=_strat_pcts,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Presente in <b>%{x}</b> finestre su " + str(_n_win) + "<br>"
                            "Frequenza: <b>%{customdata:.1f}%</b><br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ))
                    fig_wfa_freq.update_layout(
                        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#131920",
                        font=dict(color="#e0eaf4"),
                        xaxis=dict(title="N\u00b0 finestre OOS", gridcolor="#1e2a35",
                                   range=[0, _n_win * 1.18], zeroline=False),
                        yaxis=dict(title="", gridcolor="#1e2a35", automargin=True),
                        height=max(260, len(_strat_names) * 28 + 80),
                        margin=dict(l=20, r=80, t=30, b=50), bargap=0.3,
                    )
                    st.plotly_chart(fig_wfa_freq, use_container_width=True)
                else:
                    st.caption(
                        "\u2139\ufe0f Dati `selected_strategies` non disponibili per questa run. "
                        "Riesegui la simulazione WFA per visualizzare il grafico."
                    )

    # ─── TAB OTTIMIZZATORE ────────────────────────────────────────────────────
    with tab_opt:
        df_opt = df_raw.copy()
        df_opt['Date Closed'] = pd.to_datetime(df_opt['Date Closed'])
        df_opt['Date Opened'] = pd.to_datetime(df_opt['Date Opened'])
        df_opt['P/L'] = df_opt['P/L'].apply(clean_money)
        df_opt['Opening Commissions + Fees'] = df_opt['Opening Commissions + Fees'].fillna(0).apply(clean_money)
        df_opt['Closing Commissions + Fees'] = df_opt['Closing Commissions + Fees'].fillna(0).apply(clean_money)
        df_opt['Net PnL'] = (
            df_opt['P/L']
            - df_opt['Opening Commissions + Fees']
            - df_opt['Closing Commissions + Fees']
        )
        df_opt['weekday_open'] = df_opt['Date Opened'].dt.weekday
        df_opt = df_opt.sort_values('Date Closed').reset_index(drop=True)

        render_optimizer_tab(
            df_processed=df_opt,
            group_map=st.session_state['strategy_mapping'],
            date_col='Date Closed',
            strategy_col='Strategy',
            pnl_col='Net PnL',
        )