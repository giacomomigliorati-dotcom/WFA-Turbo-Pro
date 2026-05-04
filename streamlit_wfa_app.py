import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

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

# ─── COSTANTI ────────────────────────────────────────────────────────────────
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
GIORNI_SETTIMANA = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
OOS_STEP_DAYS    = 28
DEFAULT_MAX_PER_GROUP      = 2
DEFAULT_TOP_N              = 7
DEFAULT_FULL_WEIGHT_COUNT  = 5
DEFAULT_FULL_WEIGHT_PCT    = 100
DEFAULT_BENCH_WEIGHT_PCT   = 50
DEFAULT_RANKING_METRIC     = "Omega Ratio"
DEFAULT_ROC_STEPS          = 2       # finestra ROC = 2 × 28 = 56 giorni
DEFAULT_ROC_FILTER_ENABLED = False
DEFAULT_ROC_FILTER_STEPS   = 2

RANKING_METRICS = ["Omega Ratio", "Sharpe Ratio", "Sortino Ratio", "Ulcer Index", "ROC"]

OZONE_LAYOUT = dict(
    plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
    font=dict(family='Inter, Segoe UI, sans-serif', color='#e0eaf4', size=12),
    xaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    yaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    margin=dict(l=50, r=30, t=60, b=50),
)

# ─── FUNZIONI METRICHE IS ──────────────────────────────────────────────────────────
def calc_omega(strat_is):
    if strat_is['P/L %'].notna().sum() > 0:
        pos = strat_is[strat_is['P/L %'] > 0]['P/L %'].sum()
        neg = strat_is[strat_is['P/L %'] < 0]['P/L %'].sum()
    else:
        pos = strat_is[strat_is['Net PnL'] > 0]['Net PnL'].sum()
        neg = strat_is[strat_is['Net PnL'] < 0]['Net PnL'].sum()
    return 50.0 if neg == 0 else pos / abs(neg)

def calc_sharpe(strat_is):
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    if len(daily) < 2: return -999.0
    m, s = daily.mean(), daily.std()
    return (m / s) * np.sqrt(252) if s > 0 else -999.0

def calc_sortino(strat_is):
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    if len(daily) < 2: return -999.0
    down = daily[daily < 0].std()
    return (daily.mean() / down) * np.sqrt(252) if pd.notna(down) and down > 0 else -999.0

def calc_ulcer(strat_is):
    """Ulcer Index (minore = migliore; ranking verrà invertito)."""
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    cum   = daily.cumsum()
    peak  = cum.cummax()
    dd_pct = ((cum - peak) / (peak.abs() + 1e-9)) * 100  # % drawdown
    ui = np.sqrt((dd_pct ** 2).mean())
    return ui if ui > 0 else 0.0

def calc_roc(strat_is, n_steps):
    """ROC = variazione % del Net PnL cumulativo su finestra di n_steps * OOS_STEP_DAYS."""
    window_days = n_steps * OOS_STEP_DAYS
    cutoff = strat_is['Date Closed'].max() - pd.Timedelta(days=window_days)
    recent = strat_is[strat_is['Date Closed'] > cutoff]
    if recent.empty: return -999.0
    pnl_window = recent['Net PnL'].sum()
    return pnl_window  # in $ assoluti (già ordinabile)

def compute_ranking_score(strat_is, metric, roc_steps):
    """Ritorna (score, higher_is_better)."""
    if metric == "Omega Ratio":
        return calc_omega(strat_is), True
    elif metric == "Sharpe Ratio":
        return calc_sharpe(strat_is), True
    elif metric == "Sortino Ratio":
        return calc_sortino(strat_is), True
    elif metric == "Ulcer Index":
        return calc_ulcer(strat_is), False  # minore = migliore
    elif metric == "ROC":
        return calc_roc(strat_is, roc_steps), True
    return calc_omega(strat_is), True

def passes_roc_filter(strat_is, filter_steps):
    """True se ROC >= 0 (strategia non in perdita netta nella finestra filtro)."""
    return calc_roc(strat_is, filter_steps) >= 0

# ─── FUNZIONI CORE GENERALI ────────────────────────────────────────────────────────
def clean_money(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    return float(str(val).replace('$', '').replace(',', ''))

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
    r1[3].metric("Profit Factor",      f"{m['Profit Factor']:.2f}" if m['Profit Factor'] != np.inf else "∞")
    r2 = st.columns(4)
    r2[0].metric("Avg Win",      f"${m['Avg Win ($)']:,.2f}")
    r2[1].metric("Avg Loss",     f"${m['Avg Loss ($)']:,.2f}")
    r2[2].metric("Expectancy",   f"${m['Expectancy ($)']:,.2f}")
    r2[3].metric("Max Drawdown", f"${m['Max Drawdown ($)']:,.2f}")
    r3 = st.columns(4)
    r3[0].metric("Sharpe Ratio",    f"{m['Sharpe Ratio']:.2f}")
    r3[1].metric("Sortino Ratio",   f"{m['Sortino Ratio']:.2f}")
    r3[2].metric("Calmar Ratio",    f"{m['Calmar Ratio']:.2f}")
    r3[3].metric("Recovery Factor", f"{m['Recovery Factor']:.2f}" if m['Recovery Factor'] != np.inf else "∞")
    r4 = st.columns(3)
    r4[0].metric("Avg Daily PnL", f"${m['Avg Daily PnL ($)']:,.2f}")
    r4[1].metric("Best Day",      f"${m['Best Day ($)']:,.2f}")
    r4[2].metric("Worst Day",     f"${m['Worst Day ($)']:,.2f}")

def compute_cusum_banned(is_data_strat):
    banned = []
    for day in range(7):
        day_data = is_data_strat[is_data_strat['weekday_open'] == day].sort_values('Date Closed')
        if len(day_data) >= 10:
            std = day_data['Net PnL'].std()
            if std > 0:
                k, h, c_low = 0.5 * std, 3.0 * std, 0
                for pnl in day_data['Net PnL']:
                    c_low = max(0, c_low - pnl - k)
                if c_low > h:
                    banned.append(day)
    return banned

def format_banned_days(banned_list):
    if not banned_list: return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in banned_list if d < len(GIORNI_SETTIMANA)])

# ─── FUNZIONI JSON CONFIG ────────────────────────────────────────────────────
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

# ─── INIT SESSION STATE ──────────────────────────────────────────────────────────────
for key, default in [
    ("top_n",             DEFAULT_TOP_N),
    ("full_weight_count", DEFAULT_FULL_WEIGHT_COUNT),
    ("full_weight_pct",   DEFAULT_FULL_WEIGHT_PCT),
    ("bench_weight_pct",  DEFAULT_BENCH_WEIGHT_PCT),
    ("ranking_metric",    DEFAULT_RANKING_METRIC),
    ("roc_steps",         DEFAULT_ROC_STEPS),
    ("roc_filter_enabled",DEFAULT_ROC_FILTER_ENABLED),
    ("roc_filter_steps",  DEFAULT_ROC_FILTER_STEPS),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── UPLOAD CSV ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw[~df_raw['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()
    all_strategies_in_file = sorted(df_raw['Strategy'].dropna().unique().tolist())

    if 'strategy_mapping' not in st.session_state:
        st.session_state['strategy_mapping'] = DEFAULT_GROUP_MAPPING.copy()
    if 'group_names' not in st.session_state:
        st.session_state['group_names'] = sorted(set(DEFAULT_GROUP_MAPPING.values()))
    if 'max_per_group' not in st.session_state:
        st.session_state['max_per_group'] = DEFAULT_MAX_PER_GROUP
    for strat in all_strategies_in_file:
        if strat not in st.session_state['strategy_mapping']:
            st.session_state['strategy_mapping'][strat] = ''

    # ─── SIDEBAR ─────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configurazione")

    # ── BACKUP JSON
    st.sidebar.subheader("💾 Backup configurazione")
    st.sidebar.caption("Esporta per salvare tutti i parametri. Reimporta dopo ogni refresh.")
    config_json_str = json.dumps(build_config_payload(), ensure_ascii=False, indent=2)
    st.sidebar.download_button(label="📤 Esporta configurazione JSON",
        data=config_json_str.encode("utf-8"), file_name="texano_config.json", mime="application/json")
    uploaded_config = st.sidebar.file_uploader("📥 Importa configurazione JSON", type=["json"], key="config_json_uploader")
    if uploaded_config is not None:
        try:
            parsed = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.session_state["pending_config_payload"] = parsed
            st.sidebar.success("✅ JSON caricato. Premi 'Applica' per attivarlo.")
        except Exception as e:
            st.sidebar.error(f"JSON non valido: {e}")
    if st.sidebar.button("✅ Applica configurazione JSON"):
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

    # ── PARAMETRI WFA
    st.sidebar.subheader("🎛️ Parametri WFA")

    top_n = st.sidebar.number_input("Numero strategie selezionate (Top-N)",
        min_value=1, max_value=20, value=st.session_state["top_n"], step=1, help="Default: 7")
    st.session_state["top_n"] = int(top_n)

    full_weight_count = st.sidebar.number_input("Strategie con peso pieno (rank 1…N)",
        min_value=0, max_value=int(top_n),
        value=min(st.session_state["full_weight_count"], int(top_n)), step=1)
    st.session_state["full_weight_count"] = int(full_weight_count)

    full_weight_pct = st.sidebar.number_input("Peso pieno (%)",
        min_value=1, max_value=100, value=st.session_state["full_weight_pct"], step=5)
    st.session_state["full_weight_pct"] = int(full_weight_pct)

    bench_weight_pct = st.sidebar.number_input("Peso panchina (%)",
        min_value=1, max_value=100, value=st.session_state["bench_weight_pct"], step=5)
    st.session_state["bench_weight_pct"] = int(bench_weight_pct)

    bench_count = int(top_n) - int(full_weight_count)
    st.sidebar.markdown(f"**Schema:** {int(full_weight_count)}×{int(full_weight_pct)}% + {bench_count}×{int(bench_weight_pct)}%")

    st.sidebar.divider()

    # ── METRICA DI RANKING IS
    st.sidebar.subheader("🏆 Metrica di Ranking IS")

    ranking_metric_idx = RANKING_METRICS.index(st.session_state["ranking_metric"]) \
        if st.session_state["ranking_metric"] in RANKING_METRICS else 0
    ranking_metric = st.sidebar.selectbox(
        "Metrica per ordinare le strategie nella finestra IS",
        options=RANKING_METRICS, index=ranking_metric_idx,
        help="Omega Ratio: default. Ulcer Index: minore = migliore. ROC: PnL netto su N finestre OOS."
    )
    st.session_state["ranking_metric"] = ranking_metric

    roc_steps = st.session_state["roc_steps"]
    if ranking_metric == "ROC":
        roc_steps = st.sidebar.number_input(
            f"Finestra ROC (multipli di {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, value=st.session_state["roc_steps"], step=1,
            help=f"Es. 2 = finestra di {2*OOS_STEP_DAYS} giorni. Usa i dati IS dell'ultimo anno."
        )
        st.session_state["roc_steps"] = int(roc_steps)
        st.sidebar.caption(f"📅 Finestra ROC attiva: **{int(roc_steps) * OOS_STEP_DAYS} giorni**")

    st.sidebar.divider()

    # ── FILTRO ROC < 0
    st.sidebar.subheader("🚫 Filtro ROC < 0")
    roc_filter_enabled = st.sidebar.toggle(
        "Attiva filtro ROC < 0",
        value=st.session_state["roc_filter_enabled"],
        help="Se attivo, esclude le strategie con Net PnL negativo sulla finestra configurata."
    )
    st.session_state["roc_filter_enabled"] = roc_filter_enabled

    roc_filter_steps = st.session_state["roc_filter_steps"]
    if roc_filter_enabled:
        roc_filter_steps = st.sidebar.number_input(
            f"Finestra filtro ROC (multipli di {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, value=st.session_state["roc_filter_steps"], step=1,
            help=f"Es. 2 = ultimi {2*OOS_STEP_DAYS} giorni della finestra IS."
        )
        st.session_state["roc_filter_steps"] = int(roc_filter_steps)
        st.sidebar.caption(
            f"📅 Finestra filtro ROC: **{int(roc_filter_steps) * OOS_STEP_DAYS} giorni** — "
            f"le strategie con PnL netto negativo in questo periodo vengono escluse."
        )

    st.sidebar.divider()

    # ── LIMITE MAX PER GRUPPO
    st.sidebar.subheader("Limite per gruppo")
    max_per_group = st.sidebar.number_input("Max strategie stesso gruppo",
        min_value=1, max_value=int(top_n),
        value=min(st.session_state['max_per_group'], int(top_n)), step=1, help="Default 2")
    st.session_state['max_per_group'] = int(max_per_group)

    st.sidebar.divider()

    # ── GESTIONE GRUPPI
    st.sidebar.subheader("Gestione gruppi")
    new_group_name = st.sidebar.text_input("Nome nuovo gruppo", key="new_group_name")
    if st.sidebar.button("➕ Crea gruppo"):
        clean_name = new_group_name.strip()
        if clean_name:
            if clean_name not in st.session_state['group_names']:
                st.session_state['group_names'].append(clean_name)
                st.session_state['group_names'] = sorted(st.session_state['group_names'])
                st.sidebar.success(f"Gruppo '{clean_name}' creato")
                st.rerun()
            else:
                st.sidebar.warning("Gruppo già esistente")
        else:
            st.sidebar.warning("Inserisci un nome valido")

    if st.session_state['group_names']:
        group_to_rename = st.sidebar.selectbox("Gruppo da rinominare",
            options=st.session_state['group_names'], key="group_to_rename")
        renamed_group = st.sidebar.text_input("Nuovo nome", key="renamed_group")
        if st.sidebar.button("✏️ Rinomina gruppo"):
            old_name, new_name = group_to_rename, renamed_group.strip()
            if not new_name:
                st.sidebar.warning("Inserisci un nome valido")
            elif new_name in st.session_state['group_names'] and new_name != old_name:
                st.sidebar.warning("Nome già esistente")
            else:
                st.session_state['group_names'] = [new_name if g == old_name else g
                    for g in st.session_state['group_names']]
                st.session_state['strategy_mapping'] = {
                    s: (new_name if g == old_name else g)
                    for s, g in st.session_state['strategy_mapping'].items()}
                st.sidebar.success(f"'{old_name}' → '{new_name}'")
                st.rerun()

    st.sidebar.divider()

    # ── ASSEGNAZIONE STRATEGIE
    st.sidebar.subheader("Assegnazione strategie")
    group_options = [''] + st.session_state['group_names']
    with st.sidebar.expander("📋 Modifica gruppo di ogni strategia", expanded=True):
        updated_mapping = {}
        for strat in all_strategies_in_file:
            cur = st.session_state['strategy_mapping'].get(strat, '')
            idx = group_options.index(cur) if cur in group_options else 0
            sel = st.selectbox(label=strat, options=group_options, index=idx, key=f"group_{strat}")
            updated_mapping[strat] = sel
        if st.button("✅ Salva assegnazioni"):
            st.session_state['strategy_mapping'] = updated_mapping
            st.sidebar.success("Assegnazioni aggiornate — esporta il JSON!")
            st.rerun()

    strategy_mapping = st.session_state['strategy_mapping']
    still_unmapped = [s for s in all_strategies_in_file if not strategy_mapping.get(s, '').strip()]
    if still_unmapped:
        st.error(f"🛑 BLOCCO DI SICUREZZA: Strategie senza gruppo: {', '.join(still_unmapped)}")
        st.info("Apri la sidebar (▶), assegna un gruppo a ogni strategia e salva.")
        st.stop()

    # ─── PULSANTE LANCIA WFA ─────────────────────────────────────────────────
    st.markdown("---")
    col_launch, col_info = st.columns([3, 7])
    with col_launch:
        launch = st.button("▶ Lancia simulazione WFA", type="primary", use_container_width=True)
    with col_info:
        roc_filter_label = (
            f" &nbsp;|&nbsp; 🚫 ROC filter: <b style='color:#ef4444'>"
            f"{int(roc_filter_steps)*OOS_STEP_DAYS}gg</b>"
        ) if roc_filter_enabled else ""
        ranking_label_color = "#f59e0b" if ranking_metric != "Omega Ratio" else "#00d4ff"
        st.markdown(
            f"<div style='padding-top:10px; color:#7a9abf; font-size:0.88rem;'>"
            f"Ranking: <b style='color:{ranking_label_color}'>{ranking_metric}</b>"
            + (f" ({int(roc_steps)*OOS_STEP_DAYS}gg)" if ranking_metric=="ROC" else "")
            + f" &nbsp;|&nbsp; Top-N: <b style='color:#00d4ff'>{int(top_n)}</b>"
            f" &nbsp;|&nbsp; Pesi: <b style='color:#00d4ff'>{int(full_weight_count)}×{int(full_weight_pct)}%</b>"
            f" + <b style='color:#f59e0b'>{bench_count}×{int(bench_weight_pct)}%</b>"
            f" &nbsp;|&nbsp; Max/gr: <b style='color:#00d4ff'>{int(max_per_group)}</b>"
            + roc_filter_label + "</div>",
            unsafe_allow_html=True
        )

    if launch:
        # Snapshot parametri al momento del click
        _top_n    = int(st.session_state["top_n"])
        _fwc      = int(st.session_state["full_weight_count"])
        _fwp      = st.session_state["full_weight_pct"] / 100.0
        _bwp      = st.session_state["bench_weight_pct"] / 100.0
        _mpg      = int(st.session_state["max_per_group"])
        _sm       = dict(st.session_state["strategy_mapping"])
        _metric   = st.session_state["ranking_metric"]
        _roc_s    = int(st.session_state["roc_steps"])
        _roc_fe   = bool(st.session_state["roc_filter_enabled"])
        _roc_fs   = int(st.session_state["roc_filter_steps"])

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
            min_date  = df_filtered['Date Closed'].min()
            max_date  = df_filtered['Date Closed'].max()
            current_oos_start = min_date + pd.DateOffset(months=12)
            oos_results, historical_allocations, cusum_exclusions_log, roc_filter_log = [], [], [], []

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

                    # ─ Filtro ROC < 0 (opzionale)
                    if _roc_fe and not passes_roc_filter(strat_is, _roc_fs):
                        roc_filter_log.append({
                            'OOS Start': current_oos_start, 'Strategy': strat,
                            'Motivo': f"ROC < 0 su {_roc_fs*OOS_STEP_DAYS}gg"
                        })
                        continue

                    # ─ Metrica di ranking
                    score, higher_is_better = compute_ranking_score(strat_is, _metric, _roc_s)

                    # ─ CUSUM weekday ban
                    banned_days = compute_cusum_banned(strat_is)
                    active_days = strat_is['weekday_open'].unique().tolist()
                    remaining   = [d for d in active_days if d not in banned_days]
                    if not remaining:
                        cusum_exclusions_log.append({
                            'OOS Start': current_oos_start, 'Strategy': strat,
                            'Motivo': f"Tutti i giorni bannati: {format_banned_days(banned_days)}"
                        })
                        continue

                    is_metrics.append({
                        'Strategy': strat, 'Group': _sm[strat],
                        'Score': score, 'Higher_is_better': higher_is_better,
                        'Net PnL IS': strat_is['Net PnL'].sum(),
                        'Banned Days': banned_days
                    })

                is_metrics_df = pd.DataFrame(is_metrics)
                if not is_metrics_df.empty:
                    # Ulcer Index: ordinamento inverso
                    asc = False if is_metrics_df['Higher_is_better'].all() else True
                    is_metrics_df = is_metrics_df.sort_values(
                        ['Score', 'Net PnL IS'], ascending=[asc, False]
                    )
                    selected_strategies, group_counts = [], {}
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
                            'OOS Start':   current_oos_start,
                            'OOS End':     current_oos_end,
                            'Rank':        rank + 1,
                            'Strategy':    strat_name,
                            'Group':       strat_row['Group'],
                            'Score IS':    round(strat_row['Score'], 4),
                            'Metric':      _metric,
                            'Weight':      weight,
                            'Banned Days': banned,
                            'Top-N':       _top_n,
                            'Max Per Group': _mpg,
                        })
                        strat_oos = oos_data[oos_data['Strategy'] == strat_name].copy()
                        if not strat_oos.empty:
                            strat_oos = strat_oos[~strat_oos['weekday_open'].isin(banned)]
                            strat_oos['Weighted Net PnL'] = strat_oos['Net PnL'] * weight
                            oos_results.append(strat_oos)

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
            st.success("✅ Elaborazione completata!")

    # ─── RISULTATI ──────────────────────────────────────────────────────────────
    if st.session_state.get("wfa_results") is not None:
        res           = st.session_state["wfa_results"]
        final_oos_df  = res["final_oos_df"]
        hist_alloc_df = res["hist_alloc_df"]
        exclusions_df = res["exclusions_df"]
        roc_filter_df = res.get("roc_filter_df", pd.DataFrame())
        run_params    = res["params"]

        # ── 1. ALLOCAZIONE CORRENTE
        st.header("1. Allocazione Corrente")
        latest_start = hist_alloc_df['OOS Start'].max()
        latest_end   = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start]['OOS End'].iloc[0]
        today        = pd.Timestamp.today().normalize()
        days_left    = (latest_end - today).days

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.info(f"📅 **Dal:** {latest_start.strftime('%d %b %Y')}")
        c2.info(f"🔚 **Fino al:** {latest_end.strftime('%d %b %Y')}")
        if days_left > 0:
            c3.warning(f"⏳ **Rotazione tra:** {days_left}gg")
        else:
            c3.error("🔄 **Rotazione scaduta**")
        c4.info(f"🏆 **Ranking:** {run_params['ranking_metric']}")
        c5.info(f"🎛️ Top-N: **{run_params['top_n']}** | Max/gr: **{run_params['max_per_group']}**")
        c6.info(
            f"⚖️ **{run_params['full_weight_count']}×{run_params['full_weight_pct']}%** "
            f"+ **{run_params['top_n']-run_params['full_weight_count']}×{run_params['bench_weight_pct']}%**"
        )

        latest_alloc = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start].copy()
        disp = latest_alloc[['Rank','Strategy','Group','Score IS','Metric','Weight','Banned Days']].copy()
        disp.rename(columns={'Score IS': f"Score IS ({run_params['ranking_metric']})", 'Metric': 'Metrica'}, inplace=True)
        disp['Weight'] = (disp['Weight'] * 100).round(0).astype(int).astype(str) + '%'
        disp['Giorni Spenti (CUSUM)'] = disp['Banned Days'].apply(format_banned_days)
        disp.drop(columns=['Banned Days', 'Metrica'], inplace=True)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Log esclusioni
        _show_logs = []
        if not exclusions_df.empty:
            latest_excl = exclusions_df[exclusions_df['OOS Start'] == latest_start]
            if not latest_excl.empty: _show_logs.append(("CUSUM", latest_excl))
        if not roc_filter_df.empty:
            latest_roc_excl = roc_filter_df[roc_filter_df['OOS Start'] == latest_start]
            if not latest_roc_excl.empty: _show_logs.append(("ROC filter", latest_roc_excl))
        for log_label, log_df in _show_logs:
            st.warning(f"⚠️ Strategie escluse ({log_label}):")
            st.dataframe(log_df[['Strategy','Motivo']], use_container_width=True, hide_index=True)

        # ── 2. METRICHE OOS
        st.header("2. Metriche Out-Of-Sample")
        tab_global, tab_recent = st.tabs(["📊 Storico Completo OOS", "📈 Dal 1° Settembre 2025"])
        with tab_global:
            render_metrics(calculate_metrics(final_oos_df), "Metriche — Storico Completo OOS")
        with tab_recent:
            recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
            render_metrics(calculate_metrics(recent_df), "Metriche — Dal 1° Settembre 2025")

        # ── 3. EQUITY COMPLETA
        st.header("3. Curva Equity — Storico Completo OOS")
        all_daily = final_oos_df.groupby(final_oos_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
        all_cum   = all_daily.cumsum()
        fig_all = go.Figure()
        fig_all.add_trace(go.Scatter(
            x=all_cum.index, y=all_cum.values, mode='lines', fill='tozeroy',
            line=dict(color='#00d4ff', width=2), fillcolor='rgba(0,212,255,0.08)'
        ))
        fig_all.update_layout(title="Equity Cumulativa WFA — Storico Completo",
            xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
        st.plotly_chart(fig_all, use_container_width=True)

        # ── 4. EQUITY RECENTE
        st.header("4. Curva Equity — Dal 1° Settembre 2025")
        recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
        if not recent_df.empty:
            rec_daily = recent_df.groupby(recent_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
            rec_cum   = rec_daily.cumsum()
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Scatter(
                x=rec_cum.index, y=rec_cum.values, mode='lines', fill='tozeroy',
                line=dict(color='#f59e0b', width=2), fillcolor='rgba(245,158,11,0.08)'
            ))
            fig_rec.update_layout(title="Equity Cumulativa WFA — Dal 1° Settembre 2025",
                xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
            st.plotly_chart(fig_rec, use_container_width=True)
        else:
            st.info("Nessun dato OOS disponibile dal 1° settembre 2025.")

        # ── 5. EXPORT
        st.header("5. Esporta Dati")
        hist_export = hist_alloc_df.copy()
        hist_export['Banned Days'] = hist_export['Banned Days'].apply(
            lambda x: format_banned_days(x) if isinstance(x, list) else x)
        equity_csv_df = pd.DataFrame({
            'Date': all_cum.index, 'Daily PnL': all_daily.values, 'Cumulative PnL': all_cum.values})

        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        with col_b1:
            st.download_button("📥 Allocazioni Storiche",
                data=hist_export.to_csv(index=False).encode('utf-8'), file_name="wfa_allocations.csv", mime="text/csv")
        with col_b2:
            st.download_button("📥 Trade OOS Filtrati",
                data=final_oos_df.to_csv(index=False).encode('utf-8'), file_name="wfa_oos_trades.csv", mime="text/csv")
        with col_b3:
            st.download_button("📥 Equity Line Completa",
                data=equity_csv_df.to_csv(index=False).encode('utf-8'), file_name="equity_full.csv", mime="text/csv")
        with col_b4:
            if not recent_df.empty:
                rec_equity_csv = pd.DataFrame({
                    'Date': rec_cum.index, 'Daily PnL': rec_daily.values, 'Cumulative PnL': rec_cum.values})
                st.download_button("📥 Equity Line Set 2025+",
                    data=rec_equity_csv.to_csv(index=False).encode('utf-8'), file_name="equity_sep2025.csv", mime="text/csv")
        with col_b5:
            combined_excl = pd.concat([exclusions_df, roc_filter_df]).reset_index(drop=True) \
                if not roc_filter_df.empty else exclusions_df
            if not combined_excl.empty:
                combined_excl['OOS Start'] = combined_excl['OOS Start'].astype(str)
                st.download_button("📥 Log Esclusioni",
                    data=combined_excl.to_csv(index=False).encode('utf-8'), file_name="wfa_exclusions.csv", mime="text/csv")

else:
    st.info("⬆️ Carica un file CSV per iniziare.")
