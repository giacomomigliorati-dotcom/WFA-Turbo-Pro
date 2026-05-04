import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import itertools

st.set_page_config(page_title="Texano's Walk Forward", layout="wide")

# ─── OZONE THEME CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{
  --oz-bg:#0d1117;--oz-panel:#1a2332;--oz-panel2:#1e2d42;--oz-border:#2a3f5f;
  --oz-cyan:#00d4ff;--oz-cyan-dim:#0099bb;--oz-blue:#3b82f6;
  --oz-amber:#f59e0b;--oz-red:#ef4444;--oz-green:#22c55e;
  --oz-text:#e0eaf4;--oz-muted:#7a9abf;
}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif!important;background-color:var(--oz-bg)!important;color:var(--oz-text)!important;}
.stApp,.main .block-container{background-color:var(--oz-bg)!important;}
h1{font-size:2rem!important;font-weight:700!important;letter-spacing:-0.5px;
  background:linear-gradient(90deg,#00d4ff 0%,#3b82f6 60%,#818cf8 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;padding-bottom:4px;}
h2{color:var(--oz-cyan)!important;font-weight:600!important;border-bottom:1px solid var(--oz-border);padding-bottom:6px;margin-bottom:12px;}
h3{color:#7dd3fc!important;font-weight:500!important;}
[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background-color:#101820!important;border-right:1px solid var(--oz-border)!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--oz-cyan)!important;font-size:0.85rem!important;text-transform:uppercase;letter-spacing:0.08em;}
[data-testid="stMetric"]{background:var(--oz-panel)!important;border:1px solid var(--oz-border)!important;border-radius:8px!important;padding:14px 16px!important;}
[data-testid="stMetricLabel"]{color:var(--oz-muted)!important;font-size:0.72rem!important;text-transform:uppercase;letter-spacing:0.07em;}
[data-testid="stMetricValue"]{color:var(--oz-cyan)!important;font-size:1.35rem!important;font-weight:700!important;}
[data-testid="stDataFrame"]{border:1px solid var(--oz-border)!important;border-radius:8px!important;overflow:hidden;}
.stButton>button,[data-testid="baseButton-secondary"]{
  background:linear-gradient(135deg,#0d2a3f 0%,#0d1f30 100%)!important;
  border:1px solid var(--oz-cyan-dim)!important;color:var(--oz-cyan)!important;
  border-radius:6px!important;font-weight:500!important;font-family:'Inter',sans-serif!important;
  letter-spacing:0.03em;transition:all 0.15s ease;}
.stButton>button:hover{background:linear-gradient(135deg,#0d3a55 0%,#0d2d44 100%)!important;
  border-color:var(--oz-cyan)!important;box-shadow:0 0 10px rgba(0,212,255,0.25)!important;}
[data-testid="baseButton-primary"]>button,button[kind="primary"]{
  background:linear-gradient(135deg,#005f7a 0%,#007a99 100%)!important;
  border:1.5px solid var(--oz-cyan)!important;color:#ffffff!important;
  font-size:1rem!important;font-weight:700!important;box-shadow:0 0 14px rgba(0,212,255,0.35)!important;}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#1a3a52 0%,#122840 100%)!important;
  border:1px solid var(--oz-blue)!important;color:#93c5fd!important;border-radius:6px!important;font-weight:500!important;}
[data-testid="stDownloadButton"]>button:hover{box-shadow:0 0 10px rgba(59,130,246,0.3)!important;}
[data-testid="stFileUploader"]{border:1px dashed var(--oz-border)!important;border-radius:8px!important;background:var(--oz-panel)!important;padding:6px;}
[data-testid="stExpander"]{border:1px solid var(--oz-border)!important;border-radius:8px!important;background:var(--oz-panel)!important;}
[data-testid="stAlert"][data-baseweb="notification"]{border-radius:8px!important;}
.stInfo{background:rgba(0,212,255,0.08)!important;border-left:3px solid var(--oz-cyan)!important;}
.stWarning{background:rgba(245,158,11,0.10)!important;border-left:3px solid var(--oz-amber)!important;}
.stError{background:rgba(239,68,68,0.10)!important;border-left:3px solid var(--oz-red)!important;}
.stSuccess{background:rgba(34,197,94,0.08)!important;border-left:3px solid var(--oz-green)!important;}
[data-testid="stSelectbox"]>div>div,
[data-testid="stNumberInput"]>div>div>input,
[data-testid="stTextInput"]>div>div>input{
  background-color:var(--oz-panel2)!important;border:1px solid var(--oz-border)!important;
  color:var(--oz-text)!important;border-radius:6px!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{border-bottom:1px solid var(--oz-border)!important;background:transparent!important;}
[data-testid="stTabs"] [data-baseweb="tab"]{color:var(--oz-muted)!important;font-weight:500!important;}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]{color:var(--oz-cyan)!important;border-bottom:2px solid var(--oz-cyan)!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--oz-cyan)!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--oz-bg);}
::-webkit-scrollbar-thumb{background:var(--oz-border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--oz-cyan-dim);}
</style>
""", unsafe_allow_html=True)

# ─── TITOLO ─────────────────────────────────────────────────────────────
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

**2. Metrica di Ranking IS** — Omega Ratio (default), Sharpe, Sortino, Ulcer Index, ROC.

**3. CUSUM Weekday Killer** — Blacklist weekday per combinazioni strategia/giorno anomale.

**4. Anti-overlap + pesi** — Top-N strategie, max per gruppo, pesi pieno/panchina configurabili.

**5. Grid Search** — Testa automaticamente 4.320 combinazioni di parametri e mostra le top-20
   per la metrica obiettivo scelta (Calmar, Sharpe, Sortino, Omega, Net PnL).
   Grazie alla **cache IS pre-calcolata** l'intera ricerca richiede solo 1-3 minuti su Streamlit Cloud.

---
### 💾 Backup
**📤 Esporta JSON** → salva. Al prossimo accesso: **📥 Importa** → **✅ Applica**.
        """)

# ─── COSTANTI ──────────────────────────────────────────────────────────────
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
OOS_STEP_DAYS = 28
DEFAULT_MAX_PER_GROUP     = 2
DEFAULT_TOP_N             = 7
DEFAULT_FULL_WEIGHT_COUNT = 5
DEFAULT_FULL_WEIGHT_PCT   = 100
DEFAULT_BENCH_WEIGHT_PCT  = 50
DEFAULT_RANKING_METRIC    = "Omega Ratio"
DEFAULT_ROC_STEPS         = 2
DEFAULT_ROC_FILTER_ENABLED= False
DEFAULT_ROC_FILTER_STEPS  = 2
DEFAULT_GS_OBJECTIVE      = "Calmar Ratio"

RANKING_METRICS  = ["Omega Ratio", "Sharpe Ratio", "Sortino Ratio", "Ulcer Index", "ROC"]
GS_OBJECTIVES    = ["Calmar Ratio", "Sharpe Ratio", "Sortino Ratio", "Omega Ratio", "Net PnL"]
# Griglia fissa Grid Search (no ROC)
GS_FWP_VALUES    = [50, 75, 100]   # full weight %
GS_BWP_VALUES    = [25, 50, 75]    # bench weight %
GS_MPG_VALUES    = [1, 2]          # max per gruppo
GS_METRICS       = ["Omega Ratio", "Sharpe Ratio", "Sortino Ratio", "Ulcer Index"]
GS_TOP_N_VALUES  = list(range(3, 11))  # 3..10

OZONE_LAYOUT = dict(
    plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
    font=dict(family='Inter, Segoe UI, sans-serif', color='#e0eaf4', size=12),
    xaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    yaxis=dict(gridcolor='#2a3f5f', linecolor='#2a3f5f', zerolinecolor='#2a3f5f'),
    margin=dict(l=50, r=30, t=60, b=50),
)

# ─── FUNZIONI METRICHE IS ─────────────────────────────────────────────────────
def _omega_from_arrays(pct_arr, pnl_arr):
    valid = pct_arr[~np.isnan(pct_arr)]
    if len(valid) > 0:
        pos = valid[valid > 0].sum()
        neg = valid[valid < 0].sum()
    else:
        pos = pnl_arr[pnl_arr > 0].sum()
        neg = pnl_arr[pnl_arr < 0].sum()
    return 50.0 if neg == 0 else float(pos / abs(neg))

def _sharpe_from_daily(daily_arr):
    if len(daily_arr) < 2: return -999.0
    m, s = daily_arr.mean(), daily_arr.std(ddof=1)
    return float((m / s) * np.sqrt(252)) if s > 0 else -999.0

def _sortino_from_daily(daily_arr):
    if len(daily_arr) < 2: return -999.0
    down = daily_arr[daily_arr < 0]
    ds = down.std(ddof=1) if len(down) > 1 else 0.0
    return float((daily_arr.mean() / ds) * np.sqrt(252)) if ds > 0 else -999.0

def _ulcer_from_daily(daily_arr):
    cum  = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    denom = np.abs(peak) + 1e-9
    dd   = ((cum - peak) / denom) * 100
    ui   = float(np.sqrt(np.mean(dd ** 2)))
    return ui if ui > 0 else 0.0

def _roc_from_pnl(pnl_arr_sorted_dates, date_arr, cutoff_date):
    mask = date_arr > cutoff_date
    return float(pnl_arr_sorted_dates[mask].sum()) if mask.any() else -999.0

# ─── CACHE IS PRE-CALCOLATA ─────────────────────────────────────────────────────
def build_is_cache(df_filtered, strategy_mapping):
    """
    Pre-calcola per ogni finestra OOS e per ogni strategia:
      - omega, sharpe, sortino, ulcer
      - cusum_banned (lista di weekday bloccati)
      - net_pnl_is (totale $)
      - active_days (set di weekday con trade)
    Ritorna un dict: cache[(oos_start, strategy)] = {...}
    """
    min_date  = df_filtered['Date Closed'].min()
    max_date  = df_filtered['Date Closed'].max()
    oos_start = min_date + pd.DateOffset(months=12)
    cache     = {}

    # Pre-group per strategia per velocità
    grouped = {s: g for s, g in df_filtered.groupby('Strategy')}

    while oos_start <= max_date:
        is_start = oos_start - pd.DateOffset(months=12)

        for strat, g in grouped.items():
            if strat not in strategy_mapping: continue
            is_data = g[(g['Date Closed'] >= is_start) & (g['Date Closed'] < oos_start)]
            if len(is_data) < 5: continue

            pnl_arr  = is_data['Net PnL'].values.astype(float)
            pct_arr  = is_data['P/L %'].values.astype(float)
            dc_arr   = is_data['Date Closed'].values  # numpy datetime64
            wd_arr   = is_data['weekday_open'].values.astype(int)

            # Aggregazione giornaliera vettorizzata per Sharpe/Sortino/Ulcer
            dates_dt = pd.to_datetime(dc_arr).normalize()
            daily_s  = pd.Series(pnl_arr, index=dates_dt).groupby(level=0).sum()
            daily_arr= daily_s.values.astype(float)

            omega   = _omega_from_arrays(pct_arr, pnl_arr)
            sharpe  = _sharpe_from_daily(daily_arr)
            sortino = _sortino_from_daily(daily_arr)
            ulcer   = _ulcer_from_daily(daily_arr)

            # CUSUM ban vettorizzato
            banned = []
            for day in range(7):
                mask_d = (wd_arr == day)
                if mask_d.sum() < 10: continue
                d_pnl = pnl_arr[mask_d]
                std   = d_pnl.std()
                if std <= 0: continue
                k, h  = 0.5 * std, 3.0 * std
                c_low = 0.0
                for v in d_pnl:
                    c_low = max(0.0, c_low - v - k)
                if c_low > h:
                    banned.append(day)

            active_days = list(np.unique(wd_arr))
            remaining   = [d for d in active_days if d not in banned]

            cache[(oos_start, strat)] = {
                'omega':       omega,
                'sharpe':      sharpe,
                'sortino':     sortino,
                'ulcer':       ulcer,
                'net_pnl_is':  float(pnl_arr.sum()),
                'banned':      banned,
                'active_days': active_days,
                'remaining':   remaining,
                'group':       strategy_mapping[strat],
            }

        oos_start = oos_start + pd.Timedelta(days=OOS_STEP_DAYS)

    return cache

# ─── MOTORE WFA CON CACHE ────────────────────────────────────────────────────────
def run_wfa_from_cache(df_filtered, cache, top_n, fwc, fwp, bwp, mpg, metric):
    """
    Esegue il WFA usando la cache IS pre-calcolata.
    Ritorna (final_oos_df, hist_alloc_df, exclusions_df).
    metric: stringa tra Omega Ratio / Sharpe Ratio / Sortino Ratio / Ulcer Index
    """
    metric_key = {'Omega Ratio': 'omega', 'Sharpe Ratio': 'sharpe',
                  'Sortino Ratio': 'sortino', 'Ulcer Index': 'ulcer'}[metric]
    higher_better = metric != 'Ulcer Index'

    min_date  = df_filtered['Date Closed'].min()
    max_date  = df_filtered['Date Closed'].max()
    oos_start = min_date + pd.DateOffset(months=12)

    oos_results, hist_alloc, excl_log = [], [], []

    # Pre-index OOS data per velocità
    df_filtered_idx = df_filtered.set_index('Date Closed').sort_index()

    while oos_start <= max_date:
        oos_end   = oos_start + pd.Timedelta(days=OOS_STEP_DAYS)
        oos_slice = df_filtered_idx.loc[
            (df_filtered_idx.index >= oos_start) & (df_filtered_idx.index < oos_end)
        ].reset_index()

        # Raccogli metriche dalla cache
        candidates = []
        for (k_oos, k_strat), v in cache.items():
            if k_oos != oos_start: continue
            if not v['remaining']:
                excl_log.append({'OOS Start': oos_start, 'Strategy': k_strat,
                    'Motivo': f"Tutti i giorni bannati: {format_banned_days(v['banned'])}"})
                continue
            candidates.append({
                'strategy': k_strat, 'group': v['group'],
                'score': v[metric_key], 'net_pnl_is': v['net_pnl_is'],
                'banned': v['banned']
            })

        if candidates:
            candidates.sort(key=lambda x: (x['score'] if higher_better else -x['score'],
                                           x['net_pnl_is']), reverse=True)
            selected, gc = [], {}
            for c in candidates:
                if len(selected) >= top_n: break
                if gc.get(c['group'], 0) < mpg:
                    selected.append(c)
                    gc[c['group']] = gc.get(c['group'], 0) + 1

            for rank, c in enumerate(selected):
                weight = fwp if rank < fwc else bwp
                hist_alloc.append({
                    'OOS Start': oos_start, 'OOS End': oos_end,
                    'Rank': rank+1, 'Strategy': c['strategy'], 'Group': c['group'],
                    'Score IS': round(c['score'], 4), 'Metric': metric,
                    'Weight': weight, 'Banned Days': c['banned'],
                    'Top-N': top_n, 'Max Per Group': mpg,
                })
                st_oos = oos_slice[oos_slice['Strategy'] == c['strategy']].copy()
                if not st_oos.empty:
                    st_oos = st_oos[~st_oos['weekday_open'].isin(c['banned'])]
                    st_oos['Weighted Net PnL'] = st_oos['Net PnL'] * weight
                    oos_results.append(st_oos)

        oos_start = oos_end

    final_oos = pd.concat(oos_results).sort_values('Date Closed').reset_index(drop=True) \
        if oos_results else pd.DataFrame()
    return final_oos, pd.DataFrame(hist_alloc), pd.DataFrame(excl_log)

# ─── METRICHE OOS AGGREGATE ─────────────────────────────────────────────────────
def clean_money(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    return float(str(val).replace('$','').replace(',',''))

def _oos_scalar_metrics(df):
    """Ritorna dict con Calmar/Sharpe/Sortino/Omega/NetPnL dalla colonna Weighted Net PnL."""
    if df.empty or 'Weighted Net PnL' not in df.columns: return None
    pnl = df['Weighted Net PnL'].values.astype(float)
    daily_s = pd.Series(pnl, index=pd.to_datetime(df['Date Closed']).dt.normalize()).groupby(level=0).sum()
    daily   = daily_s.values.astype(float)
    if len(daily) < 2: return None
    total   = float(pnl.sum())
    cum     = np.cumsum(daily)
    peak    = np.maximum.accumulate(cum)
    max_dd  = float(np.max(peak - cum))
    m, s    = daily.mean(), daily.std(ddof=1)
    down    = daily[daily < 0]
    ds      = down.std(ddof=1) if len(down) > 1 else 0.0
    sharpe  = float((m/s)*np.sqrt(252)) if s > 0 else 0.0
    sortino = float((m/ds)*np.sqrt(252)) if ds > 0 else 0.0
    days    = (daily_s.index.max() - daily_s.index.min()).days
    calmar  = float((total*(365.25/days))/max_dd) if max_dd > 0 and days > 0 else 0.0
    pos_sum = float(pnl[pnl > 0].sum())
    neg_sum = float(pnl[pnl < 0].sum())
    omega   = 50.0 if neg_sum == 0 else float(pos_sum/abs(neg_sum))
    return {'Net PnL': total, 'Max Drawdown ($)': max_dd, 'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino, 'Calmar Ratio': calmar, 'Omega Ratio': omega}

def calculate_metrics(df):
    if df.empty: return {}
    pnl = df['Weighted Net PnL']
    total_pnl   = pnl.sum()
    total_trades= len(pnl)
    wins, losses= pnl[pnl > 0], pnl[pnl < 0]
    win_rate    = len(wins)/total_trades if total_trades > 0 else 0
    avg_win     = wins.mean() if len(wins) > 0 else 0
    avg_loss    = losses.mean() if len(losses) > 0 else 0
    pf          = wins.sum()/abs(losses.sum()) if len(losses)>0 and losses.sum()!=0 else np.inf
    expectancy  = (win_rate*avg_win)+((1-win_rate)*avg_loss)
    daily_pnl   = df.groupby(df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
    cum_pnl     = daily_pnl.cumsum()
    max_dd      = (cum_pnl.cummax()-cum_pnl).max()
    m, s        = daily_pnl.mean(), daily_pnl.std()
    down        = daily_pnl[daily_pnl < 0].std()
    sharpe      = (m/s)*np.sqrt(252) if s > 0 else 0
    sortino     = (m/down)*np.sqrt(252) if pd.notna(down) and down > 0 else 0
    days        = (daily_pnl.index.max()-daily_pnl.index.min()).days
    calmar      = (total_pnl*(365.25/days))/max_dd if max_dd > 0 and days > 0 else 0
    rf          = total_pnl/max_dd if max_dd > 0 else np.inf
    return {
        'Total Weighted PnL': total_pnl,'Total Trades': total_trades,'Win Rate': win_rate,
        'Profit Factor': pf,'Avg Win ($)': avg_win,'Avg Loss ($)': avg_loss,
        'Expectancy ($)': expectancy,'Max Drawdown ($)': max_dd,'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,'Calmar Ratio': calmar,'Recovery Factor': rf,
        'Avg Daily PnL ($)': daily_pnl.mean(),'Best Day ($)': daily_pnl.max(),'Worst Day ($)': daily_pnl.min(),
    }

def render_metrics(m, label):
    st.subheader(label)
    if not m: st.info("Dati insufficienti."); return
    r1 = st.columns(4)
    r1[0].metric("Weighted Net PnL", f"${m['Total Weighted PnL']:,.2f}")
    r1[1].metric("Totale Trade",     f"{int(m['Total Trades'])}")
    r1[2].metric("Win Rate",          f"{m['Win Rate']*100:.1f}%")
    r1[3].metric("Profit Factor",     f"{m['Profit Factor']:.2f}" if m['Profit Factor']!=np.inf else "∞")
    r2 = st.columns(4)
    r2[0].metric("Avg Win",     f"${m['Avg Win ($)']:,.2f}")
    r2[1].metric("Avg Loss",    f"${m['Avg Loss ($)']:,.2f}")
    r2[2].metric("Expectancy",  f"${m['Expectancy ($)']:,.2f}")
    r2[3].metric("Max Drawdown",f"${m['Max Drawdown ($)']:,.2f}")
    r3 = st.columns(4)
    r3[0].metric("Sharpe Ratio",    f"{m['Sharpe Ratio']:.2f}")
    r3[1].metric("Sortino Ratio",   f"{m['Sortino Ratio']:.2f}")
    r3[2].metric("Calmar Ratio",    f"{m['Calmar Ratio']:.2f}")
    r3[3].metric("Recovery Factor", f"{m['Recovery Factor']:.2f}" if m['Recovery Factor']!=np.inf else "∞")
    r4 = st.columns(3)
    r4[0].metric("Avg Daily PnL",f"${m['Avg Daily PnL ($)']:,.2f}")
    r4[1].metric("Best Day",     f"${m['Best Day ($)']:,.2f}")
    r4[2].metric("Worst Day",    f"${m['Worst Day ($)']:,.2f}")

def format_banned_days(bl):
    if not bl: return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in bl if d < len(GIORNI_SETTIMANA)])

# ─── FUNZIONI JSON CONFIG ────────────────────────────────────────────────────
def build_config_payload():
    return {
        "version": 4,
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
        "gs_objective":         st.session_state.get("gs_objective",             DEFAULT_GS_OBJECTIVE),
    }

def apply_config_payload(payload, all_strategies):
    if not isinstance(payload, dict): raise ValueError("JSON non valido.")
    gn = payload.get("group_names", [])
    sm = payload.get("strategy_mapping", {})
    if not isinstance(gn, list):  raise ValueError("group_names deve essere una lista.")
    if not isinstance(sm, dict):  raise ValueError("strategy_mapping deve essere un dizionario.")
    cg  = sorted({str(g).strip() for g in gn if str(g).strip()})
    cm  = {str(s).strip(): str(g).strip() for s, g in sm.items() if str(s).strip()}
    mg  = sorted(set(cg) | {g for g in cm.values() if g})
    cur = st.session_state.get("strategy_mapping", {}).copy()
    cur.update(cm)
    for s in all_strategies: cur.setdefault(s, "")
    st.session_state["group_names"]       = mg
    st.session_state["strategy_mapping"]  = cur
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
    gs_obj = payload.get("gs_objective", DEFAULT_GS_OBJECTIVE)
    st.session_state["gs_objective"] = gs_obj if gs_obj in GS_OBJECTIVES else DEFAULT_GS_OBJECTIVE

# ─── INIT SESSION STATE ────────────────────────────────────────────────────────────
for key, default in [
    ("top_n",             DEFAULT_TOP_N),
    ("full_weight_count", DEFAULT_FULL_WEIGHT_COUNT),
    ("full_weight_pct",   DEFAULT_FULL_WEIGHT_PCT),
    ("bench_weight_pct",  DEFAULT_BENCH_WEIGHT_PCT),
    ("ranking_metric",    DEFAULT_RANKING_METRIC),
    ("roc_steps",         DEFAULT_ROC_STEPS),
    ("roc_filter_enabled",DEFAULT_ROC_FILTER_ENABLED),
    ("roc_filter_steps",  DEFAULT_ROC_FILTER_STEPS),
    ("gs_objective",      DEFAULT_GS_OBJECTIVE),
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

    # ─── SIDEBAR ────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configurazione")
    st.sidebar.subheader("💾 Backup")
    st.sidebar.caption("Esporta JSON per salvare tutto. Reimporta dopo ogni refresh.")
    config_json_str = json.dumps(build_config_payload(), ensure_ascii=False, indent=2)
    st.sidebar.download_button("📤 Esporta configurazione JSON",
        data=config_json_str.encode("utf-8"), file_name="texano_config.json", mime="application/json")
    uploaded_config = st.sidebar.file_uploader("📥 Importa configurazione JSON", type=["json"], key="config_json_uploader")
    if uploaded_config is not None:
        try:
            parsed = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.session_state["pending_config_payload"] = parsed
            st.sidebar.success("✅ JSON caricato. Premi 'Applica'.")
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

    st.sidebar.subheader("🎛️ Parametri WFA")
    top_n = st.sidebar.number_input("Top-N strategie",
        min_value=1, max_value=20, value=st.session_state["top_n"], step=1)
    st.session_state["top_n"] = int(top_n)
    full_weight_count = st.sidebar.number_input("Strategie peso pieno (rank 1…N)",
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

    st.sidebar.subheader("🏆 Metrica di Ranking IS")
    ranking_metric_idx = RANKING_METRICS.index(st.session_state["ranking_metric"]) \
        if st.session_state["ranking_metric"] in RANKING_METRICS else 0
    ranking_metric = st.sidebar.selectbox("Metrica ranking IS",
        options=RANKING_METRICS, index=ranking_metric_idx)
    st.session_state["ranking_metric"] = ranking_metric
    roc_steps = st.session_state["roc_steps"]
    if ranking_metric == "ROC":
        roc_steps = st.sidebar.number_input(f"Finestra ROC (multipli {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, value=roc_steps, step=1)
        st.session_state["roc_steps"] = int(roc_steps)
        st.sidebar.caption(f"📅 ROC attiva: **{int(roc_steps)*OOS_STEP_DAYS}gg**")
    st.sidebar.divider()

    st.sidebar.subheader("🚫 Filtro ROC < 0")
    roc_filter_enabled = st.sidebar.toggle("Attiva filtro ROC < 0",
        value=st.session_state["roc_filter_enabled"])
    st.session_state["roc_filter_enabled"] = roc_filter_enabled
    roc_filter_steps = st.session_state["roc_filter_steps"]
    if roc_filter_enabled:
        roc_filter_steps = st.sidebar.number_input(f"Finestra filtro (multipli {OOS_STEP_DAYS}gg)",
            min_value=1, max_value=24, value=roc_filter_steps, step=1)
        st.session_state["roc_filter_steps"] = int(roc_filter_steps)
        st.sidebar.caption(f"📅 Filtro ROC: **{int(roc_filter_steps)*OOS_STEP_DAYS}gg**")
    st.sidebar.divider()

    st.sidebar.subheader("Limite per gruppo")
    max_per_group = st.sidebar.number_input("Max strategie stesso gruppo",
        min_value=1, max_value=int(top_n),
        value=min(st.session_state['max_per_group'], int(top_n)), step=1)
    st.session_state['max_per_group'] = int(max_per_group)
    st.sidebar.divider()

    st.sidebar.subheader("Gestione gruppi")
    new_group_name = st.sidebar.text_input("Nome nuovo gruppo", key="new_group_name")
    if st.sidebar.button("➕ Crea gruppo"):
        cn = new_group_name.strip()
        if cn:
            if cn not in st.session_state['group_names']:
                st.session_state['group_names'].append(cn)
                st.session_state['group_names'] = sorted(st.session_state['group_names'])
                st.sidebar.success(f"Gruppo '{cn}' creato")
                st.rerun()
            else: st.sidebar.warning("Gruppo già esistente")
        else: st.sidebar.warning("Inserisci un nome valido")
    if st.session_state['group_names']:
        gtr = st.sidebar.selectbox("Gruppo da rinominare", options=st.session_state['group_names'], key="group_to_rename")
        rg  = st.sidebar.text_input("Nuovo nome", key="renamed_group")
        if st.sidebar.button("✏️ Rinomina gruppo"):
            nn = rg.strip()
            if not nn: st.sidebar.warning("Inserisci un nome valido")
            elif nn in st.session_state['group_names'] and nn != gtr: st.sidebar.warning("Nome già esistente")
            else:
                st.session_state['group_names'] = [nn if g == gtr else g for g in st.session_state['group_names']]
                st.session_state['strategy_mapping'] = {s:(nn if g==gtr else g) for s,g in st.session_state['strategy_mapping'].items()}
                st.sidebar.success(f"'{gtr}' → '{nn}'")
                st.rerun()
    st.sidebar.divider()

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
            st.sidebar.success("Assegnazioni aggiornate — esporta JSON!")
            st.rerun()

    strategy_mapping = st.session_state['strategy_mapping']
    still_unmapped = [s for s in all_strategies_in_file if not strategy_mapping.get(s,'').strip()]
    if still_unmapped:
        st.error(f"🛑 BLOCCO DI SICUREZZA: Strategie senza gruppo: {', '.join(still_unmapped)}")
        st.info("Apri la sidebar, assegna un gruppo e salva.")
        st.stop()

    # ─── PREPROCESSING DATI (condiviso tra WFA e Grid Search) ────────────────
    @st.cache_data(show_spinner=False)
    def preprocess_df(file_id):
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df[~df['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()
        df['Date Opened'] = pd.to_datetime(df['Date Opened'])
        df['Date Closed'] = pd.to_datetime(df['Date Closed'])
        df['weekday_open'] = df['Date Opened'].dt.weekday
        df['P/L'] = df['P/L'].apply(clean_money)
        df['P/L %'] = df['P/L %'].apply(
            lambda x: float(str(x).replace('%','').replace(',','')) if pd.notna(x) else np.nan)
        df['Opening Commissions + Fees'] = df['Opening Commissions + Fees'].fillna(0).apply(clean_money)
        df['Closing Commissions + Fees'] = df['Closing Commissions + Fees'].fillna(0).apply(clean_money)
        df['Net PnL'] = df['P/L'] - df['Opening Commissions + Fees'] - df['Closing Commissions + Fees']
        return df.sort_values('Date Closed').reset_index(drop=True)

    df_filtered = preprocess_df(uploaded_file.file_id)

    # ─── PULSANTE LANCIA WFA + TABS ────────────────────────────────────────
    st.markdown("---")
    col_launch, col_info = st.columns([3, 7])
    with col_launch:
        launch = st.button("▶ Lancia simulazione WFA", type="primary", use_container_width=True)
    with col_info:
        rfl = (f" &nbsp;| 🚫 ROC filter <b style='color:#ef4444'>{int(roc_filter_steps)*OOS_STEP_DAYS}gg</b>")\
              if roc_filter_enabled else ""
        rc  = "#f59e0b" if ranking_metric != "Omega Ratio" else "#00d4ff"
        st.markdown(
            f"<div style='padding-top:10px;color:#7a9abf;font-size:0.88rem;'>"
            f"Ranking: <b style='color:{rc}'>{ranking_metric}</b>"
            + (f" ({int(roc_steps)*OOS_STEP_DAYS}gg)" if ranking_metric=="ROC" else "")
            + f" &nbsp;| Top-N: <b style='color:#00d4ff'>{int(top_n)}</b>"
            f" &nbsp;| Pesi: <b style='color:#00d4ff'>{int(full_weight_count)}×{int(full_weight_pct)}%</b>"
            f" + <b style='color:#f59e0b'>{bench_count}×{int(bench_weight_pct)}%</b>"
            f" &nbsp;| Max/gr: <b style='color:#00d4ff'>{int(max_per_group)}</b>{rfl}</div>",
            unsafe_allow_html=True)

    if launch:
        _top_n  = int(st.session_state["top_n"])
        _fwc    = int(st.session_state["full_weight_count"])
        _fwp    = st.session_state["full_weight_pct"] / 100.0
        _bwp    = st.session_state["bench_weight_pct"] / 100.0
        _mpg    = int(st.session_state["max_per_group"])
        _sm     = dict(st.session_state["strategy_mapping"])
        _metric = st.session_state["ranking_metric"]
        _roc_s  = int(st.session_state["roc_steps"])
        _roc_fe = bool(st.session_state["roc_filter_enabled"])
        _roc_fs = int(st.session_state["roc_filter_steps"])

        with st.spinner("Pre-calcolo cache IS…"):
            cache = build_is_cache(df_filtered, _sm)
        with st.spinner("Esecuzione WFA…"):
            final_oos, hist_alloc, excl_df = run_wfa_from_cache(
                df_filtered, cache, _top_n, _fwc, _fwp, _bwp, _mpg, _metric
            )
        if final_oos.empty:
            st.error("Nessun trade OOS. Dataset troppo corto (< 13 mesi)?")
        else:
            st.session_state["wfa_results"] = {
                "final_oos_df":  final_oos,
                "hist_alloc_df": hist_alloc,
                "exclusions_df": excl_df,
                "is_cache":      cache,
                "params": {"top_n": _top_n, "full_weight_count": _fwc,
                           "full_weight_pct": int(_fwp*100), "bench_weight_pct": int(_bwp*100),
                           "max_per_group": _mpg, "ranking_metric": _metric,
                           "roc_steps": _roc_s, "roc_filter_enabled": _roc_fe, "roc_filter_steps": _roc_fs}
            }
            st.success("✅ Elaborazione completata!")

    # ─── TABS RISULTATI ─────────────────────────────────────────────────────────
    tab_wfa, tab_gs = st.tabs(["📊 WFA — Risultati", "🔍 Grid Search Ottimizzazione"])

    # ─────────────────────────────────────────────────────────────────
    with tab_wfa:
        if st.session_state.get("wfa_results") is None:
            st.info("▶ Premi il pulsante sopra per lanciare la simulazione WFA.")
        else:
            res           = st.session_state["wfa_results"]
            final_oos_df  = res["final_oos_df"]
            hist_alloc_df = res["hist_alloc_df"]
            exclusions_df = res["exclusions_df"]
            run_params    = res["params"]

            st.header("1. Allocazione Corrente")
            latest_start = hist_alloc_df['OOS Start'].max()
            latest_end   = hist_alloc_df[hist_alloc_df['OOS Start']==latest_start]['OOS End'].iloc[0]
            today        = pd.Timestamp.today().normalize()
            days_left    = (latest_end - today).days
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.info(f"📅 **Dal:** {latest_start.strftime('%d %b %Y')}")
            c2.info(f"🔚 **Fino al:** {latest_end.strftime('%d %b %Y')}")
            c3.warning(f"⏳ **Tra:** {days_left}gg") if days_left>0 else c3.error("🔄 Scaduta")
            c4.info(f"🏆 **Ranking:** {run_params['ranking_metric']}")
            c5.info(f"🎛️ Top-N: **{run_params['top_n']}** | Max/gr: **{run_params['max_per_group']}**")
            c6.info(f"⚖️ **{run_params['full_weight_count']}×{run_params['full_weight_pct']}%** + "
                    f"**{run_params['top_n']-run_params['full_weight_count']}×{run_params['bench_weight_pct']}%**")

            la = hist_alloc_df[hist_alloc_df['OOS Start']==latest_start].copy()
            disp = la[['Rank','Strategy','Group','Score IS','Metric','Weight','Banned Days']].copy()
            disp.rename(columns={'Score IS': f"Score ({run_params['ranking_metric']})", 'Metric':'Metrica'}, inplace=True)
            disp['Weight'] = (disp['Weight']*100).round(0).astype(int).astype(str)+'%'
            disp['Giorni Spenti'] = disp['Banned Days'].apply(format_banned_days)
            disp.drop(columns=['Banned Days','Metrica'], inplace=True)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            if not exclusions_df.empty:
                le = exclusions_df[exclusions_df['OOS Start']==latest_start]
                if not le.empty:
                    st.warning("⚠️ Escluse (CUSUM — tutti i giorni bannati):")
                    st.dataframe(le[['Strategy','Motivo']], use_container_width=True, hide_index=True)

            st.header("2. Metriche Out-Of-Sample")
            t_glob, t_rec = st.tabs(["📊 Storico Completo", "📈 Dal 1° Set 2025"])
            with t_glob:
                render_metrics(calculate_metrics(final_oos_df), "Metriche — Storico OOS")
            with t_rec:
                rec = final_oos_df[final_oos_df['Date Closed']>=pd.to_datetime('2025-09-01')].copy()
                render_metrics(calculate_metrics(rec), "Metriche — Dal 1° Set 2025")

            st.header("3. Curva Equity — Storico Completo")
            ad = final_oos_df.groupby(final_oos_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
            ac = ad.cumsum()
            fig_a = go.Figure(go.Scatter(x=ac.index, y=ac.values, mode='lines', fill='tozeroy',
                line=dict(color='#00d4ff', width=2), fillcolor='rgba(0,212,255,0.08)'))
            fig_a.update_layout(title="Equity Cumulativa WFA — Storico",
                xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
            st.plotly_chart(fig_a, use_container_width=True)

            st.header("4. Curva Equity — Dal 1° Set 2025")
            rec2 = final_oos_df[final_oos_df['Date Closed']>=pd.to_datetime('2025-09-01')].copy()
            if not rec2.empty:
                rd = rec2.groupby(rec2['Date Closed'].dt.date)['Weighted Net PnL'].sum()
                rc2= rd.cumsum()
                fig_r = go.Figure(go.Scatter(x=rc2.index, y=rc2.values, mode='lines', fill='tozeroy',
                    line=dict(color='#f59e0b', width=2), fillcolor='rgba(245,158,11,0.08)'))
                fig_r.update_layout(title="Equity WFA — Dal 1° Set 2025",
                    xaxis_title="Data", yaxis_title="Net PnL ($)", **OZONE_LAYOUT)
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("Nessun dato OOS dal 1° settembre 2025.")

            st.header("5. Esporta Dati")
            he = hist_alloc_df.copy()
            he['Banned Days'] = he['Banned Days'].apply(lambda x: format_banned_days(x) if isinstance(x,list) else x)
            eq_csv = pd.DataFrame({'Date': ac.index, 'Daily PnL': ad.values, 'Cumulative PnL': ac.values})
            cb1,cb2,cb3,cb4 = st.columns(4)
            with cb1: st.download_button("📥 Allocazioni",   data=he.to_csv(index=False).encode(), file_name="wfa_allocations.csv", mime="text/csv")
            with cb2: st.download_button("📥 Trade OOS",     data=final_oos_df.to_csv(index=False).encode(), file_name="wfa_oos_trades.csv", mime="text/csv")
            with cb3: st.download_button("📥 Equity Completa", data=eq_csv.to_csv(index=False).encode(), file_name="equity_full.csv", mime="text/csv")
            with cb4:
                if not rec2.empty:
                    rc_csv = pd.DataFrame({'Date': rc2.index, 'Daily PnL': rd.values, 'Cumulative PnL': rc2.values})
                    st.download_button("📥 Equity Set 2025+", data=rc_csv.to_csv(index=False).encode(), file_name="equity_sep2025.csv", mime="text/csv")

    # ─────────────────────────────────────────────────────────────────
    with tab_gs:
        st.header("🔍 Grid Search — Ottimizzazione Parametri")
        st.markdown("""
        Testa automaticamente **4.320 combinazioni** di parametri usando la **cache IS pre-calcolata**
        (nessun ricalcolo ridondante). Seleziona la metrica obiettivo da massimizzare e premi il pulsante.
        """)

        gs_col1, gs_col2 = st.columns([3, 5])
        with gs_col1:
            gs_obj_idx = GS_OBJECTIVES.index(st.session_state["gs_objective"]) \
                if st.session_state["gs_objective"] in GS_OBJECTIVES else 0
            gs_objective = st.selectbox(
                "🎯 Metrica obiettivo da massimizzare",
                options=GS_OBJECTIVES, index=gs_obj_idx,
                help="La Grid Search troverà la combinazione di parametri che massimizza questa metrica sull'intero OOS."
            )
            st.session_state["gs_objective"] = gs_objective
        with gs_col2:
            n_combos = sum(
                (tn+1) for tn in GS_TOP_N_VALUES
            ) * len(GS_FWP_VALUES) * len(GS_BWP_VALUES) * len(GS_MPG_VALUES) * len(GS_METRICS)
            st.markdown(
                f"<div style='padding-top:28px;color:#7a9abf;font-size:0.9rem;'>"
                f"📋 Spazio: <b style='color:#00d4ff'>{n_combos:,} combinazioni</b> &nbsp;| "
                f"Metriche IS: <b style='color:#f59e0b'>{', '.join(GS_METRICS)}</b><br>"
                f"Top-N: 3–10 &nbsp;| FW%: {GS_FWP_VALUES} &nbsp;| BW%: {GS_BWP_VALUES} &nbsp;| Max/gr: {GS_MPG_VALUES}"
                f"</div>", unsafe_allow_html=True)

        run_gs = st.button("🔍 Lancia Grid Search", type="primary", use_container_width=False)

        if run_gs:
            if st.session_state.get("wfa_results") is None or "is_cache" not in st.session_state.get("wfa_results", {}):
                st.warning("⚠️ Prima di eseguire la Grid Search, lancia almeno una volta la simulazione WFA per costruire la cache IS.")
            else:
                cache = st.session_state["wfa_results"]["is_cache"]

                # Costruisci lista combinazioni (pruning: bench_weight irrilevante se fwc==top_n)
                combos = []
                for top_n_g in GS_TOP_N_VALUES:
                    for fwc_g in range(0, top_n_g + 1):
                        for fwp_g in GS_FWP_VALUES:
                            for bwp_g in GS_BWP_VALUES:
                                # Pruning: se tutte le strategie hanno peso pieno, bwp non conta
                                if fwc_g == top_n_g and bwp_g != GS_BWP_VALUES[0]:
                                    continue
                                for mpg_g in GS_MPG_VALUES:
                                    for metric_g in GS_METRICS:
                                        combos.append((top_n_g, fwc_g, fwp_g, bwp_g, mpg_g, metric_g))

                total_c  = len(combos)
                gs_results = []
                prog_bar = st.progress(0, text=f"0 / {total_c} combinazioni testate…")
                obj_key  = {'Calmar Ratio':'Calmar Ratio','Sharpe Ratio':'Sharpe Ratio',
                            'Sortino Ratio':'Sortino Ratio','Omega Ratio':'Omega Ratio','Net PnL':'Net PnL'}[gs_objective]

                for i, (tn, fwc, fwp, bwp, mpg, metric) in enumerate(combos):
                    fwp_f = fwp / 100.0
                    bwp_f = bwp / 100.0
                    final_oos_g, _, _ = run_wfa_from_cache(
                        df_filtered, cache, tn, fwc, fwp_f, bwp_f, mpg, metric
                    )
                    sc = _oos_scalar_metrics(final_oos_g)
                    if sc:
                        gs_results.append({
                            'Top-N': tn, 'FW Count': fwc, 'FW %': fwp, 'BW %': bwp,
                            'Max/Gr': mpg, 'Metrica IS': metric,
                            'Net PnL': round(sc['Net PnL'], 2),
                            'Calmar Ratio': round(sc['Calmar Ratio'], 3),
                            'Sharpe Ratio': round(sc['Sharpe Ratio'], 3),
                            'Sortino Ratio': round(sc['Sortino Ratio'], 3),
                            'Omega Ratio':  round(sc['Omega Ratio'], 3),
                            'Max DD ($)':   round(sc['Max Drawdown ($)'], 2),
                        })
                    if (i + 1) % 20 == 0 or i == total_c - 1:
                        prog_bar.progress((i+1)/total_c, text=f"{i+1} / {total_c} combinazioni testate…")

                prog_bar.empty()
                if gs_results:
                    gs_df = pd.DataFrame(gs_results).sort_values(obj_key, ascending=False).reset_index(drop=True)
                    gs_df.index += 1
                    st.session_state["gs_results_df"]   = gs_df
                    st.session_state["gs_results_obj"]  = gs_objective
                    st.success(f"✅ Grid Search completata! {len(gs_results):,} combinazioni valide su {total_c:,}.")
                else:
                    st.error("Nessun risultato valido dalla Grid Search.")

        if st.session_state.get("gs_results_df") is not None:
            gs_df  = st.session_state["gs_results_df"]
            gs_obj_disp = st.session_state.get("gs_results_obj", "")

            st.subheader(f"🏆 Top-20 combinazioni — ordinato per {gs_obj_disp}")
            st.dataframe(
                gs_df.head(20).style.highlight_max(
                    subset=[gs_obj_disp], color='rgba(0,212,255,0.18)'
                ),
                use_container_width=True
            )

            # Pulsante applica parametri ottimali
            best = gs_df.iloc[0]
            st.markdown(
                f"""
                <div style='background:var(--oz-panel);border:1px solid var(--oz-border);border-radius:8px;
                padding:14px 20px;margin:10px 0;'>
                <b style='color:#00d4ff'>🥇 Parametri ottimali ({gs_obj_disp} = {best[gs_obj_disp]:.3f})</b><br>
                <span style='color:#e0eaf4'>
                Top-N: <b>{int(best['Top-N'])}</b> &nbsp;| 
                FW Count: <b>{int(best['FW Count'])}</b> &nbsp;| 
                FW %: <b>{int(best['FW %'])}%</b> &nbsp;| 
                BW %: <b>{int(best['BW %'])}%</b> &nbsp;| 
                Max/Gr: <b>{int(best['Max/Gr'])}</b> &nbsp;| 
                Metrica IS: <b>{best['Metrica IS']}</b>
                </span></div>
                """, unsafe_allow_html=True)

            if st.button("✅ Applica parametri ottimali e rilancia WFA", type="primary"):
                st.session_state["top_n"]             = int(best['Top-N'])
                st.session_state["full_weight_count"] = int(best['FW Count'])
                st.session_state["full_weight_pct"]   = int(best['FW %'])
                st.session_state["bench_weight_pct"]  = int(best['BW %'])
                st.session_state["max_per_group"]     = int(best['Max/Gr'])
                st.session_state["ranking_metric"]    = best['Metrica IS']
                st.session_state["wfa_results"]       = None
                st.rerun()

            st.download_button("📥 Esporta tutti i risultati Grid Search",
                data=gs_df.to_csv(index=True, index_label='Rank').encode(),
                file_name="gridsearch_results.csv", mime="text/csv")

else:
    st.info("⬆️ Carica un file CSV per iniziare.")
