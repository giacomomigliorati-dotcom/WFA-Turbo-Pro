import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Texano's Walk Forward", layout="wide")

# ─── TITOLO + PULSANTE ? ─────────────────────────────────────────────────────
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
Questo strumento esegue l'aggiornamento mensile di un portafoglio algoritmico **rotazionale**
basato su un motore **Walk-Forward Analysis (WFA)** con finestra rolling di 12 mesi.

---

### ⚙️ Workflow passo per passo

**1. Preprocessing**
Il CSV caricato viene pulito automaticamente: si rimuovono le strategie *Legendary*, si calcolano
i **Net PnL** al netto delle commissioni, si estrae il giorno della settimana da ogni apertura.

**2. Mappatura gruppi personalizzabile**
Ogni strategia deve appartenere a un gruppo. I gruppi possono essere rinominati, se ne possono creare
di nuovi e ogni strategia può essere riassegnata via sidebar prima dell'elaborazione.
La configurazione dei gruppi può essere **esportata in JSON** e **reimportata** in qualsiasi nuova
sessione per non perdere le impostazioni dopo un refresh.

**3. Omega Ratio (ranking IS)**
Per ogni finestra In-Sample di 12 mesi, ogni strategia riceve un punteggio basato sull'**Omega Ratio**:
rapporto tra somma dei rendimenti positivi e valore assoluto dei negativi.

**4. Anomaly Detection CUSUM (Weekday Killer)**
Per ogni combinazione strategia/giorno con almeno 10 trade nella finestra IS, viene calcolata
una **somma cumulativa inferiore (CUSUM)**. Se la perdita cumulativa supera la soglia di allarme,
quel giorno viene inserito nella **Blacklist** per la finestra OOS successiva.
Se il CUSUM banna **tutti** i giorni su cui la strategia opera realmente, la strategia viene esclusa.

**5. Anti-overlap configurabile**
Le strategie qualificate vengono ordinate per Omega decrescente. Si seleziona un portafoglio
**Top-7** con un limite massimo configurabile di strategie per lo stesso gruppo.
- Rank 1–5 → Peso **100%**
- Rank 6–7 → Peso **50%**

**6. OOS Filtrato**
Nella finestra OOS di 28 giorni, i trade delle strategie selezionate vengono filtrati:
i trade che si aprono in un giorno blacklistato vengono rimossi. Il Net PnL rimanente
viene moltiplicato per il peso assegnato.

---

### 💾 Backup configurazione
Per non perdere gruppi e assegnazioni dopo il refresh:
1. Configura i tuoi gruppi nella sidebar.
2. Clicca **📤 Esporta configurazione JSON** e salva il file `texano_config.json`.
3. Al prossimo accesso, carica il file nella sidebar e clicca **✅ Applica configurazione JSON**.
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
DEFAULT_MAX_PER_GROUP = 2
TOP_N = 7

# ─── FUNZIONI CORE ───────────────────────────────────────────────────────────
def clean_money(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return val
    return float(str(val).replace('$', '').replace(',', ''))

def calculate_metrics(df):
    if df.empty:
        return {}
    pnl = df['Weighted Net PnL']
    total_pnl = pnl.sum()
    total_trades = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    daily_pnl = df.groupby(df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
    cum_pnl = daily_pnl.cumsum()
    max_dd = (cum_pnl.cummax() - cum_pnl).max()
    mean_d = daily_pnl.mean()
    std_d = daily_pnl.std()
    downside = daily_pnl[daily_pnl < 0].std()
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0
    sortino = (mean_d / downside) * np.sqrt(252) if pd.notna(downside) and downside > 0 else 0
    days = (daily_pnl.index.max() - daily_pnl.index.min()).days
    calmar = (total_pnl * (365.25 / days)) / max_dd if max_dd > 0 and days > 0 else 0
    recovery_factor = total_pnl / max_dd if max_dd > 0 else np.inf
    return {
        'Total Weighted PnL': total_pnl,
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Avg Win ($)': avg_win,
        'Avg Loss ($)': avg_loss,
        'Expectancy ($)': expectancy,
        'Max Drawdown ($)': max_dd,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Recovery Factor': recovery_factor,
        'Avg Daily PnL ($)': daily_pnl.mean(),
        'Best Day ($)': daily_pnl.max(),
        'Worst Day ($)': daily_pnl.min(),
    }

def render_metrics(m, label):
    st.subheader(label)
    if not m:
        st.info("Dati insufficienti.")
        return
    r1 = st.columns(4)
    r1[0].metric("Weighted Net PnL", f"${m['Total Weighted PnL']:,.2f}")
    r1[1].metric("Totale Trade", f"{int(m['Total Trades'])}")
    r1[2].metric("Win Rate", f"{m['Win Rate']*100:.1f}%")
    r1[3].metric("Profit Factor", f"{m['Profit Factor']:.2f}" if m['Profit Factor'] != np.inf else "∞")
    r2 = st.columns(4)
    r2[0].metric("Avg Win", f"${m['Avg Win ($)']:,.2f}")
    r2[1].metric("Avg Loss", f"${m['Avg Loss ($)']:,.2f}")
    r2[2].metric("Expectancy", f"${m['Expectancy ($)']:,.2f}")
    r2[3].metric("Max Drawdown", f"${m['Max Drawdown ($)']:,.2f}")
    r3 = st.columns(4)
    r3[0].metric("Sharpe Ratio", f"{m['Sharpe Ratio']:.2f}")
    r3[1].metric("Sortino Ratio", f"{m['Sortino Ratio']:.2f}")
    r3[2].metric("Calmar Ratio", f"{m['Calmar Ratio']:.2f}")
    r3[3].metric("Recovery Factor", f"{m['Recovery Factor']:.2f}" if m['Recovery Factor'] != np.inf else "∞")
    r4 = st.columns(3)
    r4[0].metric("Avg Daily PnL", f"${m['Avg Daily PnL ($)']:,.2f}")
    r4[1].metric("Best Day", f"${m['Best Day ($)']:,.2f}")
    r4[2].metric("Worst Day", f"${m['Worst Day ($)']:,.2f}")

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
    if not banned_list:
        return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in banned_list if d < len(GIORNI_SETTIMANA)])

# ─── FUNZIONI JSON CONFIG ─────────────────────────────────────────────────────
def build_config_payload():
    return {
        "version": 1,
        "group_names": st.session_state.get("group_names", []),
        "strategy_mapping": st.session_state.get("strategy_mapping", {}),
        "max_per_group": int(st.session_state.get("max_per_group", DEFAULT_MAX_PER_GROUP)),
    }

def apply_config_payload(payload, all_strategies):
    if not isinstance(payload, dict):
        raise ValueError("Il file JSON non contiene un oggetto valido.")
    group_names = payload.get("group_names", [])
    strategy_mapping = payload.get("strategy_mapping", {})
    max_per_group = payload.get("max_per_group", DEFAULT_MAX_PER_GROUP)
    if not isinstance(group_names, list):
        raise ValueError("group_names deve essere una lista.")
    if not isinstance(strategy_mapping, dict):
        raise ValueError("strategy_mapping deve essere un dizionario.")

    clean_groups = sorted({str(g).strip() for g in group_names if str(g).strip()})
    clean_mapping = {str(s).strip(): str(g).strip() for s, g in strategy_mapping.items() if str(s).strip()}
    used_groups = {g for g in clean_mapping.values() if g}
    merged_groups = sorted(set(clean_groups) | used_groups)

    current_mapping = st.session_state.get("strategy_mapping", {}).copy()
    current_mapping.update(clean_mapping)
    for strat in all_strategies:
        current_mapping.setdefault(strat, "")

    st.session_state["group_names"] = merged_groups
    st.session_state["strategy_mapping"] = current_mapping
    st.session_state["max_per_group"] = max(1, min(TOP_N, int(max_per_group)))

# ─── UPLOAD CSV ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw[~df_raw['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()
    all_strategies_in_file = sorted(df_raw['Strategy'].dropna().unique().tolist())

    # ─── STATO SESSIONE ──────────────────────────────────────────────────────
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
    st.sidebar.header("⚙️ Configurazione Gruppi")
    st.sidebar.markdown("Crea, rinomina e assegna gruppi; imposta il limite massimo per gruppo.")

    # ── BACKUP JSON ──────────────────────────────────────────────────────────
    st.sidebar.subheader("💾 Backup configurazione")
    st.sidebar.caption("Esporta per salvare i tuoi gruppi. Reimporta dopo ogni refresh o reboot.")

    config_json_str = json.dumps(build_config_payload(), ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        label="📤 Esporta configurazione JSON",
        data=config_json_str.encode("utf-8"),
        file_name="texano_config.json",
        mime="application/json",
        help="Salva gruppi, assegnazioni e max per gruppo in un file JSON."
    )

    uploaded_config = st.sidebar.file_uploader(
        "📥 Importa configurazione JSON",
        type=["json"],
        key="config_json_uploader",
        help="Carica un file texano_config.json esportato in precedenza."
    )
    if uploaded_config is not None:
        try:
            parsed = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.session_state["pending_config_payload"] = parsed
            st.sidebar.success("✅ JSON caricato. Premi 'Applica configurazione JSON' per attivarlo.")
        except Exception as e:
            st.sidebar.error(f"JSON non valido: {e}")

    if st.sidebar.button("✅ Applica configurazione JSON"):
        if "pending_config_payload" not in st.session_state:
            st.sidebar.warning("Carica prima un file JSON nella sezione qui sopra.")
        else:
            try:
                apply_config_payload(st.session_state["pending_config_payload"], all_strategies_in_file)
                del st.session_state["pending_config_payload"]
                st.sidebar.success("Configurazione applicata con successo!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Errore: {e}")

    st.sidebar.divider()

    # ── NUOVO GRUPPO ─────────────────────────────────────────────────────────
    st.sidebar.subheader("Nuovo gruppo")
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
                st.sidebar.warning("Questo gruppo esiste già")
        else:
            st.sidebar.warning("Inserisci un nome valido")

    # ── RINOMINA GRUPPO ──────────────────────────────────────────────────────
    st.sidebar.subheader("Rinomina gruppo")
    if st.session_state['group_names']:
        group_to_rename = st.sidebar.selectbox("Seleziona gruppo", options=st.session_state['group_names'], key="group_to_rename")
        renamed_group = st.sidebar.text_input("Nuovo nome", key="renamed_group")
        if st.sidebar.button("✏️ Rinomina gruppo"):
            old_name = group_to_rename
            new_name = renamed_group.strip()
            if not new_name:
                st.sidebar.warning("Inserisci un nuovo nome valido")
            elif new_name in st.session_state['group_names'] and new_name != old_name:
                st.sidebar.warning("Esiste già un gruppo con questo nome")
            else:
                st.session_state['group_names'] = [new_name if g == old_name else g for g in st.session_state['group_names']]
                st.session_state['strategy_mapping'] = {
                    s: (new_name if g == old_name else g) for s, g in st.session_state['strategy_mapping'].items()
                }
                st.sidebar.success(f"Gruppo '{old_name}' → '{new_name}'")
                st.rerun()

    # ── LIMITE MAX PER GRUPPO ────────────────────────────────────────────────
    st.sidebar.subheader("Limite per gruppo")
    max_per_group = st.sidebar.number_input(
        "Max strategie dello stesso gruppo in Top allocation",
        min_value=1, max_value=TOP_N,
        value=st.session_state['max_per_group'],
        step=1, help="Default 2"
    )
    st.session_state['max_per_group'] = int(max_per_group)

    # ── ASSEGNAZIONE STRATEGIE ───────────────────────────────────────────────
    st.sidebar.subheader("Assegnazione strategie")
    group_options = [''] + st.session_state['group_names']
    with st.sidebar.expander("📋 Modifica gruppo di ogni strategia", expanded=True):
        updated_mapping = {}
        for strat in all_strategies_in_file:
            current_group = st.session_state['strategy_mapping'].get(strat, '')
            default_idx = group_options.index(current_group) if current_group in group_options else 0
            selected_group = st.selectbox(label=strat, options=group_options, index=default_idx, key=f"group_{strat}")
            updated_mapping[strat] = selected_group
        if st.button("✅ Salva assegnazioni"):
            st.session_state['strategy_mapping'] = updated_mapping
            st.sidebar.success("Assegnazioni aggiornate — ricorda di esportare il JSON per salvarle!")
            st.rerun()

    strategy_mapping = st.session_state['strategy_mapping']
    still_unmapped = [s for s in all_strategies_in_file if not strategy_mapping.get(s, '').strip()]
    if still_unmapped:
        st.error(f"🛑 BLOCCO DI SICUREZZA: Strategie senza gruppo assegnato: {', '.join(still_unmapped)}")
        st.info("Apri la sidebar (▶), assegna un gruppo a ogni strategia e salva le assegnazioni.")
        st.stop()

    # ─── WFA ENGINE ──────────────────────────────────────────────────────────
    with st.spinner('Elaborazione Walk-Forward in corso...'):
        df_filtered = df_raw.copy()
        df_filtered['Date Opened'] = pd.to_datetime(df_filtered['Date Opened'])
        df_filtered['Date Closed'] = pd.to_datetime(df_filtered['Date Closed'])
        df_filtered['weekday_open'] = df_filtered['Date Opened'].dt.weekday
        df_filtered['P/L'] = df_filtered['P/L'].apply(clean_money)
        df_filtered['P/L %'] = df_filtered['P/L %'].apply(
            lambda x: float(str(x).replace('%', '').replace(',', '')) if pd.notna(x) else np.nan)
        df_filtered['Opening Commissions + Fees'] = df_filtered['Opening Commissions + Fees'].fillna(0).apply(clean_money)
        df_filtered['Closing Commissions + Fees'] = df_filtered['Closing Commissions + Fees'].fillna(0).apply(clean_money)
        df_filtered['Net PnL'] = (
            df_filtered['P/L'] - df_filtered['Opening Commissions + Fees'] - df_filtered['Closing Commissions + Fees']
        )
        df_filtered = df_filtered.sort_values('Date Closed').reset_index(drop=True)

        min_date = df_filtered['Date Closed'].min()
        max_date = df_filtered['Date Closed'].max()
        current_oos_start = min_date + pd.DateOffset(months=12)
        oos_results, historical_allocations, cusum_exclusions_log = [], [], []

        while current_oos_start <= max_date:
            current_oos_end = current_oos_start + pd.Timedelta(days=28)
            current_is_start = current_oos_start - pd.DateOffset(months=12)
            is_data = df_filtered[(df_filtered['Date Closed'] >= current_is_start) & (df_filtered['Date Closed'] < current_oos_start)].copy()
            oos_data = df_filtered[(df_filtered['Date Closed'] >= current_oos_start) & (df_filtered['Date Closed'] < current_oos_end)].copy()

            if is_data.empty:
                current_oos_start = current_oos_end
                continue

            valid_strats = is_data['Strategy'].value_counts()
            valid_strats = valid_strats[valid_strats >= 5].index.tolist()
            is_metrics = []

            for strat in valid_strats:
                strat_is = is_data[is_data['Strategy'] == strat].copy()
                if strat_is['P/L %'].notna().sum() > 0:
                    pos_sum = strat_is[strat_is['P/L %'] > 0]['P/L %'].sum()
                    neg_sum = strat_is[strat_is['P/L %'] < 0]['P/L %'].sum()
                else:
                    pos_sum = strat_is[strat_is['Net PnL'] > 0]['Net PnL'].sum()
                    neg_sum = strat_is[strat_is['Net PnL'] < 0]['Net PnL'].sum()
                omega = 50.0 if neg_sum == 0 else pos_sum / abs(neg_sum)
                banned_days = compute_cusum_banned(strat_is)
                active_days = strat_is['weekday_open'].unique().tolist()
                remaining = [d for d in active_days if d not in banned_days]

                if not remaining:
                    cusum_exclusions_log.append({
                        'OOS Start': current_oos_start,
                        'Strategy': strat,
                        'Motivo': f"Tutti i giorni operativi bannati: {format_banned_days(banned_days)}"
                    })
                    continue

                is_metrics.append({
                    'Strategy': strat,
                    'Group': strategy_mapping[strat],
                    'Omega': omega,
                    'Net PnL IS': strat_is['Net PnL'].sum(),
                    'Banned Days': banned_days
                })

            is_metrics_df = pd.DataFrame(is_metrics)
            if not is_metrics_df.empty:
                is_metrics_df = is_metrics_df.sort_values(['Omega', 'Net PnL IS'], ascending=[False, False])
                selected_strategies = []
                group_counts = {}

                for _, row in is_metrics_df.iterrows():
                    if len(selected_strategies) >= TOP_N:
                        break
                    grp = row['Group']
                    if group_counts.get(grp, 0) < st.session_state['max_per_group']:
                        selected_strategies.append(row)
                        group_counts[grp] = group_counts.get(grp, 0) + 1

                for rank, strat_row in enumerate(selected_strategies):
                    weight = 1.0 if rank < 5 else 0.5
                    strat_name, banned = strat_row['Strategy'], strat_row['Banned Days']
                    historical_allocations.append({
                        'OOS Start': current_oos_start,
                        'OOS End': current_oos_end,
                        'Rank': rank + 1,
                        'Strategy': strat_name,
                        'Group': strat_row['Group'],
                        'Omega IS': strat_row['Omega'],
                        'Weight': weight,
                        'Banned Days': banned,
                        'Max Per Group': st.session_state['max_per_group']
                    })
                    strat_oos = oos_data[oos_data['Strategy'] == strat_name].copy()
                    if not strat_oos.empty:
                        strat_oos = strat_oos[~strat_oos['weekday_open'].isin(banned)]
                        strat_oos['Weighted Net PnL'] = strat_oos['Net PnL'] * weight
                        oos_results.append(strat_oos)

            current_oos_start = current_oos_end

        if not oos_results:
            st.error("Nessun trade OOS generato. Verifica che il dataset copra almeno 13 mesi di storia.")
            st.stop()

        final_oos_df = pd.concat(oos_results).sort_values('Date Closed').reset_index(drop=True)
        hist_alloc_df = pd.DataFrame(historical_allocations)
        exclusions_df = pd.DataFrame(cusum_exclusions_log)

    st.success('Elaborazione completata!')

    # ─── 1. ALLOCAZIONE CORRENTE ─────────────────────────────────────────────
    st.header("1. Allocazione Corrente")
    latest_start = hist_alloc_df['OOS Start'].max()
    latest_end = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start]['OOS End'].iloc[0]
    today = pd.Timestamp.today().normalize()
    days_left = (latest_end - today).days

    val_col1, val_col2, val_col3, val_col4 = st.columns(4)
    val_col1.info(f"📅 **Allocazione attiva dal:** {latest_start.strftime('%d %b %Y')}")
    val_col2.info(f"🔚 **Valida fino al:** {latest_end.strftime('%d %b %Y')}")
    if days_left > 0:
        val_col3.warning(f"⏳ **Prossima rotazione tra:** {days_left} giorni ({latest_end.strftime('%d %b %Y')})")
    else:
        val_col3.error("🔄 **Rotazione scaduta** — aggiorna il CSV")
    val_col4.info(f"🧩 **Max per gruppo:** {st.session_state['max_per_group']}")

    latest_alloc = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start].copy()
    disp = latest_alloc[['Rank', 'Strategy', 'Group', 'Omega IS', 'Weight', 'Banned Days']].copy()
    disp['Weight'] = (disp['Weight'] * 100).astype(int).astype(str) + '%'
    disp['Omega IS'] = disp['Omega IS'].round(4)
    disp['Giorni Spenti (CUSUM)'] = disp['Banned Days'].apply(format_banned_days)
    disp.drop(columns=['Banned Days'], inplace=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    if not exclusions_df.empty:
        latest_excl = exclusions_df[exclusions_df['OOS Start'] == latest_start]
        if not latest_excl.empty:
            st.warning("⚠️ Strategie escluse perché CUSUM ha bannato tutti i loro giorni operativi:")
            st.dataframe(latest_excl[['Strategy', 'Motivo']], use_container_width=True, hide_index=True)

    # ─── 2. METRICHE OOS ─────────────────────────────────────────────────────
    st.header("2. Metriche Out-Of-Sample")
    tab_global, tab_recent = st.tabs(["📊 Storico Completo OOS", "📈 Dal 1° Settembre 2025"])
    with tab_global:
        render_metrics(calculate_metrics(final_oos_df), "Metriche — Storico Completo OOS")
    with tab_recent:
        recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
        render_metrics(calculate_metrics(recent_df), "Metriche — Dal 1° Settembre 2025")

    # ─── 3. EQUITY COMPLETA ──────────────────────────────────────────────────
    st.header("3. Curva Equity — Storico Completo OOS")
    all_daily = final_oos_df.groupby(final_oos_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
    all_cum = all_daily.cumsum()
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=all_cum.index, y=all_cum.values, mode='lines', fill='tozeroy', line=dict(color='#00b4d8')))
    fig_all.update_layout(title="Equity Cumulativa WFA — Storico Completo", xaxis_title="Data", yaxis_title="Net PnL ($)")
    st.plotly_chart(fig_all, use_container_width=True)

    # ─── 4. EQUITY RECENTE ───────────────────────────────────────────────────
    st.header("4. Curva Equity — Dal 1° Settembre 2025")
    recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
    if not recent_df.empty:
        rec_daily = recent_df.groupby(recent_df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
        rec_cum = rec_daily.cumsum()
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Scatter(x=rec_cum.index, y=rec_cum.values, mode='lines', fill='tozeroy', line=dict(color='#f77f00')))
        fig_rec.update_layout(title="Equity Cumulativa WFA — Dal 1° Settembre 2025", xaxis_title="Data", yaxis_title="Net PnL ($)")
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.info("Nessun dato OOS disponibile dal 1° settembre 2025.")

    # ─── 5. EXPORT ───────────────────────────────────────────────────────────
    st.header("5. Esporta Dati")
    hist_export = hist_alloc_df.copy()
    hist_export['Banned Days'] = hist_export['Banned Days'].apply(lambda x: format_banned_days(x) if isinstance(x, list) else x)
    equity_csv_df = pd.DataFrame({'Date': all_cum.index, 'Daily PnL': all_daily.values, 'Cumulative PnL': all_cum.values})
    equity_recent_csv_df = pd.DataFrame()
    if not recent_df.empty:
        equity_recent_csv_df = pd.DataFrame({'Date': rec_cum.index, 'Daily PnL': rec_daily.values, 'Cumulative PnL': rec_cum.values})

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.download_button("📥 Allocazioni Storiche", data=hist_export.to_csv(index=False).encode('utf-8'), file_name="wfa_allocations.csv", mime="text/csv")
    with col_b2:
        st.download_button("📥 Trade OOS Filtrati", data=final_oos_df.to_csv(index=False).encode('utf-8'), file_name="wfa_oos_trades.csv", mime="text/csv")
    with col_b3:
        st.download_button("📥 Equity Line Completa", data=equity_csv_df.to_csv(index=False).encode('utf-8'), file_name="equity_full.csv", mime="text/csv")
    with col_b4:
        if not equity_recent_csv_df.empty:
            st.download_button("📥 Equity Line Set 2025+", data=equity_recent_csv_df.to_csv(index=False).encode('utf-8'), file_name="equity_sep2025.csv", mime="text/csv")
