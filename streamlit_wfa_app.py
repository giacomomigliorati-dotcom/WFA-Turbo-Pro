import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
based su un motore **Walk-Forward Analysis (WFA)** con finestra rolling di 12 mesi.

---

### ⚙️ Workflow passo per passo

**1. Preprocessing**  
Il CSV caricato viene pulito automaticamente: si rimuovono le strategie *Legendary*, si calcolano 
i **Net PnL** al netto delle commissioni, si estrae il giorno della settimana da ogni apertura.

**2. Blocco di Sicurezza**  
Se il dataset contiene strategie non assegnate a nessuna delle 5 famiglie strategiche ammesse 
(A–E), il programma **si ferma** e chiede all'utente di assegnare la famiglia mancante tramite 
i menù a tendina nella sidebar prima di procedere.

**3. Omega Ratio (ranking IS)**  
Per ogni finestra In-Sample di 12 mesi, ogni strategia riceve un punteggio basato sull'**Omega Ratio**: 
rapporto tra somma dei rendimenti positivi e valore assoluto dei negativi. L'Omega misura la qualità 
dell'intera distribuzione dei rendimenti, non solo media e varianza. Un Omega > 1 indica un sistema profittevole.

**4. Anomaly Detection CUSUM (Weekday Killer)**  
Per ogni combinazione strategia/giorno con almeno 10 trade nella finestra IS, viene calcolata 
una **somma cumulativa inferiore (CUSUM)**. Se la perdita cumulativa su un determinato giorno 
supera la soglia di allarme `h = 3 × std`, quel giorno viene inserito nella **Blacklist** per 
la finestra OOS successiva.  
⚠️ Regola aggiuntiva: se il CUSUM banna **tutti** i giorni su cui la strategia opera realmente, 
la strategia viene **esclusa completamente** dalla selezione (es. Monday Condor bannata di lunedì).

**5. Filtro Anti-Overlap e Rotazione Frazionata**  
Le strategie qualificate vengono ordinate per Omega decrescente. Si seleziona un portafoglio 
**Core di 7 strategie (Top-7)** con un limite massimo di **2 strategie per famiglia**.  
- Rank 1–5 → Peso **100%** (Core Pieno)  
- Rank 6–7 → Peso **50%** (Panchina)

**6. OOS Filtrato**  
Nella finestra OOS di 28 giorni, i trade delle 7 strategie selezionate vengono filtrati: 
i trade che si aprono in un giorno blacklistato vengono rimossi. Il Net PnL rimanente 
viene moltiplicato per il peso assegnato (100% o 50%).

---

### 📊 Output prodotti
- **Allocazione Corrente**: tabella con rank, famiglia, Omega, peso e giorni spenti per la prossima rotazione
- **Metriche OOS**: Weighted Net PnL, Max Drawdown, Sharpe Ratio, Calmar Ratio (globali e dal 1° Set 2025)
- **Curve di Equity**: curva ponderata su tutto lo storico OOS e dal 1° settembre 2025
- **Export CSV**: allocazioni storiche e trade OOS filtrati scaricabili direttamente dall'app

---

### 🗂️ Famiglie Strategiche Ammesse
| Label | Tipo |
|---|---|
| **A** | Iron Condor variants |
| **B** | RIC / DCS / DSC / DC variants |
| **C** | Short Put / Rain variants |
| **D** | Long Call / Butterfly / Boost Up |
| **E** | Bearish (PDown) |
        """)

# ─── MAPPATURA FAMIGLIE (base) ────────────────────────────────────────────────
DEFAULT_STRATEGY_MAPPING = {
    'Condor ZM+': 'A', 'TEST - Iron Condor delle 20:00': 'A', 'Monday Condor': 'A', 'IC Friday - Croccante (optimized)': 'A',
    'TEST - DC 2-5': 'B', "Giusti's DCS": 'B', "Giusti's 1DSC": 'B', 'TEST - RIC by TradingMonk': 'B', "1DSC - Jack's": 'B', 'TEST - RIC from Hell': 'B',
    'RS Rain orario uscita delta': 'C', 'TEST - Sell put 16:00': 'C', 'TEST - Sell put 20:00': 'C',
    'Bull call VIX - Morning': 'D', 'TEST - Call butterfly': 'D', 'TEST-Boost Up for RIC': 'D',
    'PDown w/VIX': 'E', 'TEST - PDown': 'E'
}

FAMIGLIE_LABELS = {
    'A': 'A — Iron Condor',
    'B': 'B — RIC / DCS / DC',
    'C': 'C — Short Put / Rain',
    'D': 'D — Long Call / Butterfly',
    'E': 'E — Bearish (PDown)'
}

GIORNI_SETTIMANA = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']

def clean_money(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    return float(str(val).replace('$', '').replace(',', ''))

def calculate_metrics(df):
    if df.empty: return 0, 0, 0, 0
    total_pnl = df['Weighted Net PnL'].sum()
    daily_pnl = df.groupby(df['Date Closed'].dt.date)['Weighted Net PnL'].sum()
    daily_cum_pnl = daily_pnl.cumsum()
    max_dd = (daily_cum_pnl.cummax() - daily_cum_pnl).max()
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    calmar = 0
    days = (daily_pnl.index.max() - daily_pnl.index.min()).days
    if max_dd > 0 and days > 0:
        calmar = (total_pnl * (365.25 / days)) / max_dd
    return total_pnl, max_dd, sharpe, calmar

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

# ─── UPLOAD FILE ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw[~df_raw['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()

    all_strategies_in_file = sorted(df_raw['Strategy'].dropna().unique().tolist())

    # ─── SIDEBAR: ASSEGNAZIONE / MODIFICA FAMIGLIE ───────────────────────────
    st.sidebar.header("🗂️ Assegnazione Famiglie")
    st.sidebar.markdown("Modifica o assegna la famiglia per ogni strategia presente nel dataset.")

    if 'strategy_mapping' not in st.session_state:
        st.session_state['strategy_mapping'] = DEFAULT_STRATEGY_MAPPING.copy()

    unmapped_detected = [s for s in all_strategies_in_file if s not in st.session_state['strategy_mapping']]
    if unmapped_detected:
        st.sidebar.warning(f"⚠️ Strategie senza famiglia: {', '.join(unmapped_detected)}")

    famiglia_options = list(FAMIGLIE_LABELS.keys())
    famiglia_display = list(FAMIGLIE_LABELS.values())

    with st.sidebar.expander("📋 Modifica mappatura strategie", expanded=bool(unmapped_detected)):
        updated_mapping = {}
        for strat in all_strategies_in_file:
            current_fam = st.session_state['strategy_mapping'].get(strat, None)
            default_idx = famiglia_options.index(current_fam) if current_fam in famiglia_options else 0
            selected_display = st.selectbox(
                label=strat,
                options=famiglia_display,
                index=default_idx,
                key=f"fam_{strat}"
            )
            selected_fam = famiglia_options[famiglia_display.index(selected_display)]
            updated_mapping[strat] = selected_fam
        if st.button("✅ Salva mappatura"):
            st.session_state['strategy_mapping'] = updated_mapping
            st.sidebar.success("Mappatura aggiornata!")
            st.rerun()

    strategy_mapping = st.session_state['strategy_mapping']

    # Blocco di sicurezza finale
    still_unmapped = [s for s in all_strategies_in_file if s not in strategy_mapping]
    if still_unmapped:
        st.error(f"🛑 BLOCCO DI SICUREZZA: Strategie senza famiglia assegnata: {', '.join(still_unmapped)}")
        st.info("Apri la sidebar (▶) e assegna una famiglia a ogni strategia, poi clicca 'Salva mappatura'.")
        st.stop()

    # ─── ELABORAZIONE WFA ────────────────────────────────────────────────────
    with st.spinner('Elaborazione Walk-Forward in corso...'):

        df_filtered = df_raw.copy()
        df_filtered['Date Opened'] = pd.to_datetime(df_filtered['Date Opened'])
        df_filtered['Date Closed'] = pd.to_datetime(df_filtered['Date Closed'])
        df_filtered['weekday_open'] = df_filtered['Date Opened'].dt.weekday
        df_filtered['P/L'] = df_filtered['P/L'].apply(clean_money)
        df_filtered['P/L %'] = df_filtered['P/L %'].apply(
            lambda x: float(str(x).replace('%', '').replace(',', '')) if pd.notna(x) else np.nan
        )
        df_filtered['Opening Commissions + Fees'] = df_filtered['Opening Commissions + Fees'].fillna(0).apply(clean_money)
        df_filtered['Closing Commissions + Fees'] = df_filtered['Closing Commissions + Fees'].fillna(0).apply(clean_money)
        df_filtered['Net PnL'] = (
            df_filtered['P/L']
            - df_filtered['Opening Commissions + Fees']
            - df_filtered['Closing Commissions + Fees']
        )
        df_filtered = df_filtered.sort_values('Date Closed').reset_index(drop=True)

        min_date = df_filtered['Date Closed'].min()
        max_date = df_filtered['Date Closed'].max()

        current_oos_start = min_date + pd.DateOffset(months=12)
        oos_results, historical_allocations, cusum_exclusions_log = [], [], []

        while current_oos_start <= max_date:
            current_oos_end = current_oos_start + pd.Timedelta(days=28)
            current_is_start = current_oos_start - pd.DateOffset(months=12)

            is_data = df_filtered[
                (df_filtered['Date Closed'] >= current_is_start) &
                (df_filtered['Date Closed'] < current_oos_start)
            ].copy()
            oos_data = df_filtered[
                (df_filtered['Date Closed'] >= current_oos_start) &
                (df_filtered['Date Closed'] < current_oos_end)
            ].copy()

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
                        'OOS Start': current_oos_start, 'Strategy': strat,
                        'Motivo': f"Tutti i giorni operativi bannati: {format_banned_days(banned_days)}"
                    })
                    continue

                is_metrics.append({
                    'Strategy': strat, 'Family': strategy_mapping[strat],
                    'Omega': omega, 'Net PnL IS': strat_is['Net PnL'].sum(),
                    'Banned Days': banned_days
                })

            is_metrics_df = pd.DataFrame(is_metrics)
            if not is_metrics_df.empty:
                is_metrics_df = is_metrics_df.sort_values(['Omega', 'Net PnL IS'], ascending=[False, False])
                selected_strategies, family_counts = [], {f: 0 for f in set(strategy_mapping.values())}

                for _, row in is_metrics_df.iterrows():
                    if len(selected_strategies) >= 7: break
                    if family_counts.get(row['Family'], 0) < 2:
                        selected_strategies.append(row)
                        family_counts[row['Family']] = family_counts.get(row['Family'], 0) + 1

                for rank, strat_row in enumerate(selected_strategies):
                    weight = 1.0 if rank < 5 else 0.5
                    strat_name, banned = strat_row['Strategy'], strat_row['Banned Days']
                    historical_allocations.append({
                        'OOS Start': current_oos_start, 'Rank': rank + 1,
                        'Strategy': strat_name, 'Family': strat_row['Family'],
                        'Omega IS': strat_row['Omega'], 'Weight': weight, 'Banned Days': banned
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
    st.header("1. Allocazione Corrente (Prossima Rotazione)")
    latest_start = hist_alloc_df['OOS Start'].max()
    latest_alloc = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start].copy()

    disp = latest_alloc[['Rank', 'Strategy', 'Family', 'Omega IS', 'Weight', 'Banned Days']].copy()
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
    col1, col2 = st.columns(2)
    g_pnl, g_dd, g_sharpe, g_calmar = calculate_metrics(final_oos_df)
    recent_df = final_oos_df[final_oos_df['Date Closed'] >= pd.to_datetime('2025-09-01')].copy()
    r_pnl, r_dd, r_sharpe, r_calmar = calculate_metrics(recent_df)

    with col1:
        st.subheader("Globale (Tutto lo storico OOS)")
        st.metric("Weighted Net PnL", f"${g_pnl:,.2f}")
        st.metric("Max Drawdown", f"${g_dd:,.2f}")
        st.metric("Sharpe Ratio", f"{g_sharpe:.2f}")
        st.metric("Calmar Ratio", f"{g_calmar:.2f}")
    with col2:
        st.subheader("Recente (Dal 1 Set 2025)")
        st.metric("Weighted Net PnL", f"${r_pnl:,.2f}")
        st.metric("Max Drawdown", f"${r_dd:,.2f}")
        st.metric("Sharpe Ratio", f"{r_sharpe:.2f}")
        st.metric("Calmar Ratio", f"{r_calmar:.2f}")

    # ─── 3. EQUITY — GLOBALE ─────────────────────────────────────────────────
    st.header("3. Curva Equity — Storico Completo OOS")
    all_daily = final_oos_df.groupby(final_oos_df['Date Closed'].dt.date)['Weighted Net PnL'].sum().cumsum()
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=all_daily.index, y=all_daily.values,
        mode='lines', fill='tozeroy', name='Equity Globale',
        line=dict(color='#00b4d8')
    ))
    fig_all.update_layout(
        title="Equity Cumulativa WFA — Storico Completo",
        xaxis_title="Data", yaxis_title="Net PnL ($)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # ─── 4. EQUITY — DAL 1 SET 2025 ──────────────────────────────────────────
    st.header("4. Curva Equity — Dal 1° Settembre 2025")
    if not recent_df.empty:
        rec_daily = recent_df.groupby(recent_df['Date Closed'].dt.date)['Weighted Net PnL'].sum().cumsum()
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Scatter(
            x=rec_daily.index, y=rec_daily.values,
            mode='lines', fill='tozeroy', name='Equity Set 2025',
            line=dict(color='#f77f00')
        ))
        fig_rec.update_layout(
            title="Equity Cumulativa WFA — Dal 1° Settembre 2025",
            xaxis_title="Data", yaxis_title="Net PnL ($)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.info("Nessun dato OOS disponibile dal 1° settembre 2025.")

    # ─── 5. EXPORT ───────────────────────────────────────────────────────────
    st.header("5. Esporta Dati")
    hist_export = hist_alloc_df.copy()
    hist_export['Banned Days'] = hist_export['Banned Days'].apply(
        lambda x: format_banned_days(x) if isinstance(x, list) else x
    )
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.download_button("📥 Allocazioni Storiche (CSV)", data=hist_export.to_csv(index=False).encode('utf-8'),
                           file_name="wfa_allocations.csv", mime="text/csv")
    with col_b2:
        st.download_button("📥 Trade OOS Filtrati (CSV)", data=final_oos_df.to_csv(index=False).encode('utf-8'),
                           file_name="wfa_oos_trades.csv", mime="text/csv")
