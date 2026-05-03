import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="WFA Portfolio Allocator", layout="wide")
st.title("Rotational Portfolio WFA (28 Giorni) & CUSUM Anomaly Detection")
st.markdown('''
Questa applicazione esegue l'aggiornamento mensile del portafoglio rotazionale.
Carica il tuo file CSV esportato per ottenere la nuova allocazione e le metriche di backtest OOS.
''')

strategy_mapping = {
    'Condor ZM+': 'A', 'TEST - Iron Condor delle 20:00': 'A', 'Monday Condor': 'A', 'IC Friday - Croccante (optimized)': 'A',
    'TEST - DC 2-5': 'B', "Giusti's DCS": 'B', "Giusti's 1DSC": 'B', 'TEST - RIC by TradingMonk': 'B', "1DSC - Jack's": 'B', 'TEST - RIC from Hell': 'B',
    'RS Rain orario uscita delta': 'C', 'TEST - Sell put 16:00': 'C', 'TEST - Sell put 20:00': 'C',
    'Bull call VIX - Morning': 'D', 'TEST - Call butterfly': 'D', 'TEST-Boost Up for RIC': 'D',
    'PDown w/VIX': 'E', 'TEST - PDown': 'E'
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
        ann_ret = total_pnl * (365.25 / days)
        calmar = ann_ret / max_dd
    return total_pnl, max_dd, sharpe, calmar

# FIX: CUSUM calcolato SOLO sui dati IS passati come argomento (mai sull'intero dataset)
def compute_cusum_banned(is_data_strat):
    """
    Riceve i soli trade IS (finestra rolling 12 mesi) di una singola strategia.
    Restituisce la lista dei weekday (0=Lun..4=Ven) da blacklistare per il successivo OOS.
    """
    banned = []
    for day in range(7):
        day_data = is_data_strat[is_data_strat['weekday_open'] == day].sort_values('Date Closed')
        if len(day_data) >= 10:
            std = day_data['Net PnL'].std()
            if std > 0:
                k = 0.5 * std
                h = 3.0 * std
                c_low = 0
                for pnl in day_data['Net PnL']:
                    c_low = max(0, c_low - pnl - k)
                if c_low > h:
                    banned.append(day)
    return banned

def format_banned_days(banned_list):
    if not banned_list:
        return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in banned_list if d < len(GIORNI_SETTIMANA)])

uploaded_file = st.file_uploader("Carica il dataset dei trade (CSV)", type=['csv'])

if uploaded_file is not None:
    with st.spinner('Elaborazione in corso...'):

        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        # 1. PREPROCESSING
        df_filtered = df[~df['Strategy'].astype(str).str.contains('Legendary', na=False, case=False)].copy()

        # BLOCCO DI SICUREZZA: strategie non mappate
        unique_strategies = df_filtered['Strategy'].unique().tolist()
        unmapped = [s for s in unique_strategies if s not in strategy_mapping]
        if unmapped:
            st.error(f"\U0001f6d1 BLOCCO DI SICUREZZA: Trovate strategie NON mappate: {', '.join(unmapped)}")
            st.info("Aggiungi queste strategie al dizionario `strategy_mapping` specificando la famiglia (A, B, C, D o E).")
            st.stop()

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
        df_filtered = df_filtered.sort_values(by='Date Closed').reset_index(drop=True)

        min_date = df_filtered['Date Closed'].min()
        max_date = df_filtered['Date Closed'].max()

        # 2. MOTORE WALK-FORWARD (IS 12 mesi, OOS 28 giorni)
        current_oos_start = min_date + pd.DateOffset(months=12)
        oos_results = []
        historical_allocations = []

        while current_oos_start <= max_date:
            current_oos_end = current_oos_start + pd.Timedelta(days=28)
            current_is_start = current_oos_start - pd.DateOffset(months=12)

            # IS: SOLO i trade nella finestra rolling 12 mesi
            is_data = df_filtered[
                (df_filtered['Date Closed'] >= current_is_start) &
                (df_filtered['Date Closed'] < current_oos_start)
            ].copy()

            # OOS: 28 giorni successivi
            oos_data = df_filtered[
                (df_filtered['Date Closed'] >= current_oos_start) &
                (df_filtered['Date Closed'] < current_oos_end)
            ].copy()

            if is_data.empty:
                current_oos_start = current_oos_end
                continue

            strats_in_is = is_data['Strategy'].value_counts()
            valid_strats = strats_in_is[strats_in_is >= 5].index.tolist()

            is_metrics = []
            for strat in valid_strats:
                # FIX: strat_is è sempre il subset IS della finestra corrente
                strat_is = is_data[is_data['Strategy'] == strat].copy()

                # Omega Ratio su IS
                if strat_is['P/L %'].notna().sum() > 0:
                    pos_sum = strat_is[strat_is['P/L %'] > 0]['P/L %'].sum()
                    neg_sum = strat_is[strat_is['P/L %'] < 0]['P/L %'].sum()
                else:
                    pos_sum = strat_is[strat_is['Net PnL'] > 0]['Net PnL'].sum()
                    neg_sum = strat_is[strat_is['Net PnL'] < 0]['Net PnL'].sum()

                omega = 50.0 if neg_sum == 0 else pos_sum / abs(neg_sum)
                total_net_pnl = strat_is['Net PnL'].sum()

                # FIX: CUSUM riceve SOLO strat_is (finestra IS), mai df_filtered intero
                banned_days = compute_cusum_banned(strat_is)

                is_metrics.append({
                    'Strategy': strat,
                    'Family': strategy_mapping[strat],
                    'Omega': omega,
                    'Net PnL IS': total_net_pnl,
                    'Banned Days': banned_days
                })

            is_metrics_df = pd.DataFrame(is_metrics)
            if not is_metrics_df.empty:
                is_metrics_df = is_metrics_df.sort_values(
                    by=['Omega', 'Net PnL IS'], ascending=[False, False]
                )
                selected_strategies = []
                family_counts = {fam: 0 for fam in set(strategy_mapping.values())}

                for _, row in is_metrics_df.iterrows():
                    if len(selected_strategies) >= 7:
                        break
                    fam = row['Family']
                    if family_counts.get(fam, 0) < 2:
                        selected_strategies.append(row)
                        family_counts[fam] = family_counts.get(fam, 0) + 1

                for rank, strat_row in enumerate(selected_strategies):
                    weight = 1.0 if rank < 5 else 0.5
                    strat_name = strat_row['Strategy']
                    banned = strat_row['Banned Days']

                    historical_allocations.append({
                        'OOS Start': current_oos_start,
                        'Rank': rank + 1,
                        'Strategy': strat_name,
                        'Family': strat_row['Family'],
                        'Omega IS': strat_row['Omega'],
                        'Weight': weight,
                        'Banned Days': banned   # lista nativa, non stringa
                    })

                    strat_oos = oos_data[oos_data['Strategy'] == strat_name].copy()
                    if not strat_oos.empty:
                        # Applica blacklist CUSUM sull'OOS
                        strat_oos = strat_oos[~strat_oos['weekday_open'].isin(banned)]
                        strat_oos['Weighted Net PnL'] = strat_oos['Net PnL'] * weight
                        oos_results.append(strat_oos)

            current_oos_start = current_oos_end

        if not oos_results:
            st.error("Nessun trade OOS generato. Verifica che il dataset copra almeno 13 mesi di storia.")
            st.stop()

        final_oos_df = pd.concat(oos_results).sort_values(by='Date Closed').reset_index(drop=True)
        hist_alloc_df = pd.DataFrame(historical_allocations)

    st.success('Elaborazione completata con successo!')

    # --- 1. ALLOCAZIONE CORRENTE ---
    st.header("1. Allocazione Corrente (Prossima Rotazione)")

    latest_start = hist_alloc_df['OOS Start'].max()
    latest_alloc = hist_alloc_df[hist_alloc_df['OOS Start'] == latest_start].copy()

    display_alloc = latest_alloc[['Rank', 'Strategy', 'Family', 'Omega IS', 'Weight', 'Banned Days']].copy()
    display_alloc['Weight'] = (display_alloc['Weight'] * 100).astype(int).astype(str) + '%'
    display_alloc['Omega IS'] = display_alloc['Omega IS'].round(4)
    # FIX: Banned Days è già una lista nativa, non serve eval()
    display_alloc['Giorni Spenti (CUSUM)'] = display_alloc['Banned Days'].apply(format_banned_days)
    display_alloc.drop(columns=['Banned Days'], inplace=True)

    st.dataframe(display_alloc, use_container_width=True, hide_index=True)

    # --- 2. METRICHE OOS ---
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

    # --- 3. EQUITY CURVE ---
    st.header("3. Curva Equity (Dal 1 Set 2025)")
    if not recent_df.empty:
        recent_daily = recent_df.groupby(recent_df['Date Closed'].dt.date)['Weighted Net PnL'].sum().cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_daily.index, y=recent_daily.values,
            mode='lines', fill='tozeroy', name='Cumulative Equity',
            line=dict(color='#00b4d8')
        ))
        fig.update_layout(
            title="Equity Cumulativa WFA 28-Days Ponderata",
            xaxis_title="Data", yaxis_title="Net PnL ($)",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nessun dato OOS disponibile dal 1° settembre 2025.")

    # --- 4. EXPORT ---
    st.header("4. Esporta Dati")

    # Serializza Banned Days come stringa per il CSV
    hist_alloc_export = hist_alloc_df.copy()
    hist_alloc_export['Banned Days'] = hist_alloc_export['Banned Days'].apply(
        lambda x: format_banned_days(x) if isinstance(x, list) else x
    )
    csv_alloc = hist_alloc_export.to_csv(index=False).encode('utf-8')
    csv_oos = final_oos_df.to_csv(index=False).encode('utf-8')

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button("Scarica Allocazioni Storiche (CSV)", data=csv_alloc, file_name="wfa_allocations.csv", mime="text/csv")
    with col_btn2:
        st.download_button("Scarica Trade OOS (CSV)", data=csv_oos, file_name="wfa_oos_trades.csv", mime="text/csv")
