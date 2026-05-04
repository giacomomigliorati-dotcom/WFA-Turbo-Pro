import numpy as np
import pandas as pd

# Costanti condivise
GIORNI_SETTIMANA = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
OOS_STEP_DAYS    = 28

DEFAULT_RANKING_METRIC = "Omega Ratio"
RANKING_METRICS = ["Omega Ratio", "Sharpe Ratio", "Sortino Ratio", "Ulcer Index", "ROC"]


# ─── FUNZIONI METRICHE IS ──────────────────────────────────────────────────────────

def calc_omega(strat_is: pd.DataFrame) -> float:
    """Omega Ratio basato su P/L % se disponibile, altrimenti Net PnL assoluto."""
    if strat_is['P/L %'].notna().sum() > 0:
        pos = strat_is[strat_is['P/L %'] > 0]['P/L %'].sum()
        neg = strat_is[strat_is['P/L %'] < 0]['P/L %'].sum()
    else:
        pos = strat_is[strat_is['Net PnL'] > 0]['Net PnL'].sum()
        neg = strat_is[strat_is['Net PnL'] < 0]['Net PnL'].sum()
    return 50.0 if neg == 0 else pos / abs(neg)


def calc_sharpe(strat_is: pd.DataFrame) -> float:
    """Sharpe Ratio annualizzato su Net PnL giornaliero."""
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    if len(daily) < 2:
        return -999.0
    m, s = daily.mean(), daily.std()
    return (m / s) * np.sqrt(252) if s > 0 else -999.0


def calc_sortino(strat_is: pd.DataFrame) -> float:
    """Sortino Ratio annualizzato su Net PnL giornaliero."""
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    if len(daily) < 2:
        return -999.0
    down = daily[daily < 0].std()
    return (daily.mean() / down) * np.sqrt(252) if pd.notna(down) and down > 0 else -999.0


def calc_ulcer(strat_is: pd.DataFrame) -> float:
    """Ulcer Index (minore = migliore; ranking verrà invertito)."""
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    cum = daily.cumsum()
    peak = cum.cummax()
    dd_pct = ((cum - peak) / (peak.abs() + 1e-9)) * 100  # % drawdown
    ui = np.sqrt((dd_pct ** 2).mean())
    return ui if ui > 0 else 0.0


def calc_roc(strat_is: pd.DataFrame, n_steps: int) -> float:
    """ROC = variazione % del Net PnL cumulativo su finestra di n_steps * OOS_STEP_DAYS."""
    window_days = n_steps * OOS_STEP_DAYS
    cutoff = strat_is['Date Closed'].max() - pd.Timedelta(days=window_days)
    recent = strat_is[strat_is['Date Closed'] > cutoff]
    if recent.empty:
        return -999.0
    pnl_window = recent['Net PnL'].sum()
    return pnl_window  # in $ assoluti (già ordinabile)


def compute_ranking_score(strat_is: pd.DataFrame, metric: str, roc_steps: int):
    """Ritorna (score, higher_is_better) per la metrica IS selezionata."""
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
    # fallback
    return calc_omega(strat_is), True


def passes_roc_filter(strat_is: pd.DataFrame, filter_steps: int) -> bool:
    """True se ROC >= 0 (strategia non in perdita netta nella finestra filtro)."""
    return calc_roc(strat_is, filter_steps) >= 0


# ─── CUSUM WEEKDAY KILLER ─────────────────────────────────────────────────────

def compute_cusum_banned(is_data_strat: pd.DataFrame):
    """Restituisce la lista di weekday (0=Mon) bannati via CUSUM."""
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
    """Restituisce i weekday bannati in forma leggibile ("Lun, Mar, ...")."""
    if not banned_list:
        return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in banned_list if d < len(GIORNI_SETTIMANA)])
