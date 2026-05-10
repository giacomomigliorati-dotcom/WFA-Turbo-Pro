import numpy as np
import pandas as pd

# Costanti condivise
GIORNI_SETTIMANA = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
OOS_STEP_DAYS    = 28

DEFAULT_RANKING_METRIC    = "Omega Ratio"
RANKING_METRICS           = ["Omega Ratio", "Sharpe Ratio", "Sortino Ratio", "Ulcer Index", "ROC"]

# Defaults parametri WFA
DEFAULT_TOP_N             = 7
DEFAULT_FULL_WEIGHT_COUNT = 3
DEFAULT_FULL_WEIGHT_PCT   = 80
DEFAULT_BENCH_WEIGHT_PCT  = 40
DEFAULT_MAX_PER_GROUP     = 2
DEFAULT_ROC_STEPS         = 2
DEFAULT_ROC_FILTER_ENABLED = False
DEFAULT_ROC_FILTER_STEPS  = 2


# ─── FUNZIONI METRICHE IS ────────────────────────────────────────────────────

def calc_omega(strat_is: pd.DataFrame) -> float:
    """Omega Ratio: usa P/L % se disponibile, altrimenti Net PnL assoluto."""
    if 'P/L %' in strat_is.columns and strat_is['P/L %'].notna().sum() > 0:
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
    """Ulcer Index (minore = migliore; ranking verra invertito)."""
    daily = strat_is.groupby(strat_is['Date Closed'].dt.date)['Net PnL'].sum()
    cum = daily.cumsum()
    peak = cum.cummax()
    dd_pct = ((cum - peak) / (peak.abs() + 1e-9)) * 100
    ui = np.sqrt((dd_pct ** 2).mean())
    return ui if ui > 0 else 0.0


def calc_roc(strat_is: pd.DataFrame, n_steps: int) -> float:
    """ROC = Net PnL cumulativo su finestra di n_steps * OOS_STEP_DAYS."""
    window_days = n_steps * OOS_STEP_DAYS
    cutoff = strat_is['Date Closed'].max() - pd.Timedelta(days=window_days)
    recent = strat_is[strat_is['Date Closed'] > cutoff]
    if recent.empty:
        return -999.0
    return recent['Net PnL'].sum()


def compute_ranking_score(strat_is: pd.DataFrame, metric: str, roc_steps: int):
    """Ritorna (score, higher_is_better) per la metrica IS selezionata."""
    if metric == "Omega Ratio":
        return calc_omega(strat_is), True
    elif metric == "Sharpe Ratio":
        return calc_sharpe(strat_is), True
    elif metric == "Sortino Ratio":
        return calc_sortino(strat_is), True
    elif metric == "Ulcer Index":
        return calc_ulcer(strat_is), False
    elif metric == "ROC":
        return calc_roc(strat_is, roc_steps), True
    return calc_omega(strat_is), True


def passes_roc_filter(strat_is: pd.DataFrame, filter_steps: int) -> bool:
    """True se ROC >= 0 (strategia non in perdita netta nella finestra filtro)."""
    return calc_roc(strat_is, filter_steps) >= 0


# ─── CUSUM WEEKDAY KILLER ────────────────────────────────────────────────────

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
    """Restituisce i weekday bannati in forma leggibile."""
    if not banned_list:
        return "Nessuno"
    return ", ".join([GIORNI_SETTIMANA[d] for d in banned_list if d < len(GIORNI_SETTIMANA)])


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY CONTROL
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_EC_ENABLED      = False
DEFAULT_EC_CAPITAL_MODE = "Trailing"   # "Fisso" | "Trailing"
DEFAULT_EC_P80          = 80
DEFAULT_EC_P90          = 90
DEFAULT_EC_P95          = 95
EC_MIN_DAYS_IS          = 60          # soglia avviso affidabilita IS


def compute_dd_percentiles_from_is(
    strat_is: pd.DataFrame,
    p80: int = DEFAULT_EC_P80,
    p90: int = DEFAULT_EC_P90,
    p95: int = DEFAULT_EC_P95,
) -> dict:
    """Calcola i percentili del drawdown giornaliero dalla finestra IS.

    I percentili vengono calcolati sui valori ASSOLUTI del drawdown
    (sempre >= 0), in modo che soglie piu alte corrispondano a DD piu gravi.

    Ritorna un dict con chiavi:
        dd_p80, dd_p90, dd_p95  : valori di drawdown assoluto (float >= 0)
        daily_dd_series         : pd.Series drawdown giornaliero (valori <= 0)
        n_days                  : numero di giorni di trading IS
        reliable                : bool -- True se n_days >= EC_MIN_DAYS_IS
        warning                 : str | None -- messaggio se non affidabile
    """
    if strat_is is None or strat_is.empty:
        return {
            "dd_p80": 0.0, "dd_p90": 0.0, "dd_p95": 0.0,
            "daily_dd_series": pd.Series(dtype=float),
            "n_days": 0, "reliable": False,
            "warning": "Nessun dato IS disponibile.",
        }

    df = strat_is.copy()
    df["_date"] = pd.to_datetime(df["Date Closed"]).dt.date
    daily_pnl = df.groupby("_date")["Net PnL"].sum().sort_index()
    equity    = daily_pnl.cumsum()
    peak      = equity.cummax()
    dd_series = equity - peak          # valori <= 0

    dd_real   = dd_series[dd_series < 0]
    n_days    = len(daily_pnl)
    reliable  = n_days >= EC_MIN_DAYS_IS
    warning   = (
        f"Solo {n_days} giorni di trading IS (minimo consigliato: {EC_MIN_DAYS_IS})."
        if not reliable else None
    )

    if dd_real.empty:
        return {
            "dd_p80": 0.0, "dd_p90": 0.0, "dd_p95": 0.0,
            "daily_dd_series": dd_series,
            "n_days": n_days, "reliable": reliable, "warning": warning,
        }

    # BUG-1 FIX: calcola percentili sui valori assoluti del DD.
    # np.percentile su valori negativi restituisce il DD meno grave
    # (es. -10 invece di -200). Convertendo in assoluto, P80 < P90 < P95
    # e correttamente ordinato: piu alto = DD piu grave.
    abs_dd = np.abs(dd_real.values)
    return {
        "dd_p80": float(np.percentile(abs_dd, p80)),
        "dd_p90": float(np.percentile(abs_dd, p90)),
        "dd_p95": float(np.percentile(abs_dd, p95)),
        "daily_dd_series": dd_series,
        "n_days": n_days, "reliable": reliable, "warning": warning,
    }


def get_risk_factor(current_dd_abs: float, thresholds: dict) -> float:
    """Mappa il drawdown corrente (valore assoluto >= 0) a un RiskFactor.

    Le soglie in `thresholds` arrivano gia come valori positivi
    (output di compute_dd_percentiles_from_is dopo BUG-1 fix).

    Schema a 4 livelli:
        0.0    <= dd < P80  ->  1.00  (piena operativita)
        P80    <= dd < P90  ->  0.50  (half size)
        P90    <= dd < P95  ->  0.25  (quarter size)
        P95    <= dd        ->  0.00  (stop totale)
    """
    t80 = thresholds.get("dd_p80", 0.0)
    t90 = thresholds.get("dd_p90", 0.0)
    t95 = thresholds.get("dd_p95", 0.0)

    if t95 == 0.0:
        return 1.0   # nessun DD in IS => mai fermare

    if current_dd_abs < t80:
        return 1.00
    elif current_dd_abs < t90:
        return 0.50
    elif current_dd_abs < t95:
        return 0.25
    else:
        return 0.00


def apply_equity_control(
    strat_oos: pd.DataFrame,
    thresholds: dict,
    capital_mode: str = "Trailing",
    start_equity: float = 0.0,
) -> pd.DataFrame:
    """Applica l'equity control a una finestra OOS di una strategia.

    Principio: equity pura vs equity pesata
    ----------------------------------------
    Il drawdown usato per calcolare il RiskFactor e misurato sull'EQUITY PURA,
    cioe la somma dei Net PnL reali senza moltiplicatori di peso ne di RiskFactor.
    Questo e fondamentale: una strategia in "panchina" (weight=40%) non deve
    sembrare piu sana di una "titolare" (weight=80%) solo perche il suo PnL
    pesato e numericamente piu piccolo.

    Separazione netta dei due piani:
      - Tracking DD  : usa sempre Net PnL puro (colonna "Net PnL")
      - Output portafoglio: EC Weighted Net PnL = Weighted Net PnL * RiskFactor

    Logica di restart graduale
    --------------------------
    Quando RF == 0.0 (stop totale, DD >= P95), la strategia non contribuisce
    al portafoglio (EC Weighted Net PnL = 0), ma il tracking dell'equity pura
    continua ad aggiornarsi con il Net PnL reale. Questo permette al drawdown
    di ridursi naturalmente durante lo stop, rendendo possibile il rientro
    automatico ai livelli:
        0.00  ->  0.25  (DD scende sotto P95)
        0.25  ->  0.50  (DD scende sotto P90)
        0.50  ->  1.00  (DD scende sotto P80)

    In modalita Fisso il restart e garantito all'inizio di ogni finestra OOS
    perche equity e peak ripartono da zero.

    Colonne aggiunte
    ----------------
        RiskFactor            : float 0 / 0.25 / 0.5 / 1.0 per ogni trade
        EC Weighted Net PnL   : Weighted Net PnL * RiskFactor
        EC Tracking Equity    : equity pura interna usata per calcolare il DD

    Parametri
    ---------
    strat_oos       : DataFrame OOS gia pesato (colonna 'Weighted Net PnL')
    thresholds      : output di compute_dd_percentiles_from_is
    capital_mode    : "Trailing" (equity pura cumulativa tra finestre) | "Fisso"
    start_equity    : equity pura di partenza (usata solo in Trailing)
    """
    df = strat_oos.copy().sort_values("Date Closed").reset_index(drop=True)

    if "Weighted Net PnL" not in df.columns:
        df["Weighted Net PnL"] = df.get("Net PnL", 0.0)

    # In modalita Fisso si azzera a ogni finestra OOS -> restart automatico
    equity = 0.0 if capital_mode == "Fisso" else start_equity
    peak   = equity

    risk_factors      = []
    tracking_equities = []

    for pnl_raw, pnl_weighted in zip(df["Net PnL"], df["Weighted Net PnL"]):
        # 1. Calcola DD corrente sull'EQUITY PURA e ottieni RiskFactor
        dd_abs = max(0.0, peak - equity)
        rf     = get_risk_factor(dd_abs, thresholds)
        risk_factors.append(rf)
        tracking_equities.append(equity)

        # 2. Aggiorna equity di tracking usando SEMPRE il Net PnL puro.
        #    Il peso (0.8 / 0.4) e il RiskFactor NON entrano nel tracking:
        #    cio garantisce che il DD misurato rifletta la salute reale
        #    della strategia, indipendentemente dal suo peso nel portafoglio.
        equity += pnl_raw

        if equity > peak:
            peak = equity

    df["RiskFactor"]          = risk_factors
    df["EC Weighted Net PnL"] = df["Weighted Net PnL"] * df["RiskFactor"]
    df["EC Tracking Equity"]  = tracking_equities
    return df
