# WFA Turbo Pro

Rotational Algorithmic Portfolio with Walk-Forward Engine (28-day rolling), CUSUM Anomaly Detection & Streamlit Dashboard.

## Features
- Walk-Forward Analysis with 12-month IS window and 28-day OOS step
- CUSUM Weekday Anomaly Detection (Blacklist per strategy)
- Anti-Overlap filter (max 2 strategies per family)
- Fractional rotation weights (100% Core, 50% Bench)
- Safety block for unmapped strategies
- Interactive Streamlit dashboard with CSV upload

## Families
| Label | Strategies |
|---|---|
| A | Iron Condor variants |
| B | RIC / DCS / DSC / DC variants |
| C | Short Put / Rain variants |
| D | Long Call / Butterfly / Boost Up |
| E | Bearish (PDown) |

## Deploy on Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set `streamlit_wfa_app.py` as the main file
5. Click **Deploy**

## Local Run
```bash
pip install -r requirements.txt
streamlit run streamlit_wfa_app.py
```
