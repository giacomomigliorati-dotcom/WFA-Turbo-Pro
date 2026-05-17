import sys
sys.path.insert(0, '.')
from wfa_sizing import (
    StrategySizingConfig,
    PortfolioSizingParams,
    compute_contracts,
    _sl_per_contract,
    should_skip_bench,
    clean_money,
    DEFAULT_SL_USD,
    DEFAULT_SL_PCT,
)

# Test 1: Valori di default
print("TEST 1: Valori di default")
print(f"DEFAULT_SL_USD={DEFAULT_SL_USD}, DEFAULT_SL_PCT={DEFAULT_SL_PCT}")

# Test 2: Creazione di StrategySizingConfig
print("\nTEST 2: StrategySizingConfig con defaults")
cfg = StrategySizingConfig()
print(f"cap_pct={cfg.cap_pct}, sl_usd={cfg.sl_usd}, sl_pct={cfg.sl_pct}")

# Test 3: Calcolo con sl_pct > 0
print("\nTEST 3: Calcolo con sl_pct=150% (multiplo del premio)")
cfg2 = StrategySizingConfig(cap_pct=5.0, sl_pct=150.0)
premium = 100.0
sl_unit = _sl_per_contract(cfg2, premium)
print(f"Premium={premium}, sl_pct=150% → sl_unit={sl_unit} (atteso 150)")

# Test 4: Calcolo contratti con budget
print("\nTEST 4: compute_contracts")
contracts, risk = compute_contracts(
    premium=100.0,
    sizing_cfg=cfg2,
    capital=100_000.0,
    risk_factor=1.0,
    weight_wfa=0.80,
    remaining_daily_budget=10_000.0,
)
print(f"Contracts={contracts}, Risk Used={risk}")

# Test 5: should_skip_bench
print("\nTEST 5: should_skip_bench")
skip1 = should_skip_bench(weight_wfa=0.50, risk_factor=0.25)
skip2 = should_skip_bench(weight_wfa=0.80, risk_factor=0.50)
print(f"Bench (weight=0.50, rf=0.25)={skip1}, Full (weight=0.80, rf=0.50)={skip2}")

print("\n✅ TUTTI I TEST COMPLETATI")
