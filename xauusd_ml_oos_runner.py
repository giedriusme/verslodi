import xauusd_backtest as bt

# Train only on history before 2023; 2023-2026 remains untouched OOS.
bt.YEARS_MONTHS = [(y, m) for y in range(2018, 2027) for m in range(1, 13) if (y < 2026 or m <= 8)]

import xauusd_ml_oos_search as search

if __name__ == '__main__':
    search.main()
