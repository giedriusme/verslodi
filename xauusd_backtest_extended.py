import xauusd_backtest as bt

bt.TPS = [10.0, 12.0, 15.0, 17.5, 18.0, 20.0, 22.0, 25.0, 30.0]
bt.SLS = [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0]
bt.BASE_TP, bt.BASE_SL = 20.0, 25.0

if __name__ == "__main__":
    bt.main()
