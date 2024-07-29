import backtrader as bt
import yfinance as yf
import pandas as pd

# Define a simple Buy and Hold strategy
class BuyAndHold(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        if not self.position:  # not in the market
            self.buy()  # buy and hold

# Fetch historical data
data = yf.download('AAPL', '2020-01-01', '2023-01-01')
data = bt.feeds.PandasData(dataname=data)

# Initialize Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(BuyAndHold)
cerebro.adddata(data)
cerebro.broker.setcash(10000)  # Set initial cash
cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # Set stake size

# Run backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Plot the result
cerebro.plot()