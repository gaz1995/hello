import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import time
from scipy import stats
import ipywidgets as widgets
import altair as alt
import plotly.express as px


## To set the dashboard title
st.title('Financial DashBoard')
st.write("Here is the results for you!")

## To write something in the sidebar
st.sidebar.write("User Input Features")


risk_level = [('High Risk'), ('Moderate High Risk'), ('Moderate Risk'),('Moderate Low Risk'),('Low Risk')]

option = st.sidebar.selectbox(
    'Which of the following options best describes your risk profile?',
     risk_level)


min_inv_horizon = 10
max_inv_horizon = 25
investment_horizon = st.sidebar.slider(
    'Which is your investment horizon (the minimum inv.horizon is 10 years)?',
    value=15,
    min_value=min_inv_horizon,
    max_value=max_inv_horizon
)

monthly_contributions = st.sidebar.number_input(value=1000,label="What is your monthly contribution in dollar amount?",min_value=0,max_value=1000000,step=100)

inflation = st.sidebar.selectbox(label="What is your inflation expectation?",options=[0,0.01,0.02,0.03,0.04,0.05])

initial_investment = st.sidebar.number_input(value=1000,label="What is your initial investment amount in dollar amount?",min_value=0,max_value=1000000,step=500)




stdev_options = [0.05, 0.1, 0.15, 0.20, 0.25]
stdev_target = stdev_options[int(risk_level.index(option))]
leverage_options = [1,2,3,4,5]
leverage_target = leverage_options[int(risk_level.index(option))]


etf_tickers = ["SPY", "IJR", "QQQ", "VUG", "TIP", "IEI", "LQD", "GLD"]

#fetch the price data for each one of the above ETFS with
start_date = "2007-03-09"
today = dt.date.today().strftime("%Y-%m-%d")
price_df = yf.download(etf_tickers, start=start_date, end=today, progress = False)["Adj Close"]

#rename each column to the asset class they represent
price_df = price_df.rename(columns = {"SPY":"BIG_CAPS_US",
                                      "IJR":"SMALL_CAPS_US",
                                      "QQQ":"GROWTH_US",
                                      "VUG":"VALUE_US",
                                      "TIP":"TIPS",
                                      "IEI":"US_TREASURIES",
                                      "LQD":"CORP_BONDS",
                                      "GLD":"GOLD"
                                     })

returns_df = np.log(price_df/price_df.shift(1))
returns_df = returns_df.dropna()

returns_df["full_equity_returns"] = returns_df["BIG_CAPS_US"]
returns_df["sixty_forty_returns"] = returns_df["BIG_CAPS_US"]*0.6 + returns_df["US_TREASURIES"]*0.4

assets = ["BIG_CAPS_US","SMALL_CAPS_US","GROWTH_US","VALUE_US","TIPS","US_TREASURIES","CORP_BONDS","GOLD"]
assets_index = [i for i in range(len(assets))]
asset_classes = returns_df[assets]
stdev_window = 22


# create the risk parity function
def risk_parity(stdev_window=stdev_window, target_stdev=stdev_target, leverage=leverage_target):
    global assets, assets_index, asset_classes

    rolling_stdev = asset_classes.rolling(stdev_window).std() * np.sqrt(252)
    target_hist = target_stdev / rolling_stdev
    target_hist["Sum"] = target_hist.sum(axis=1)
    adj_weight = target_hist[assets].copy()
    for column in assets:
        for row in range(len(target_hist.index)):
            if target_hist["Sum"][row] > leverage:
                adj_weight[column][row] = target_hist[column][row] / target_hist["Sum"][row] * leverage
            else:
                adj_weight[column][row] = target_hist[column][row]

    adj_weight["Sum"] = adj_weight.sum(axis=1)
    adj_weight["RISK_PARITY"] = 0
    for row in range(len(asset_classes)):
        adj_weight.iloc[row, len(assets) + 1] = sum(
            asset_classes.iloc[row, assets_index] * adj_weight.iloc[row - 1, assets_index])
        adj_weight.iloc[0, len(assets) + 1] = np.nan

    return adj_weight

returns_df["strategy"] = risk_parity(target_stdev=stdev_target, leverage=leverage_target)["RISK_PARITY"]
returns_df["strategy_csum"] = returns_df["strategy"].cumsum()
returns_df["full_equity_csum"] = returns_df["full_equity_returns"].cumsum()
returns_df["sixty_forty_csum"] = returns_df["sixty_forty_returns"].cumsum()



## To plot the cumulative sum of each strategy
fig,ax = plt.subplots(figsize=(14,5))
ax.plot(returns_df["strategy_csum"], label="Strategy Cumulative Returns", color="green")
ax.plot(returns_df["full_equity_csum"], label="Full Equity Cumulative Returns", color="blue")
ax.plot(returns_df["sixty_forty_csum"], label="60/40 Cumulative Returns", color="red")
plt.title(f"Our strategy's total return was {int(returns_df['strategy_csum'].max()*100)}% while for full equity was {int(returns_df['full_equity_csum'].max()*100)}% and for the 60/40 was {int(returns_df['sixty_forty_csum'].max()*100)}% during this period.", size=13)
plt.xlabel("Years")
plt.ylabel("Cumulative Return in %")
plt.legend()
st.pyplot(fig)



def max_drawdown(log_returns):
    un_log = np.exp(log_returns)-1
    index = 100*(1+un_log).cumprod()
    peaks = index.cummax()
    daily_drawdown = ((index - peaks)/peaks)*100
    return daily_drawdown

max_d= max_drawdown(returns_df[["strategy","full_equity_returns","sixty_forty_returns"]])


fig, ax = plt.subplots(figsize=(14,5))
ax.plot(max_drawdown(returns_df["strategy"]), color="green", label="Strategy Drawdown")
ax.plot(max_drawdown(returns_df["full_equity_returns"]), color="blue", label="Full Equity Drawdown")
ax.plot(max_drawdown(returns_df["sixty_forty_returns"]), color="red", label="60/40 Drawdown")
plt.title(f"Our strategy's maximum drawdown was {int(max_drawdown(returns_df['strategy']).min())}% while for full equity was {int(max_drawdown(returns_df['full_equity_returns']).min())}% and for 60/40 was {int(max_drawdown(returns_df['sixty_forty_returns']).min())}%.", size=13)
plt.xlabel("Years")
plt.ylabel("Drawdown in %")
plt.legend()
st.pyplot(fig)



def performance(df):
    df = df.dropna()
    annual_average_return = round(df.mean() * 252, 4)
    annual_volatility = round(np.std(df) * np.sqrt(252), 4)
    info_sharpe = round(annual_average_return / annual_volatility, 4)
    cumulative_return = round(df.sum(), 4)
    skewness = round(stats.skew(df), 4)
    kurtosis = round(stats.kurtosis(df), 4)
    value_at_risk = round(-1.645 * np.std(df), 4)
    drawdown = round(max_drawdown(df).min() / 100, 4)
    positive_days = 0
    negative_days = 0
    for i in df:
        if i > 0:
            positive_days += 1
        elif i < 0:
            negative_days += 1
    positive_percent = round(positive_days / (negative_days + positive_days), 4)
    negative_percent = round(negative_days / (negative_days + positive_days), 4)

    performance_table = {"Annual Return": annual_average_return, "Annual Volatility": annual_volatility,
                         "Info Sharpe": info_sharpe, "Cumulative Return": cumulative_return, "Skewness": skewness,
                         "Kurtosis": kurtosis, "Value-at-Risk": value_at_risk, "Max. Drawdown": drawdown,
                         "Positive days": positive_percent, "Negative days": negative_percent}


    return performance_table

strategy = pd.DataFrame(performance(returns_df["strategy"]), index = ["Our Strategy"]).T
full_equity = pd.DataFrame(performance(returns_df["full_equity_returns"]), index = ["100% Equity"]).T
sixty_forty = pd.DataFrame(performance(returns_df["sixty_forty_returns"]), index = ["60% Equity/40% Bonds"]).T
performance_df = pd.merge(strategy, full_equity, left_index=True, right_index=True)
performance_df = pd.merge(performance_df, sixty_forty, left_index=True, right_index=True)
performance_df.index.name = "Performance Stats (in decimals)"

st.table(performance_df)

today_date = dt.datetime.now().year #present day
future_dates = [today_date] #create an array where the first date is today
for i in range(1,investment_horizon+1): #append to futures_dates the yearly dates until the end of the investment horizon
    future_dates.append(today_date + i)

future_expectations = pd.DataFrame()
future_expectations["year"] = future_dates
future_expectations["mean_return"] = performance(returns_df["strategy"])["Annual Return"]
future_expectations.iloc[0,1] = 0
future_expectations["mean_return_csum"] = future_expectations["mean_return"].cumsum()
future_expectations["upper_ci"] = performance(returns_df["strategy"])["Annual Return"] + stats.norm.ppf(0.95) * performance(returns_df["strategy"])["Annual Volatility"]/np.sqrt(252)
future_expectations["lower_ci"] = performance(returns_df["strategy"])["Annual Return"] - stats.norm.ppf(0.95) * performance(returns_df["strategy"])["Annual Volatility"]/np.sqrt(252)
future_expectations.iloc[0,[3,4]] = 0
future_expectations["upper_ci_csum"] = future_expectations["upper_ci"].cumsum()
future_expectations["lower_ci_csum"] = future_expectations["lower_ci"].cumsum()
future_expectations["contributions"] = monthly_contributions*12
future_expectations.iloc[0,7] = initial_investment
future_expectations["contributions_csum"] = future_expectations["contributions"].cumsum()
future_expectations.iloc[0,8] = initial_investment
future_expectations["capital_gains"] = future_expectations["contributions_csum"].shift(1)*future_expectations["mean_return"]
future_expectations.iloc[0,9] = 0
future_expectations["capital_gains_csum"] = future_expectations["capital_gains"].cumsum()
future_expectations["total"] = future_expectations["contributions"] + future_expectations["capital_gains"]
future_expectations["total_csum"] = future_expectations["total"].cumsum()
future_expectations["inflation_loss"] = -future_expectations["total_csum"].shift(1)*inflation
future_expectations.iloc[0,13] = 0
future_expectations["inflation_loss_csum"] = future_expectations["inflation_loss"].cumsum()
future_expectations["real_return_csum"] = future_expectations["total_csum"] + future_expectations["inflation_loss_csum"]
future_expectations = future_expectations.set_index("year")

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(future_expectations["mean_return_csum"]*100, color="red", label="Strategy Return")
ax.plot(future_expectations["upper_ci_csum"]*100, color="blue", label="Upper Confidence Interval", alpha=0.5)
ax.plot(future_expectations["lower_ci_csum"]*100, color="blue", label="Lower Confidence Interval", alpha=0.5)
ax.fill_between(future_expectations.index ,future_expectations["upper_ci_csum"]*100, future_expectations["lower_ci_csum"]*100, alpha=0.2)
plt.title(f"The expected returns fall between {round(future_expectations.iloc[investment_horizon,5]*100,2)}% and {round(future_expectations.iloc[investment_horizon,4]*100,2)}% in the span of {investment_horizon} years.", size=13)
plt.xlabel("Years")
plt.ylabel("Cumulative Returns in %")
plt.legend()
st.pyplot(fig)

chart_data = future_expectations[["total_csum", "real_return_csum"]]
plt.title(f'By {int(future_dates[-1])} the expected total return could reach {int(future_expectations.iloc[investment_horizon,11])} USD, OR {int(future_expectations.iloc[investment_horizon,14])} USD when discounting for inflation.', size=13)
plt.ylabel("Portfolio value in USD")
plt.xlabel("Future Years")
plt.legend(["Total Nominal Returns", "Total Real Returns"])
st.bar_chart(chart_data)


chart_data2 = future_expectations[["contributions_csum", "capital_gains_csum"]]
surpass = future_expectations.index[future_expectations["capital_gains_csum"] > future_expectations["contributions_csum"]]
plt.title(f"Capital gains begin to exceed contribution staring from {surpass}.", size=13)
plt.ylabel("Value in USD")
plt.xlabel("Future Years")
plt.legend(["Monthly Contributions", "Capital Gains"])
st.bar_chart(chart_data2)

returns_df["year"]=returns_df.index.year
grouped_df = returns_df.groupby(returns_df["year"]).sum()
grouped_df["year"]=grouped_df.index
grouped_df=grouped_df[["strategy", "full_equity_returns", "sixty_forty_returns"]]


chart_data3 = grouped_df
plt.ylabel("Past Years")
plt.xlabel("Annual Portfolio Returns in decimals")
plt.title("Our strategy outperforms most of the years")
plt.legend(["Our Strategy", "Full Equity", "60/40"])
st.bar_chart(chart_data3)




last_weights = risk_parity(target_stdev=stdev_target, leverage=leverage_target).iloc[-1]
last_weights = last_weights[assets_index]/sum(last_weights[assets_index])*100

chart_data4 = last_weights
plt.ylabel("Latest weights in %")
plt.xlabel("Asset Class")
plt.title("Verify how the weights change depending on risk apetite")
st.bar_chart(chart_data4)

