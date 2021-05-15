import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import time
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle

#Dashboard title
st.title('Financial Planning Dashboard')

st.text("The dashboard consists of two parts: \n1. Visulization \n2. Strategy Fundamentals")
st.header("1. Visualization")

#Get the inputs from the user for various variables
st.sidebar.write("User Input Features")
risk_level = [('Low Risk'), ('Moderate Low Risk'), ('Moderate Risk'), ('Moderate High Risk'), ('High Risk')]
option = st.sidebar.selectbox('Which of the following options best describes your risk profile?', risk_level)
investment_horizon = st.sidebar.slider('Which is your investment horizon (the minimum inv.horizon is 10 years)?',value=20,min_value=10,max_value=40)
monthly_contribution = st.sidebar.number_input(value=1000,label="What is your monthly contribution in dollar amount?",min_value=0,max_value=1000000,step=100)
inflation = st.sidebar.selectbox(label="What is your annual inflation expectation?",options=[0,0.01,0.015,0.02,0.025,0.03])
initial_investment = st.sidebar.number_input(value=10000,label="What is your initial investment amount in dollar amount?",min_value=10000,max_value=1000000,step=500)

#sidebar text
st.sidebar.write(f"You have a {option} appetite. You decided to begin investing with {initial_investment}$ and to contribute monthly with {monthly_contribution}$ during {investment_horizon} year. Finally you expect an annual inflation of {inflation*100}%.")

#match the risk levels to the standard deviation and leverage values
stdev_options = [0.05, 0.1, 0.15, 0.20, 0.25]
stdev_target = stdev_options[int(risk_level.index(option))]
leverage_options = [1,2,3,4,5]
leverage_target = leverage_options[int(risk_level.index(option))]

#define the ticker list
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
                                      "GLD":"GOLD"})

#create a dataframe with the log returns of all the asset classes
returns_df = np.log(price_df/price_df.shift(1))
returns_df = returns_df.dropna()

#define or benchmarks, one will be full US equities the other will be a 60/40 mix of equitie and treasuries
returns_df["full_equity_returns"] = returns_df["BIG_CAPS_US"]
returns_df["sixty_forty_returns"] = returns_df["BIG_CAPS_US"]*0.6 + returns_df["US_TREASURIES"]*0.4

#laying down our risk parity assumptions and base frames
assets = ["BIG_CAPS_US","SMALL_CAPS_US","GROWTH_US","VALUE_US","TIPS","US_TREASURIES","CORP_BONDS","GOLD"]
assets_index = [i for i in range(len(assets))]
asset_classes = returns_df[assets]
stdev_window = 22

#creating the risk parity strategy function and define the column in the returns_df
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

#cumulative returns for the main strategy and benchmarks
returns_df["strategy_csum"] = returns_df["strategy"].cumsum()
returns_df["full_equity_csum"] = returns_df["full_equity_returns"].cumsum()
returns_df["sixty_forty_csum"] = returns_df["sixty_forty_returns"].cumsum()

st.subheader("First we will consider the historical timeline")
st.write("In this plot you can verify the behaviour of these portfolios when holding them from 2007 to the present date!")
#plot the cumulative returns of the strategy and benchmarks
fig1,sp = plt.subplots(figsize=(14, 5))
sp.plot(returns_df["strategy_csum"], label="Strategy Cumulative Returns", color="green")
sp.plot(returns_df["full_equity_csum"], label="Full Equity Cumulative Returns", color="blue")
sp.plot(returns_df["sixty_forty_csum"], label="60/40 Cumulative Returns", color="red")
top_cumsum = returns_df["strategy_csum"].max()+1
min_cumsum = returns_df["full_equity_csum"].min()
re = Rectangle((13850,-0.7),1000,top_cumsum, facecolor="r", alpha=0.1)
re2 = Rectangle((18250,-0.7),200,top_cumsum, facecolor="r", alpha=0.1)
plt.annotate("Global Financial Crisis",(13900,top_cumsum-0.8),weight='bold')
plt.annotate("2020 Covid Crisis",(18000,-0.7),weight='bold')
sp.add_patch(re)
sp.add_patch(re2)
plt.title(f"Our strategy's total return was {int(returns_df['strategy_csum'].max()*100)}% while for full equity was {int(returns_df['full_equity_csum'].max()*100)}% and for the 60/40 was {int(returns_df['sixty_forty_csum'].max()*100)}% during this period.", size=13)
plt.xlabel("Years")
plt.ylabel("Cumulative Return in %")
plt.legend();
st.pyplot(fig1)

#Drawdown calculation function
def max_drawdown(log_returns):
    un_log = np.exp(log_returns)-1
    index = 100*(1+un_log).cumprod()
    peaks = index.cummax()
    daily_drawdown = ((index - peaks)/peaks)*100
    return daily_drawdown

st.write("Drawdown is very usefull at verifiying the peak-to-trough. Are you able to stomach a drop of more than 50%?")
#Plot drawdowns
fig2, sp = plt.subplots(figsize=(14,5))
sp.plot(max_drawdown(returns_df["strategy"]), color="green", label="Strategy Drawdown")
sp.plot(max_drawdown(returns_df["full_equity_returns"]), color="blue", label="Full Equity Drawdown")
sp.plot(max_drawdown(returns_df["sixty_forty_returns"]), color="red", label="60/40 Drawdown")
re = Rectangle((13850,0),1000,-56, facecolor="r", alpha=0.1)
re2 = Rectangle((18250,0),200,-56, facecolor="r", alpha=0.1)
plt.annotate("Global Financial Crisis",(13900,-58),weight='bold')
plt.annotate("2020 Covid Crisis",(18000,-58),weight='bold')
sp.add_patch(re)
sp.add_patch(re2)
plt.title(f"Our strategy's maximum drawdown was {int(max_drawdown(returns_df['strategy']).min())}% while for full equity was {int(max_drawdown(returns_df['full_equity_returns']).min())}% and for 60/40 was {int(max_drawdown(returns_df['sixty_forty_returns']).min())}%.", size=13)
plt.xlabel("Years")
plt.ylabel("Drawdown in %")
plt.legend();
st.pyplot(fig2)

#A function that creates a table with various performance metrics
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

st.write("Here you can quantitatively compare the portfolios between various metrics.")
#Merge the peformance of our stratgy and benchmarks in a single dataframe
def performance_table(returns_df=returns_df):
    strategy = pd.DataFrame(performance(returns_df["strategy"]), index = ["Our Strategy"]).T
    full_equity = pd.DataFrame(performance(returns_df["full_equity_returns"]), index = ["100% Equity"]).T
    sixty_forty = pd.DataFrame(performance(returns_df["sixty_forty_returns"]), index = ["60% Equity 40% Bonds"]).T
    performance_df = pd.merge(strategy, full_equity, left_index=True, right_index=True)
    performance_df = pd.merge(performance_df, sixty_forty, left_index=True, right_index=True)
    performance_df.index.name = "Performance Stats (in decimals)"
    return performance_df
performance_table = performance_table()
st.table(performance_table)

#dataframe with future expectations
today_date = dt.datetime.now().year
future_dates = [today_date]
for i in range(1,investment_horizon+1):
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
future_expectations["contributions"] = monthly_contribution*12
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

st.markdown("***Now we consider the future time horizon. Here, everything is an expectation and nothing is guaranteed***")
st.write("In this plot we can see the long run expected trend of our strategy and the respective confidence interval.")
#plot future returns with confidence interval
fig3, ax = plt.subplots(figsize=(14,5))
ax.plot(future_expectations["mean_return_csum"]*100, color="red", label="Strategy Return")
ax.plot(future_expectations["upper_ci_csum"]*100, color="blue", label="Upper Confidence Interval", alpha=0.5)
ax.plot(future_expectations["lower_ci_csum"]*100, color="blue", label="Lower Confidence Interval", alpha=0.5)
ax.fill_between(future_expectations.index ,future_expectations["upper_ci_csum"]*100, future_expectations["lower_ci_csum"]*100, alpha=0.2)
plt.title(f"The expected total returns fall between {round(future_expectations.iloc[investment_horizon,5]*100,2)}% and {round(future_expectations.iloc[investment_horizon,4]*100,2)}% in the span of {investment_horizon} years, considering our past performance.", size=13)
plt.xlabel("Years")
plt.ylabel("Cumulative Returns in %")
plt.legend();
st.pyplot(fig3)

st.write("We must consider the effect of inflation on the portfolio's returns. Staying in cash is a bad idea if you expect positive inflation.")
#plot nominal and real returns
fig4, ax = plt.subplots(figsize=(14,5))
future_expectations[["total_csum", "real_return_csum", "inflation_loss_csum"]].plot(kind="bar",stacked=False,ax=ax, color=["tab:blue", "tab:orange", "tab:red"])
plt.title(f'By {int(future_dates[-1])} the expected total return could reach {int(future_expectations.iloc[investment_horizon,11])} USD, or {int(future_expectations.iloc[investment_horizon,14])} USD when discounting for inflation.', size=13)
plt.ylabel("Portfolio value in USD")
plt.xlabel("Future Years")
ax.legend(["Total Nominal Returns", "Total Real Returns","Loss of purchasing power"])
ax.tick_params(labeltop=False, labelright=True)
plt.grid(axis="y", linestyle="--")
ax.set_axisbelow(True);
st.pyplot(fig4)

st.write("Savings make for most of the portfolio in the early stages and capital gains overtake later. Both are crutial for building wealth!")
#plot capital gains and contributions
fig5, ax = plt.subplots(figsize=(14,5),sharey=True)
future_expectations[["contributions_csum", "capital_gains_csum"]].plot(kind="bar",stacked=True,ax=ax,color=["tab:orange", "tab:blue"])
surpass = future_expectations.index[future_expectations["capital_gains_csum"] > future_expectations["contributions_csum"]].min()
plt.ylabel("Value in USD")
plt.xlabel("Future Years")
years_array = [i for i in future_expectations.index]
if surpass is not np.nan:
    ax.axvline(x=years_array.index(surpass), color="black",ls='--')
    plt.title(f"Capital gains begin to exceed contribution staring from {surpass}.", size=13)
    ax.legend(["Surpass Year", "Monthly Contributions", "Capital Gains"])
else:
    plt.title(f"Capital gains do not exceed the monthly contribution with these chosen preferences.", size=13)
    ax.legend(["Monthly Contributions", "Capital Gains"])
ax.tick_params(labeltop=False, labelright=True)
ax.grid(axis="y", linestyle="--")
ax.set_axisbelow(True);
st.pyplot(fig5)

st.write("What do you value in a portfolio? Consistency, high returns, downside protection? Verify if these annual results satisfy you.")
#define and plot the annual returns for each strategy
returns_df["year"]=returns_df.index.year
grouped_df = returns_df.groupby(returns_df["year"]).sum()
grouped_df["year"]=grouped_df.index
grouped_df=grouped_df[["year", "strategy", "full_equity_returns", "sixty_forty_returns"]]
fig6,ax = plt.subplots(figsize=(14,5))
outperform_count = 0
for year in grouped_df["year"]:
    if grouped_df["strategy"][year]>grouped_df["full_equity_returns"][year] and grouped_df["strategy"][year]>grouped_df["sixty_forty_returns"][year]:
        outperform_count += 1
grouped_df[["year", "strategy", "full_equity_returns", "sixty_forty_returns"]].plot(x="year", kind="bar", ax=ax, color=["tab:green", "tab:blue", "tab:orange"])
plt.ylabel("Annual Portfolio Returns in decimals")
plt.xlabel("Past Years")
plt.title(f"Our strategy outperforms {outperform_count} out of the {(dt.date.today()).year-2007} past years in terms of returns.")
plt.grid(axis="y", linestyle="--")
plt.ylim(-0.5,0.5)
ax.set_axisbelow(True)
ax.tick_params(labeltop=False, labelright=True)
mean_strategy = grouped_df["strategy"].mean()
ax.axhline(mean_strategy, color="black", ls='--');
plt.legend(["Our Strategy Mean", "Our Strategy", "Full Equity", "60/40", "Mean"], loc="lower right");
st.pyplot(fig6)

st.write("The weights are ever-changing and depend on the risk profile you choose")          
#define and plot the latest weights in our portfolio
last_weights = risk_parity(target_stdev=stdev_target, leverage=leverage_target).iloc[-1]
last_weights = last_weights[assets_index]/sum(last_weights[assets_index])*100
fig7, ax = plt.subplots(figsize=(14,5))
last_weights.plot(x="year", kind="bar", ax=ax, color=['tab:blue', 'tab:red', 'seagreen', 'tab:orange', 'tab:purple', 'teal', 'midnightblue', 'gold'])
for x in range(len(assets_index)):
    plt.text(x-0.125, last_weights[x]+1, f"{round(last_weights[x],2)}%")
plt.ylim(0,55)
plt.ylabel("Latest weights in %")
plt.xlabel("Asset Class")
plt.title(f"Since we follow risk parity, most of the portfolio will be allocated to safer assets. That is why bonds make for {round(last_weights[['US_TREASURIES','TIPS','CORP_BONDS']].sum())}% of the portfolio")
ax.tick_params(labeltop=False, labelright=True)
plt.grid(axis="y", linestyle="--")
ax.set_axisbelow(True);
st.pyplot(fig7)

st.write("This matrix allows us to verify how each assets behaves in relation to the others. Low and negative correlations are key for diversification purposes!") 
#plot correlation matrix
correlation_df = returns_df[assets].corr()
fig8, ax = plt.subplots(1,1,figsize=(12,10))
sns.heatmap(correlation_df, annot=True, cmap="viridis", cbar_kws={"label": "Correlation Intensity: 1 =Perfectly Correlated and -1 =Inversly Correlated"})
plt.ylabel("Asset Classes")
plt.title("Correlation between the assets proves what we already know: bonds and stocks have low to negative correlation.", pad=20);
st.pyplot(fig8)

st.header("2. Strategy Fundamentals")
st.write("**The 60/40 Benchmark Portfolio**")          
st.write("We use the traditional 60/40 portfolio as our benchmark, which consists of an allocation of 60% to equities (“SPY” was used in our case) and 40% to bonds (iShares 3-7 Year Treasury Bond ETF was used in our case). The idea of having a benchmark portfolio is to provide a comparison mechanism against our own trading strategy. The primary advantage of such portfolio lies in the diversification effect offered by the partially uncorrelated nature of stocks and bonds. Also, the 60/40 is designed to provide “equity-like returns with bond-like volatility”. The incorporation of bonds in the portfolio should thus increase the risk-adjusted return and therefore the Sharpe Ratio. The historical effect has been to lower maximum drawdowns compared to portfolio fully invested in equities. Although 60/40 portfolio brings some diversification effect, its risk is still highly concentrated mainly coming from the equities. Empirical test has pointed out that a portfolio that has 60% of the capital invested in equities actually has more like 80% of its risk allocated to equities. This is due to the fact that equities tend to carry so much more risk than other asset classes.  Consequently, the idea of allocating risk instead of allocating capital was emerged, which is the central point of risk parity. We will provide more detailed explanation in the following sessions.")

st.write("**The Idea behind the Risk Parity and All-Weather Portfolio**")           
st.write("The idea of “All weather strategy” was created to take advantage of the powers of diversification. It is a risk-balanced portfolio of asset classes constructed to provide equity-like returns with far less risk (hence a higher Sharpe ratio). In addition to it, the strategy is designed to provide consistent returns across a wide range of economic environment, therefore, being a robust strategy.")
st.write("The rationale behind such strategy is as the following:")
st.write("1. All asset classes are priced to have long-term expected return above cash and their return above cash is proportional to their risk. Empirical studies from “Bridgewater” have shown that asset classes tend to have similar Sharpe ratios.")
st.write("2. Since asset classes provide similar Sharpe ratios, one can use leverage to achieve desired level of return for any asset class.")
st.write("In our particular case, we adopted the principals of “All Weather” and created investment plan adapting to the individual investor´s risk appetite, investment horizon.")
          
          
          
          
          
          
          
          
          
