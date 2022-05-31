# Code: Monte-Carlo for option price
# Description: https://risksir.com/python/22-monte-carlo-python

# importing necessary modules and defining the constants
# in this example, the risk rate is 3%, stock volatility is 35%, the initial stock price is 15, the option has strike 14 and has 2 years to expiration, and we use 1000 timesteps to simulate price tracks
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

mu = 0.03
n = 1000
T = 2
S0 = 15
K = 14
sigma = 0.35

# number of simulations
M = 100000

# defining time interval and timestep
times = np.linspace(0,T,n+1)
dt = times[1] - times[0]

B = np.random.normal(0, np.sqrt(dt), size=(M,n)).T
St = np.exp( (mu - sigma ** 2 / 2) * dt + sigma * B )

# include an array of 1's
St = np.vstack([np.ones(M), St])

# multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0)
St = S0 * St.cumprod(axis=0)
ExpPayoffT = (1 / M) * np.sum(np.maximum(St[n] - K, 0))
DiscExpPayoffT = ExpPayoffT * np.exp(-mu * T)

 # Black-Scholes-Merton formula, to compare our result
 def BSM_CALL(S, K, T, r, sigma):
 d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
 d2 = d1 - sigma * np.sqrt(T)
 return S * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)
 
 # numpy array that is the same shape as St, to plot the chart
 tt = np.full(shape=(M,n+1), fill_value=times).T
 
 # plotting the chart with Monte-Carlo simulations and printing the result
 plt.figure(figsize = (10,5))
plt.plot(tt, St)
plt.xlabel("Time \((t)\)")
plt.ylabel("Stock Price \((S_t)\)")
plt.title("Stock price simulation\n \(S_0 = {0}, \mu = {1}, \sigma = {2}\)".format(S0, mu, sigma))
plt.show()
print("Expected payoff in T={0} years: {1}".format(T,ExpPayoffT))
print("Discounted Expected payoff: {0}".format(DiscExpPayoffT))
print("BSM price: {0}".format(BSM_CALL(S0, K, T, mu, sigma)))
 
