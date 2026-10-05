import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

initial = 100
risk_free = 0.05
volatility = 0.20
time_to_mature = 1
n_of_simulations = 10000
simulation_counts = [100,500,1000,5000,10000,50000,100000]
monte_carlo_prices = []
antithetic_monte_carlo_prices = []

for n_of_simulations in simulation_counts:
    np.random.seed(40)
    list_of_z = np.random.normal(0,1,n_of_simulations)
    antithetic_z = -list_of_z
    strike_price = 100
    
    future_prices = initial * np.exp(((risk_free-(0.5*(volatility**2)))*time_to_mature)+(volatility*np.sqrt(time_to_mature)*list_of_z))
    antithetic_future_prices = initial * np.exp(((risk_free-(0.5*(volatility**2)))*time_to_mature)+(volatility*np.sqrt(time_to_mature)*antithetic_z))
    
    call_payoff = np.maximum(future_prices - strike_price, 0)
    antithetic_call_payoff = np.maximum(antithetic_future_prices - strike_price, 0)

    average_payoff = np.mean(call_payoff)

    paired_payoff = (call_payoff+antithetic_call_payoff)/2

    average_paired_payoff = np.mean(paired_payoff)

    monte_carlo_price_paired = np.exp(-risk_free*time_to_mature)*average_paired_payoff
    antithetic_monte_carlo_prices.append(monte_carlo_price_paired)
    
    monte_carlo_price = np.exp(-risk_free*time_to_mature)*average_payoff
    monte_carlo_prices.append(monte_carlo_price)
    
    
np.random.seed(40)
list_of_z = np.random.normal(0,1,n_of_simulations)
strike_price = 100

print(f"First 10 standard normal draws: {list_of_z[0:10]}")
print(f"Mean of standard normal draws: {list_of_z.mean():.2f}")
print(f"Standard deviation of standard normal draws: {list_of_z.std():.2f}")

future_prices = initial * np.exp(((risk_free-(0.5*(volatility**2)))*time_to_mature)+(volatility*np.sqrt(time_to_mature)*list_of_z))

print(f"First 10 simulated stock prices: {future_prices[0:10]}")
print(f"Mean simulated stock price: {future_prices.mean():.2f}")
print(f"Standard deviation of simulated stock prices: {future_prices.std():.2f}")

plt.hist(future_prices, bins = 50, range = [45,200])
plt.xlabel("Stock Price at Maturity")
plt.ylabel("Frequency")
plt.title("Simulated Stock Price Distribution at Maturity")
plt.show()
plt.close()

call_payoff = np.maximum(future_prices - strike_price, 0)

print(f"First 10 call payoffs: {call_payoff[0:10]}")

average_payoff = np.mean(call_payoff)
print(f"Average call payoff: {average_payoff:.2f}")

monte_carlo_price = np.exp(-risk_free*time_to_mature)*average_payoff
print(f"Monte Carlo option price: {monte_carlo_price:.2f}")

d1 = (np.log(initial/strike_price)+((risk_free+(0.5*(volatility**2))*time_to_mature)))/(volatility*np.sqrt(time_to_mature))
print(f"Black-Scholes d1: {d1:.2f}")

d2 = d1-volatility*np.sqrt(time_to_mature)
print(f"Black-Scholes d2: {d2:.2f}")

nd1 = norm.cdf(d1)
nd2 = norm.cdf(d2)
print(f"N(d1): {nd1:.4f}")
print(f"N(d2): {nd2:.4f}")

black_scholes_price = (initial*nd1)-(strike_price*np.exp(-risk_free*time_to_mature)*nd2)
print(f"Black-Scholes option price: {black_scholes_price:.2f}")

pricing_error = abs(monte_carlo_price-black_scholes_price)
print(f"Monte Carlo pricing error: {pricing_error:.4f}")

standard_error = np.exp(-risk_free*time_to_mature)*(call_payoff.std(ddof=1)/np.sqrt(n_of_simulations))
print(f"Monte Carlo standard error: {standard_error:.4f}")

antithetic_standard_error = np.exp(-risk_free*time_to_mature)*(paired_payoff.std(ddof=1)/np.sqrt(n_of_simulations))
print(f"Antithetic standard error: {antithetic_standard_error:.4f}")

confidence_interval_lower = monte_carlo_price-(1.96*standard_error)
confidence_interval_upper = monte_carlo_price+(1.96*standard_error)

print(f"Monte Carlo 95% confidence interval lower bound: {confidence_interval_lower:.4f}")
print(f"Monte Carlo 95% confidence interval upper bound: {confidence_interval_upper:.4f}")

print(f"Monte Carlo convergence prices: {monte_carlo_prices}")
print(f"Antithetic Monte Carlo convergence prices: {antithetic_monte_carlo_prices}")

variance_reduction = (1-((antithetic_standard_error**2)/(standard_error**2)))*100
print(f"Variance reduction: {variance_reduction:.2f}%")

antithetic_confidence_interval_lower = antithetic_monte_carlo_prices[-1]-(1.96*antithetic_standard_error)
antithetic_confidence_interval_upper = antithetic_monte_carlo_prices[-1]+(1.96*antithetic_standard_error)

print(f"Antithetic Monte Carlo 95% confidence interval lower bound: {antithetic_confidence_interval_lower:.4f}")
print(f"Antithetic Monte Carlo 95% confidence interval upper bound: {antithetic_confidence_interval_upper:.4f}")

plt.plot(simulation_counts,monte_carlo_prices,label="Monte Carlo")
plt.plot(simulation_counts,antithetic_monte_carlo_prices,label="Antithetic Monte Carlo")
plt.xscale("log")
plt.title("Monte Carlo Convergence")
plt.xlabel("Number of Simulations")
plt.ylabel("Monte Carlo Option Price")
plt.axhline(black_scholes_price,linestyle="--",color="orange",label="Black_Scholes")
plt.legend()
plt.show()
plt.close()
