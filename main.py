import numpy as np

Initial = 100
risk_Free = 0.05
Volatility = 0.20
time_To_Mature = 1
N_of_Simulations = 10000

for i in range(N_of_Simulations):
  print(np.random.normal(0,1))
