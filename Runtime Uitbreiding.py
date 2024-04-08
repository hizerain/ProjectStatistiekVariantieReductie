# Runtime Uitbreiding
# In dit programma wordt de runtime van Uitbreiding project.py getest
# de data wordt opgeslagen in een DataFrame en gevisualiseerd met plt

import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

statement_nul = '''
MC_zonder_reductie()
'''

statement_een = '''
MC_IS()
'''

setup = '''
# Black Scholes model MC simulatie met importance sampling
# In dit programma wordt een Monte Carlo simulatie zonder variantie reductie
# en een simulatie met importance sampling uitgevoerd. Beide de
# call-optie schatters en varianties worden berekend en afgedrukt

# Importeer packages
import numpy as np
import math 
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

# Constanten voor AR-algoritme
N = 10**5 # Aantal simulaties A-R algoritme
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij A-R

# Constanten voor BS formule
S = 100 # Stock Price ($)
K = 110 # Strike Price ($)
T = 1 # Time to Maturity (years)
r = 0.05 # rick-free interest rate
sigma = 0.05 # volatility 

samples=[] # bevat AR-sample
simulaties=np.zeros(N) # bevat S(T) waardes

# PDF van een positieve normaalverdeling
def PositiefNormaalPDF(x):
    return (2/np.sqrt(2*math.pi))*np.exp(-((x)**2)/(2))

# PDF van een exponentiele verdeling
def ExponentieelPDF(x):
    return np.exp(-x)

# simulatie van exp. verdeling dmv inversie
def simuleerExponentieel(param):
    U = np.random.uniform(0,1) 
    Y = -1*np.log(1-U)
    return Y

# Simuleer een variable volgens een vastgestelde pdf (exp(1))
def AcceptanceRejection(pdf):
    U = np.random.uniform(0,1)
    Y = simuleerExponentieel(1)
    if U < (pdf(Y)/(C*ExponentieelPDF(Y))):        
        x = stats.bernoulli.rvs(0.5) 
        if x==0:
            X = Y
            return X
        else:
            X = -1*Y
            return X
    else:
        return AcceptanceRejection(pdf)

# Berekening S(T) waarde
def ST(S, T, r, sigma,simulaties, samples):
    for i in range(len(samples)):
        simulaties[i]=np.log(S) + (r-0.5*sigma**2)*T+np.sqrt(T)*samples[i]*sigma

    return np.exp(simulaties)

# de nieuwe S(T) waar we de nieuwe functie voor gebruiken
def ST_nieuwe_verdeling(S, T, r, sigma, simulaties, sample, theta):
    for i in range(len(samples)):
        simulaties[i]=np.log(S) + np.sqrt(T)*samples[i]*sigma
    return np.exp(simulaties)

# Berekent de uitkomst van Black-Scholes-vergelijking
def black_scholes_call(A,P,T,r,volatiliteit):
    d1 = (np.log(A/P) + (r + volatiliteit**2/2)*T) / (volatiliteit*np.sqrt(T))
    d2 = d1 - volatiliteit* np.sqrt(T)
    
    call = A * 1* stats.norm.cdf(d1) - P * np.exp(-r*T)*stats.norm.cdf(d2)

    return call

# monte carlo simulatie zonder variantie reductie toe te passen
def MC_zonder_reductie():
    samples = np.array([AcceptanceRejection(PositiefNormaalPDF) for i in range (N)])
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    mc_simulatie = np.maximum(berekening_ST-K,0)*np.exp(-r*T)
    call_optie_mc = np.mean(mc_simulatie)
    data_sim_zr_var = np.sqrt(np.sum((mc_simulatie-call_optie_mc)**2)/(N-1))

    return call_optie_mc

# monte carlo simulatie met importance sampling 
def MC_IS():
    theta = np.log(K/S) - 0.17 # de nieuwe parameter
    samples = np.array([AcceptanceRejection(PositiefNormaalPDF) for i in range (N)]) # we creeëren samples
    berekening_ST = ST(S, T, r, sigma, simulaties, samples) # we bereken h(x)
    mc_simulatie = np.maximum(berekening_ST-K,0)
    nieuwe_schatter = mc_simulatie*np.exp(-1/2*theta**2)*(stats.norm.pdf(0,1)/stats.norm.pdf(theta,1)) # de nieuwe schatter waar we h(x)*L(x)
    data_sim_is_var = (statistics.variance(nieuwe_schatter))
    call_optie_mc_is = np.mean(nieuwe_schatter)*np.exp(-r*T)

    return call_optie_mc_is


'''

# aantal simulaties
aantal = 100
ns = np.arange(aantal)

# monte carlo simulatie
ts = [timeit.timeit(statement_nul, setup, number=1) for n in ns]
plt.plot(ns+1, ts, 'or', label = 'Monte Carlo simulatie', color = 'royalblue')

# importance sampling 
ts_2 = [timeit.timeit(statement_een, setup, number=1) for n in ns]
plt.plot(ns+1, ts_2, 'ob', label = 'importance sampling', color = 'magenta')

# data visualiseren 
plt.ylabel('Simulatietijd (s)')
plt.xlabel('Simulatie nummer')
plt.ylim(ymin=0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right") 
plt.show()

# data als dataframe opslaan
# df = pd.DataFrame({'Tijd MC': ts, 'Tijd IS': ts_2})
