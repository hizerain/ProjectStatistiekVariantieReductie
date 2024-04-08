#Black Scholes model MC simulatie met antithetic variates
# In dit programma wordt een Monte Carlo simulatie zonder variantie reductie
# en een simulatie met antithetic variates toegepast uitgevoerd. Beide de
# call-optie schatters en varianties worden berekend en afgedrukt

#Importeer packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import yfinance as yf
import timeit
import statistics

N= 10**5 #Aantal simulaties AR algoritme
N_variates = int(N*0.5) # Aantal simulaties voor AR algoritme met antithetic sampling toegepast 
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij AR

# Constanten voor BS formule
S = 100 # Stock Price ($)
K = 100 # Strike Price ($)
T = 1/2 # Time to Maturity (years)
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
        x = stats.bernoulli.rvs(1/2, size=1) 
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

# Berekent de uitkomst van Black-Scholes-vergelijking
def black_scholes_call(S,K,T,r,sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S *1*stats.norm.cdf(d1) - K * np.exp(-r*T)*stats.norm.cdf(d2)

    return call

# Maakt een antithetic sample
def AR_simulaties(antithetic):
    samples = [AcceptanceRejection(PositiefNormaalPDF) for i in range (N_variates)]
    if (antithetic == True):
        Antithetic_sample = samples
        for i in range (N_variates):
            Antithetic_sample.append(-1*samples[i])
        return Antithetic_sample
    else:
        for i in range (N_variates):
            samples.append(AcceptanceRejection(PositiefNormaalPDF))
        return samples

# Uitvoeren Monte Carlo zonder antithetic variates (met put-call parity)
# we genereren het sample, berekenen met het sample S(T), simuleren vervolgens het object en berekenen tot slot de schatter
samples = np.array(AR_simulaties(False))  
berekening_ST = ST(S, T, r, sigma, simulaties, samples) 
variantie_mc = statistics.variance(berekening_ST)
mc_simulatie = np.maximum(K-berekening_ST,0) # simuleren het object bij de put-prijs
mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T) # de put-prijs
call_optie_mc = mc_prijs + S - K*np.exp(-r*T) # toepassen put-call parity

# Uitvoeren Monte Carlo met antithetic variates (met put-call parity)
samples = np.array(AR_simulaties(True))
berekening_ST = ST(S, T, r, sigma, simulaties, samples)
variantie_mc_as = (statistics.variance(berekening_ST))
mc_av_simulatie = np.maximum(K-berekening_ST,0)
mc_av_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
call_optie_mc_as = mc_av_prijs + S - K*np.exp(-r*T)

# printen van resultaten
print(f"De Black-Scholes vergelijking geeft als call-optie prijs: {black_scholes_call(S,K,T,r,sigma)}")
print(f"De Monte Carlo simulatie met antithetic variates geeft als call-optie prijs: {call_optie_mc_as}")

print(f"De Monte Carlo simulatie zonder reductie heeft als variantie: {variantie_mc}")
print(f"De Monte Carlo simulatie met antithetic variates heeft als variantie: {variantie_mc_as}")