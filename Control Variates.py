# Black Scholes model MC simulatie control variates
# In dit programma wordt een Monte Carlo simulatie zonder variantie reductie en
# met control variates uitgevoerd. De call-optie schatters en varianties worden 
# berekend van beide Monte Carlo simulaties en afgedrukt.

#Importeer packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import yfinance as yf
import statistics
import time

N = 10**5 #Aantal simulaties AR algoritme
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

# Berekent de uitkomst van Black-Scholes-vergelijking
def black_scholes_call(S,K,T,r,sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S *1*stats.norm.cdf(d1) - K * np.exp(-r*T)*stats.norm.cdf(d2)

    return call

# Uitvoeren Monte Carlo zonder antithetic (met put-call parity)
samples=[AcceptanceRejection(PositiefNormaalPDF) for i in range (N)] #past acceptance rejection toe
berekening_ST = ST(S, T, r, sigma, simulaties, samples) 
variantie_mc = statistics.variance(berekening_ST)
mc_simulatie = np.maximum(K-berekening_ST,0) # simuleren het object bij de put-prijs
mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T) # de put-prijs
call_optie_mc = mc_prijs + S - K*np.exp(-r*T) # toepassen put-call parity

# Uitvoeren Monte Carlo met antithetic
samples=[AcceptanceRejection(PositiefNormaalPDF) for i in range (N)] #past acceptance rejection toe
berekening_ST = ST(S, T, r, sigma, simulaties, samples)
M = np.mean(berekening_ST) # gemiddelde van de S(T) waardes
mu = np.maximum(K-berekening_ST,0) # het object dat gesimuleerd wordt
c = -statistics.covariance(berekening_ST,mu)/statistics.variance(berekening_ST) # constante bij control variates
nieuwe_schatter = schatter = mu + c*(berekening_ST-M) 
variantie_mc_cv = statistics.variance(nieuwe_schatter)
mc_cv_prijs = np.mean(nieuwe_schatter)*np.exp(-r*T)
call_optie_mc_cv = mc_cv_prijs + S - K*np.exp(-r*T)

print(f"De Black-Scholes vergelijking geeft als call-optie prijs: {black_scholes_call(S,K,T,r,sigma)}")
print(f"De Monte Carlo simulatie zonder reductie geeft als call-optie prijs: {call_optie_mc}")
print(f"De Monte Carlo simulatie met antithetic sampling geeft als call-optie prijs: {call_optie_mc_cv}")

print(f"De Monte Carlo simulatie zonder reductie heeft als variantie: {variantie_mc}")
print(f"De Monte Carlo simulatie met antithetic sampling heeft als variantie: {variantie_mc_cv}")

