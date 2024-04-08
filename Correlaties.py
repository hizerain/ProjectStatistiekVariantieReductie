# Correlaties
# In dit programma wordt de correlatie tussen de control variate en het
# gesimuleerde object bepaald om te bekijken hoe parameters deze correlatie
# beïnvloeden

#Importeer packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics

# Constanten voor AR-algoritme
N = 10**5 # Aantal simulaties A-R algoritme
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij A-R

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

# Berekent de correlatie van het object en de control variate by de call-optie
def correlatie_call():
    samples = [AcceptanceRejection(PositiefNormaalPDF) for i in range (N)] # maken een sample
    berekening_ST = ST(S, T, r, sigma, simulaties, samples) 
    X = np.maximum(berekening_ST-K,0) # gesimuleerde object
    Y = berekening_ST # control variate
    rho2 = (statistics.covariance(X,Y)**2)/(statistics.variance(X)*statistics.variance(Y))
    print(rho2)

# Berekent de correlatie van het object en de control variate by de put-optie
def correlatie_put():
    samples = [AcceptanceRejection(PositiefNormaalPDF) for i in range (N)]
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    X = np.maximum(K-berekening_ST,0) 
    Y = berekening_ST 
    rho = statistics.covariance(X,Y)/(np.sqrt(statistics.variance(X))*np.sqrt(statistics.variance(Y)))
    print(rho)

correlatie_call()
correlatie_put()

