#A-R sampling om een normaalverdeling te simuleren

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import statistics

N= 10**5 # Aantal simulaties A-R algoritme
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij A-R

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

# uitvoeren van de simulaties van AR om de normaalverdeling te vinden
samples=[AcceptanceRejection(PositiefNormaalPDF) for i in range (N)]
mu_MLE = (sum(samples)/len(samples)) # schatter mu
sigma2_MLE = sum([(sample - mu_MLE)**2 for sample in samples])/(N-1) # schatter sigma2

# visualisatie van de verdelingen van samples
x = np.linspace(-5,5)
y = stats.norm.pdf(x, loc=mu_MLE, scale=sigma2_MLE)
plt.plot(y)
plt.plot(stats.norm.pdf(x,loc=0,scale=1)) # pdf van N(0,1) voor vergelijking
plt.show()
