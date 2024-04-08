# Runtime Basisproject
# In dit programma wordt de runtime van Basisproject.py getest
# de data wordt opgeslagen in een DataFrame en gevisualiseerd met plt

import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# simulatie tijd voor Acceptance-Rejection
statement_nul = '''
AR_simulaties(False)
'''

# simulatie tijd voor de Monte Carlo Simulatie
statement_een = '''
MC_zonder_reductie()
'''

# simulatie tijd voor Monte Carlo Simulatie met Antithetic Sampling toegepast
statement_twee = '''
MC_AV()
'''

# simulatie tijd voor Monte Carlo Simulatie met Control Variates toegepast
statement_drie = '''
MC_CV()
'''

# simulatie tijd voor Monte Carlo Simulatie met Antithetic Sampling en Control Variates toegepast
statement_vier = '''
MC_AV_CV()
'''

setup_vier = '''
#Black Scholes model Monte Carlo Simulatie
#Importeer packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
#import yfinance as yf 
import statistics

# Constanten voor AR-algoritme
N = 10**5 # Aantal simulaties A-R algoritme
N_variates = int(0.5*N)
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij A-R
antithetic = False # Of we Antithetic Variates toepassen in een simulatie of niet

# COnstante van BS vergelijking
S = 942.89 # Stock price ($)
K = 585 # Strike price ($)
T = 1 # Time to maturity (years)
r = 0.05 # risk-free interest (percentage)
sigma = 2.56982779418945  # volatility (percentage)

sim = [] # lijst met de waardes van S(T)
samples=[] 

simulaties=np.zeros(N)

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

#Berekent de Black Scholes formule
def black_scholes_call(S,K,T,r,sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S *1*stats.norm.cdf(d1) - K * np.exp(-r*T)*stats.norm.cdf(d2)

    return call

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

def CV(berekening_ST):
    M = np.mean(berekening_ST)
    mu = np.maximum(K-berekening_ST,0)
    constante_cv = -statistics.covariance(berekening_ST,mu)/statistics.variance(berekening_ST)
    nieuwe_schatter = mu + constante_cv*(berekening_ST-M)

    return nieuwe_schatter

def MC_zonder_reductie():
    samples = AR_simulaties(False)
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    data_sim_zr_var.append(statistics.variance(berekening_ST))
    mc_simulatie = np.maximum(K-berekening_ST,0)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie = mc_prijs + S - K*np.exp(-r*T)
    
    return call_optie

def MC_AV():
    samples = AR_simulaties(True)
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    data_sim_av_var.append(statistics.variance(berekening_ST))
    mc_simulatie = np.maximum(K-berekening_ST,0)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie = mc_prijs + S - K*np.exp(-r*T)

    return call_optie

def MC_CV():
    samples = AR_simulaties(False)
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    mc_simulatie = CV(berekening_ST)
    data_sim_cv_var.append(statistics.variance(mc_simulatie))
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie = mc_prijs + S - K*np.exp(-r*T)
    
    return call_optie

def MC_AV_CV():
    samples = AR_simulaties(True)
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    mc_simulatie = CV(berekening_ST)
    data_sim_av_cv_var.append(statistics.variance(mc_simulatie))
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie = mc_prijs + S - K*np.exp(-r*T)
    
    return call_optie
'''
# aantal simulaties
aantal = 100
ns = np.arange(aantal)

# AR simulatie
# ar = [timeit.timeit(statement_nul, setup_vier, number=1) for n in ns]

# monte carlo simulatie
# ts = [timeit.timeit(statement_een, setup_vier, number=1) for n in ns]
# plt.plot(ns+1, ts, 'or', label = 'Monte Carlo simulatie', color = 'royalblue')

# Antithetic Variates 
# ts_2 = [timeit.timeit(statement_twee, setup_vier, number=1) for n in ns]
# plt.plot(ns+1, ts_2, 'ob', label = 'Antithetic Variates', color = 'magenta')

# Control Variates
# ts_3 = [timeit.timeit(statement_drie, setup_vier, number=1) for n in ns]
# plt.plot(ns+1, ts_3, 'og', label = 'Control Variates', color = 'red')

# Antithetic Variates en Control Variates
ts_4 = [timeit.timeit(statement_vier, setup_vier, number=1) for n in ns]
plt.plot(ns+1, ts_4, 'om', label = 'Antithetic en Control Variates', color = 'gold')

# plotten van de resultaten 
plt.ylabel('Simulatietijd (s)')
plt.xlabel('Simulatie nummer')
plt.ylim(ymin=0, ymax=6)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right") 
plt.show()

# resultaten van runtimes als dataframe opslaan
df = pd.DataFrame({'Tijd AR': ar, 'Tijd MC': ts, 'Tijd AV': ts_2, 'Tijd CV': ts_3, 'Tijd AV&CV': ts_4})
