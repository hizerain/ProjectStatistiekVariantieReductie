# Black Scholes model Monte Carlo Simulatie
# In dit programma wordt een Monte Carlo simulatie zonder variantie reductie,
# met antithetic variates, met control variates én met antithetic én control variates
# uitgevoerd. De call-optie schatters en varianties worden berekend van de 4 verschillende
# soorten monte carlo simulaties en afgedrukt

#Importeer packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import statistics

# Constanten voor AR-algoritme
N = 10**5 # Aantal simulaties A-R algoritme
N_variates = int(0.5*N) # Aantal simulaties voor AR algoritme met antithetic sampling toegepast
C = np.sqrt((2*np.exp(1))/np.pi) # optimale C bij A-R
antithetic = False # Of we Antithetic Variates toepassen in een simulatie of niet

# Constanten voor BS formule
S = 100 # Stock Price ($)
K = 100 # Strike Price ($)
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

# Berekent de uitkomst van Black-Scholes-vergelijking
def black_scholes_call(S,K,T,r,sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S *1*stats.norm.cdf(d1) - K * np.exp(-r*T)*stats.norm.cdf(d2)

    return call

# maakt een antithetic sample
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

# berekening van de nieuwe schatter Control variates
def CV(berekening_ST):
    M = np.mean(berekening_ST) 
    mu = np.maximum(K-berekening_ST,0) # gesimuleerde object
    constante_cv = -statistics.covariance(berekening_ST,mu)/statistics.variance(berekening_ST)
    nieuwe_schatter = mu + constante_cv*(berekening_ST-M)

    return nieuwe_schatter

# berekening van de monte carlo schatter zonder variantie reductie
# we genereren het sample, berekenen met het sample S(T), simuleren vervolgens het object en berekenen tot slot de schatter

def MC_zonder_reductie():
    samples = np.array(AR_simulaties(False))
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    data_sim_zr_var = statistics.variance(berekening_ST)
    mc_simulatie = np.maximum(K-berekening_ST,0)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie_mc = mc_prijs + S - K*np.exp(-r*T)
    
    print(f"De Monte Carlo simulatie zonder reductie geeft als call-optie prijs: {call_optie_mc}")
    print(f"De Monte Carlo simulatie zonder reductie heeft als variantie: {data_sim_zr_var}")

    return call_optie_mc

# berekening van de schatter met antithetic variates
def MC_AV():
    samples = np.array(AR_simulaties(True))
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    data_sim_av_var = statistics.variance(berekening_ST)
    mc_simulatie = np.maximum(K-berekening_ST,0)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie_mc_av = mc_prijs + S - K*np.exp(-r*T)

    print(f"De Monte Carlo simulatie met antithetic variates geeft als call-optie prijs: {call_optie_mc_av}")
    print(f"De Monte Carlo simulatie met antithetic variates heeft als variantie: {data_sim_av_var}")

    return call_optie_mc_av

# berekening van de schatter met control variates
def MC_CV():
    samples = np.array(AR_simulaties(False))
    berekening_ST = ST(S, T, r, sigma, simulaties, samples) # dit wordt gebruikt als control variate
    mc_simulatie = CV(berekening_ST)
    data_sim_cv_var = statistics.variance(mc_simulatie)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie_mc_cv = mc_prijs + S - K*np.exp(-r*T)

    print(f"De Monte Carlo simulatie met control variates geeft als call-optie prijs: {call_optie_mc_cv}")
    print(f"De Monte Carlo simulatie met control variates heeft als variantie: {data_sim_cv_var}")
    
    return call_optie_mc_cv

# berekening van de schatter met antithetic en control variates
def MC_AV_CV():
    samples = np.array(AR_simulaties(True))
    berekening_ST = ST(S, T, r, sigma, simulaties, samples)
    mc_simulatie = CV(berekening_ST)
    data_sim_av_cv_var = statistics.variance(mc_simulatie)
    mc_prijs = np.mean(mc_simulatie)*np.exp(-r*T)
    call_optie_mc_av_cv = mc_prijs + S - K*np.exp(-r*T)

    print(f"De Monte Carlo simulatie met antithetic en control variates geeft als call-optie prijs: {call_optie_mc_av_cv}")
    print(f"De Monte Carlo simulatie met antithetic en control variates heeft als variantie: {data_sim_av_cv_var}")
    
    return call_optie_mc_av_cv

print(f"De Black-Scholes vergelijking geeft als call-optie prijs: {black_scholes_call(S,K,T,r,sigma)}")
MC_zonder_reductie()
MC_AV()
MC_CV()
MC_AV_CV()



