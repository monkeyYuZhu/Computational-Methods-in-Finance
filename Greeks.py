# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:52:49 2020

@author: Yu Zhu

There are two approaches to calculate Greeks: analytical function and finite difference 
"""
#===================================================================================================================
import numpy as np
import math
from scipy.stats import norm

#===================================================================================================================
'''
A financial institution has just sold some seven-month European call options on the Japanese yen.
Suppose that the spot exchange rate is 0.80 cent per yen, 
the strike price is 0.81 cent per yen, 
the risk free interest rate in the US is 8% per annum, 
the risk free interest rate in Japan is 5% per annum,
and the volatility of the yen is 15% per annum.
Calculate the delta, gamma, vega, theta andrho of the option.
Interpret each number. 

Foreign currency option
a foreign currency is analogous to a stock paying a known dividend yield (John Hull)

s0 = 0.8
k = 0.81
r = 0.08
rf = 0.05
sigma = 0.15
T = 7/12 # which is delta previously
'''

# method to get Greeks
def greeks(s0, k, r, rf, sigma, T, c, p):
    '''
    s0: asset price
    k: strike price
    r: risk free rate
    rf: divedend yield
    sigma: implied volatility
    T: time to maturity (annual)
    c: is it call
    p: is it put
    '''
    d1 = (np.log(s0/k) + (r - rf + 0.5*sigma**2)*T)*(sigma*math.sqrt(T))**(-1)
    d2 = d1 - sigma*math.sqrt(T)
    greeks_dict = {}

    # greek value, rf here is a dividend yield, analytical solution
    if c:
        n_d1 = float(norm.cdf(d1))
        n_d2 = float(norm.cdf(d2))
        n_pdf_d1 = (2*math.pi)**(-0.5)*math.exp(-0.5*d1**2)
    
        gamma = n_pdf_d1*math.exp(-rf*T)*(s0*sigma*math.sqrt(T))**(-1)
        vega = s0*math.sqrt(T)*n_pdf_d1*math.exp(-rf*T)
        delta = math.exp(-rf*T)*n_d1
        theta = (-1)*(s0*n_pdf_d1*sigma*math.exp(-rf*T))/(2*math.sqrt(T)) + rf*s0*n_d1*math.exp(-rf*T) - r*k*math.exp(-r*T)*n_d2
        rho = k*T*math.exp(-r*T)*n_d2
    elif p:
        n_d1 = float(norm.cdf(-d1))
        n_d2 = float(norm.cdf(-d2))
        n_pdf_d1 = (2*math.pi)**(-0.5)*math.exp(-0.5*(-1)*d1**2)
        
        gamma = n_pdf_d1*math.exp(-rf*T)*(s0*sigma*math.sqrt(T))**(-1)
        vega = s0*math.sqrt(T)*n_pdf_d1*math.exp(-rf*T)
        delta = (-1)*math.exp(-rf*T)*n_d1       
        theta = (-1)*(s0*n_pdf_d1*sigma*math.exp(-rf*T))/(2*math.sqrt(T)) - rf*s0*n_d1*math.exp(-rf*T) + r*k*math.exp(-r*T)*n_d2
        rho = (-1)*k*T*math.exp(-r*T)*n_d2
    else:
        pass
    
    # put each Greek related attribute in a dictionary
    greeks_dict['Asset Spot Price'] = s0 
    greeks_dict['Strike'] = k 
    greeks_dict['Implied Volatility'] = sigma
    greeks_dict['Time to Maturity'] = T 
    greeks_dict['Delta'] = delta
    greeks_dict['Gamma'] = gamma
    greeks_dict['Theta'] = theta
    greeks_dict['Vega'] = vega
    greeks_dict['Rho'] = rho
    
    return greeks_dict
    
#===================================================================================================================
# testing

try:
    result = greeks(s0=0.8, k=0.81, r=0.08, rf=0.05, sigma=0.15, T=7/12, c=True, p=False)
    print(result)
except:
    print("Error found.")
finally:
    print("Testing finished.")

#===================================================================================================================



