# -*- coding: utf-8 -*-
"""
5/23/2020

Yu Zhu
"""
#===================================================================================================================
import numpy as np
import math
from scipy.stats import norm

#===================================================================================================================
# implement the Black-Scholes formula
def blackscholes(s0, sigma, T, t, k, r, c=False, p=False):
    '''
    s0: current stock price s0
    sigma: volatility
    T: end
    t: start
    k: strike price
    r: short-term interest rate
    '''
    d1 = (np.log(s0/k) + (r - sigma + 0.5*sigma**2)*(T - t))/sigma*math.sqrt((T - t)*1.0) 
    d2 = d1 - sigma*math.sqrt(T - t)
    
    call_price = s0*math.exp(-sigma*(T - t))*norm.cdf(d1) - k*math.exp(-r*(T - t))*norm.cdf(d2) 
    put_price = k*math.exp(-r*(T - t))*norm.cdf(-d2) - s0*math.exp(-sigma*(T - t))*norm.cdf(-d1)
    
    price = None
    if c and p:
        print("Option type specification identical.")    
    elif c:
        price = call_price
    elif p:
        price = put_price
    else:
        print("No type of option was specified, no result provided.")
        pass
    
    return price

#===================================================================================================================
# testing
try:
    result = blackscholes(s0=10, sigma=0.07, T=1, t=0.5, k=5, r=0.006, c=True, p=False)
    print(result)
except:
    print("Error found. ")
finally:
    print("Testing finished.")
