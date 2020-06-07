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
def blackscholes(sigma, s0, div, tau, k, r, c=False, p=False):
    '''
    s0: current stock price s0
    sigma: volatility
    tau: (T - t)/252 such as from 2017/6/17 to 2017/5/1 there are 47 days, annualized 47/252 year, exclude the last day 
    k: strike price
    r: short-term interest rate
    '''
    delta = tau/252.0
    d1 = (np.log(s0/k*1.0) + (r - div + 0.5*sigma**2)*delta)/(sigma*math.sqrt(delta*1.0)*1.0) 
    d2 = d1 - sigma*math.sqrt(delta)
    
    call_price = s0*math.exp(-div*delta)*float(norm.cdf(d1)) - k*math.exp(-r*delta)*float(norm.cdf(d2)) 
    put_price = k*math.exp(-r*delta)*float(norm.cdf(-d2)) - s0*math.exp(-div*delta)*float(norm.cdf(-d1))
    
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
