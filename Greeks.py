# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:52:49 2020

@author: Yu Zhu

There are two approaches to calculate Greeks: analytical function and finite difference 
"""
#===================================================================================================================
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import matplotlib.ticker as mtick

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
'''
try:
    result = greeks(s0=948.23, k=900, r=0.0091, rf=0, sigma=0.165751, T=0.186508, c=True, p=False)
    print(result)
except:
    print("Error found.")
finally:
    print("Testing finished.")
'''
#===================================================================================================================
# the method to put all the greeks into a table given unique implied volatility and strike price
def greekTable(source, s0=948.23, r=0.0091, rf=0, tau=47, c=True, p=False):
    
    T = tau/252
    greek_list = []
    for i in range(len(source.index)):
        result = greeks(s0=s0, k=source['Strike'][i], r=r, rf=rf, sigma=source['Implied Volatility'][i], T=T, c=c, p=p)
        greek_list.append(result)

    greek_table = pd.DataFrame(greek_list)
    
    return greek_table

#===================================================================================================================
# testing
'''
try:
    result = greekTable(source=vol_dic1_c, s0=948.23, r=0.0091, rf=0, tau=47, c=True, p=False)
    print(result)
except:
    print("Error found.")
finally:
    print("Testing finished.")
'''

#===================================================================================================================

greek_table_c_1 = greekTable(source=vol_dic1_c, s0=948.23, r=0.0091, rf=0, tau=47, c=True, p=False)
greek_table_c_2 = greekTable(source=vol_dic2_c, s0=948.23, r=0.0091, rf=0, tau=81, c=True, p=False)
greek_table_c_3 = greekTable(source=vol_dic3_c, s0=948.23, r=0.0091, rf=0, tau=109, c=True, p=False)
greek_table_p_1 = greekTable(source=vol_dic1_p, s0=948.23, r=0.0091, rf=0, tau=47, c=False, p=True)
greek_table_p_2 = greekTable(source=vol_dic2_p, s0=948.23, r=0.0091, rf=0, tau=81, c=False, p=True)
greek_table_p_3 = greekTable(source=vol_dic3_p, s0=948.23, r=0.0091, rf=0, tau=109, c=False, p=True)

# call delta
plt.figure(figsize=(10,6))
plt.style.use('bmh')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '-'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Delta'], label='Call: 1 month')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Delta'], label='Call: 2 month')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Delta'], label='Call: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Delta")
plt.title("AMZN Call Delta Variation")
plt.legend()

# put delta
plt.figure(figsize=(10,6))
plt.style.use('bmh')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '-'
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Delta'], label='Put: 1 month')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Delta'], label='Put: 2 month')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Delta'], label='Put: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Delta")
plt.title("AMZN Put Delta Variation")
plt.legend()

# call theta
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Theta'], label='Call: 1 month')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Theta'], label='Call: 2 month')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Theta'], label='Call: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Theta")
plt.title("AMZN Call Theta Variation")
plt.legend()

# put theta
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Theta'], label='Put: 1 month')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Theta'], label='Put: 2 month')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Theta'], label='Put: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Theta")
plt.title("AMZN Put Theta Variation")
plt.legend()


# call gamma
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Gamma'], label='Call: 1 month')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Gamma'], label='Call: 2 month')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Gamma'], label='Call: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Gamma")
plt.title("AMZN Call Gamma Variation")
plt.legend()

# put gamma
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Gamma'], label='Put: 1 month')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Gamma'], label='Put: 2 month')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Gamma'], label='Put: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Gamma")
plt.title("AMZN Put Gamma Variation")
plt.legend()


# call vega
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Vega'], label='Call: 1 month')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Vega'], label='Call: 2 month')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Vega'], label='Call: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Vega")
plt.title("AMZN Call Vega Variation")
plt.legend()


# put vega
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Vega'], label='Put: 1 month')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Vega'], label='Put: 2 month')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Vega'], label='Put: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Vega")
plt.title("AMZN Put Vega Variation")
plt.legend()


# call rho
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Rho'], label='Call: 1 month')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Rho'], label='Call: 2 month')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Rho'], label='Call: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Rho")
plt.title("AMZN Call Rho Variation")
plt.legend()

# put rho
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Rho'], label='Put: 1 month')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Rho'], label='Put: 2 month')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Rho'], label='Put: 3 month')
plt.xlabel("Strike Price")
plt.ylabel("Rho")
plt.title("AMZN Put Rho Variation")
plt.legend()

## call gamma vs. delta
plt.figure(figsize=(12,7))
plt.style.use('bmh')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '-'
plt.plot(greek_table_c_1['Strike'], greek_table_c_1['Delta'], label='Call: 1 month Delta')
plt.plot(greek_table_c_2['Strike'], greek_table_c_2['Delta'], label='Call: 2 month Delta')
plt.plot(greek_table_c_3['Strike'], greek_table_c_3['Delta'], label='Call: 3 month Delta')
plt.plot(greek_table_p_1['Strike'], greek_table_p_1['Delta'], label='Put: 1 month Delta')
plt.plot(greek_table_p_2['Strike'], greek_table_p_2['Delta'], label='Put: 2 month Delta')
plt.plot(greek_table_p_3['Strike'], greek_table_p_3['Delta'], label='Put: 3 month Delta')
plt.xlabel("Strike Price")
plt.ylabel("Delta")
plt.title("AMZN Delta Variation")
plt.legend()


completetable = greek_table_c_1.append(greek_table_c_2.append(greek_table_c_3))
x = completetable['Strike'] 
y = completetable['Time to Maturity']
z = completetable['Vega']

def plot3D(x, y, z, fig, ax):
    ax.plot(x, y, z, 'o', color = 'pink')
    ax.set_xlabel("Stock Price", rotation=0, fontsize=15, labelpad=30)
    ax.set_ylabel("Time-to-Maturity (Years)", rotation=0, fontsize=15, labelpad=60)
    ax.set_zlabel("Vega", rotation=0, fontsize=15, labelpad=25)
    
def make_surf(x, y, z):
    xx, yy = np.meshgrid(np.linspace(min(x), max(x), 230), np.linspace(min(y), max(y), 230))
    zz = griddata(np.array([x, y]).T, np.array(z), (xx, yy), method = 'linear')
    return xx, yy, zz

def mesh_plot2(x, y, z, fig, ax):
    xx, yy, zz = make_surf(x, y, z)
    ax.plot_surface(xx, yy, zz, cmap=cm.rainbow)
    ax.contour(xx, yy, zz)
    
def combine_plots(x, y, z):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(13,10))
    ax = Axes3D(fig, azim = -50, elev = 30)
    mesh_plot2(x, y, z, fig, ax)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    plot3D(x, y, z, fig, ax)
    #ax.zaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals = None, symbol='%'))
    ax.set_title("AMZN Vega Surface", pad=1)
    plt.show()    

# draw the plot
combine_plots(x, y, z)
