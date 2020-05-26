# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:33:29 2020

@author: Yu Zhu
"""
#===================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
from Black_Scholes import blackscholes

#===================================================================================================================
# read option related data
amzn_3mc = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_3_month_call.csv")
amzn_6mc = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_6_month_call.csv")
amzn_9mc = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_9_month_call.csv")
amzn_3mp = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_3_month_put.csv")
amzn_6mp = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_6_month_put.csv")
amzn_9mp = pd.read_csv(r"D:\Academic\FE 621 Computational Methods in Finance\HW1\AMZN_9_month_put.csv")


#===================================================================================================================
# Application of Black-Scholes model - calcualte implied volatility (volatility)
# implement the Bisection method to find the root of arbitrary functions
# f(x) = Cbs(s0, k, T, r, x) - (B + A)/2

# the bisection method
def bisection(f, left=0, right=1, tolerance= 10**(-6)):
    
    root = None
    if f(left) == 0:
        root = left
    elif f(right) == 0:
        root = right
    else:
        while right - left > tolerance:
            mid = f((left + right)*0.5)

            # check the symbol of the product
            if mid == 0:        
                root = (left + right)*0.5
                #print("it's right in the middle")
                break
            elif f(left)*mid < 0:
                #print("go left a little bit")
                right = (left + right)*0.5
            elif f(right)*mid < 0:
                #print("go right a little bit")
                left = (left + right)*0.5
            else:
                print("There may be multiple roots within this interval.")
                break
    
        root = (left + right)*0.5
    
    return root

# the secant method
def secant(f, x0, x1, iteration, tolerance=10**(-6)):
    
    if x1 - x0 > tolerance:
        for i in range(iteration):
            try:
                x2 = x1 - f(x1)*(x1 - x0)/float(f(x1) - f(x0))*1.0
                x0, x1 = x1, x2
            except:
                break
    root = x2
    return root
    
def voltable_secant(source, tau, call, put):
    
    def targetfun(sigma):
        result = blackscholes(sigma, s0=948.23, tau=tau, k=source['Strike'][i], r=0.0091, c=call, p=put) - (source['Bid'][i] + source['Ask'][i])* 0.5
        return result
    
    k_list = []
    bid_list = []
    ask_list = []
    vol_list = []
    vol_dic = {}
    for i in range(len(source.index)):
        k_list.append(source['Strike'][i])
        bid_list.append(source['Bid'][i])
        ask_list.append(source['Ask'][i])
        vol_list.append(secant(targetfun, 0, 1, 5)) # number of iteration need to be considered carefully
    
    vol_dic['Strike'] = k_list
    vol_dic['Bid'] = bid_list
    vol_dic['Ask'] = ask_list    
    vol_dic['Implied Volatility'] = vol_list
    vol_dic['Time to Maturity'] = tau/252
    
    vol_table = pd.DataFrame(vol_dic)
    return vol_table 

vol_dic1_c_secant = voltable_secant(source = amzn_3mc, tau=47, call=True, put=False)


#===================================================================================================================
#option implied volatility calculation

def voltable(source, tau, call, put):
    
    def targetfun(sigma):
        result = blackscholes(sigma, s0=948.23, tau=tau, k=source['Strike'][i], r=0.0091, c=call, p=put) - (source['Bid'][i] + source['Ask'][i])* 0.5
        return result
    
    k_list = []
    bid_list = []
    ask_list = []
    vol_list = []
    vol_dic = {}
    for i in range(len(source.index)):
        k_list.append(source['Strike'][i])
        bid_list.append(source['Bid'][i])
        ask_list.append(source['Ask'][i])
        vol_list.append(bisection(targetfun))
    
    vol_dic['Strike'] = k_list
    vol_dic['Bid'] = bid_list
    vol_dic['Ask'] = ask_list    
    vol_dic['Implied Volatility'] = vol_list
    vol_dic['Time to Maturity'] = tau/252
    
    vol_table = pd.DataFrame(vol_dic)
    return vol_table 


#===================================================================================================================
# calculate imlied volatilities    
vol_dic1_c = voltable(source = amzn_3mc, tau=47, call=True, put=False)
vol_dic2_c = voltable(source = amzn_6mc, tau=81, call=True, put=False)
vol_dic3_c = voltable(source = amzn_9mc, tau=109, call=True, put=False)
vol_dic1_p = voltable(source = amzn_3mp, tau=47, call=False, put=True)
vol_dic2_p = voltable(source = amzn_6mp, tau=81, call=False, put=True)
vol_dic3_p = voltable(source = amzn_9mp, tau=109, call=False, put=True)

# volatility smile or smirk or frown
def plotting2d(source1, source2, source3, source4, source5, source6):
    # plotting
    plt.style.use('bmh')
    fig = plt.figure()
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.linestyle'] = '--'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (18, 11))
    ax1.plot(source1['Strike'], source1['Implied Volatility'], color="grey" , label='call 1 month expire', marker=".", markerfacecolor='blue', markersize=24)
    ax1.plot(source2['Strike'], source2['Implied Volatility'], color="grey" , label='call 2 months expire', marker=".", markerfacecolor='green', markersize=24)
    ax1.plot(source3['Strike'], source3['Implied Volatility'], color="grey" , label='call 3 months expire', marker=".", markerfacecolor='pink', markersize=24)
    ax2.plot(source4['Strike'], source4['Implied Volatility'], color="grey" , label='put 1 month expire', marker="^", markerfacecolor='blue', markersize=18)
    ax2.plot(source5['Strike'], source5['Implied Volatility'], color="grey" , label='put 2 months expire', marker="^", markerfacecolor='green', markersize=18)
    ax2.plot(source6['Strike'], source6['Implied Volatility'], color="grey" , label='put 3 months expire', marker="^", markerfacecolor='pink', markersize=18)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals = None, symbol='%'))
    ax1.set_xlabel("Strike Price")
    ax1.set_ylabel('Implied \nVolatility', rotation=0, fontsize=20, labelpad=45)
    ax1.set_title(f"AMZN Implied Volatility call (1, 2, 3) month(s) expire")
    ax1.legend(prop=dict(size=18))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals = None, symbol='%'))
    ax2.set_xlabel("Strike Price")
    ax2.set_ylabel('Implied \nVolatility', rotation=0, fontsize=20, labelpad=45)
    ax2.set_title(f"AMZN Implied Volatility put (1, 2, 3) month(s) expire")
    ax2.legend(prop=dict(size=18))

# try plotting function
plotting2d(vol_dic1_c, vol_dic2_c, vol_dic3_c, vol_dic1_p, vol_dic2_p, vol_dic3_p)


'''
# another method
plt.figure(figsize=(10,8))
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = 'dotted'
plt.plot(vol_dic['Strike'], vol_dic['Implied Volatility'], label='Volatility')
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("AMZN Implied Volatility 1 month expire")
plt.legend()
'''

# The volatility surface is a functionos strike, and time-to-maturity and implied
# volatility
#time to np.linspace(47/252, 109/252, num=20)
completetable = vol_dic1_c.append(vol_dic2_c.append(vol_dic3_c))
#completetable = vol_dic1_p.append(vol_dic2_p.append(vol_dic3_p))
x = completetable['Strike'] 
y = completetable['Time to Maturity']
z = completetable['Implied Volatility']

def plot3D(x, y, z, fig, ax):
    ax.plot(x, y, z, 'o', color = 'pink')
    ax.set_xlabel("Strike", rotation=0, fontsize=15, labelpad=30)
    ax.set_ylabel("Time-to-Maturity (Years)", rotation=0, fontsize=15, labelpad=60)
    ax.set_zlabel("Implied \nVolatility", rotation=0, fontsize=15, labelpad=20)
    
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
    fig = plt.figure(figsize=(15,10))
    ax = Axes3D(fig, azim = -65, elev = 10)
    mesh_plot2(x, y, z, fig, ax)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    plot3D(x, y, z, fig, ax)
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals = None, symbol='%'))
    ax.set_title("AMZN Implied Volatility Surface", pad=0.5)
    plt.show()    

# draw the plot
combine_plots(x, y, z)

