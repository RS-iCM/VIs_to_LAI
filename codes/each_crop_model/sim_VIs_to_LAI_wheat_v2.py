######################################################################################
# Simulate LAI in relation to vegetation indices (VIs) for wheat
# Input files: Four VIs (MTVI1, NDVI, OSAVI, & RDVI)
# output files: simulated LAI
# Coded by J Ko and Tim Ng
# Last update: June 30, 2025
######################################################################################
import os
import math
import csv
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn import preprocessing
import numpy as np
from numpy import diag
from numpy.linalg import inv
import pandas as pd
import yaml
import string
import matplotlib.pyplot as plt
from math import exp, pi, cos, acos, sin, tan, log
from scipy.optimize import minimize
from scipy.optimize import curve_fit

#** Set the current working directory ********
path = os.path.abspath(os.getcwd())+'//'

#** Read data *************
def read_data(path_n_fn):
    data = np.genfromtxt(path_n_fn, dtype='float')
    return data

#** DNN reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#************************************************************
def DNN_reg(path):
    DNN_FN = path+'models/wheat_NN.h5'
    with open(DNN_FN, 'rb') as file:
            DNN_model = tf.keras.models.load_model(DNN_FN, compile=False)   
            
    return DNN_model

#** ML(Extra Trees ) reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#************************************************************
def ML_reg(path):   
    pkl_FN = path+'models/pickle_gradient_boost_Wheat.pkl'
    with open(pkl_FN, 'rb') as file:
            pickle_model = pickle.load(file)          
            
    return pickle_model
 
#** Empirical reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#*****************************************************


# Define exponential regression function
def exp_model(x, a, b):
    return a * np.exp(b * x)

def exponential_regression(df, output_path):
    """
    Perform exponential regression on the given DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame containing vegetation indices and LAI.
    
    Returns:
    dict: Dictionary containing regression parameters.
    """
    # Variables of interest
    vi_names = ["MTVI1", "NDVI", "OSAVI", "RDVI"]
    LAI = df[:,1]

    # Fit models and store parameters
    params = {}
    for i, vi in enumerate(vi_names, 1):
        x = df[:,i+1]
        popt, _ = curve_fit(exp_model, x, LAI, maxfev=10000)
        params[f"para_a{i}"] = popt[0]
        params[f"para_b{i}"] = popt[1]

    # Prepare content for output text file
    output_lines = ["# Regression parameters from VIs and LAI"]
    for i, vi in enumerate(vi_names, 1):
        a_key = f"para_a{i}"
        b_key = f"para_b{i}"
        lines = [
            f"{a_key}: {params[a_key]:.2f}  # [para_a{i}*exp({vi}*para_b{i})]",
            f"{b_key}: {params[b_key]:.2f}  # [para_a{i}*exp({vi}*para_b{i})]"
        ]
        output_lines.extend(lines)

    # Save to the exponential regression model to a text file
    #output_path = "/mnt/data/LAI_VI_exponential_parameters.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

def empirical_NDVI(df_in, para_a1,para_b1, flag):
    id_NDVI = 2
    data = df_in[:,id_NDVI]
    SLAI = para_a1*np.exp(para_b1*data)
    
    SLAI = SLAI.flatten()
    
    print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    print('after', SLAI) 
    
    return SLAI

# --- Regress log(VI) against log(LAI) ---
def get_regr_coef_VI_LAI(wobs_data):
    n_wobs = wobs_data.shape[0]
    n_VI = wobs_data.shape[1]-2
    X = np.empty((n_wobs,3))
    X[:,0] = 1
    X[:,1] = np.log(wobs_data[:,1]) # LAI
    X[:,2] = wobs_data[:,0]         # DOY
    Y = np.empty((n_wobs,n_VI))
    Y[:,0:n_VI] = np.log(wobs_data[:,2:])
    coef = np.zeros((n_VI,3))    
    A = np.matmul(X[:,0:2].conj().transpose(), X[:,0:2])
    B = np.matmul(X[:,0:2].conj().transpose(), Y)
    coef[:,0:2] = np.matmul(B.conj().transpose(), inv(A))
    residuals = Y - np.matmul(X, coef.conj().transpose())
    Sigma = np.matmul(residuals.conj().transpose(), residuals) / (n_wobs-2)
    return coef,Sigma

#--- Sub-procedure to retrieve VI from the VI data---
def get_LAI_from_VI(data, coef,Sigma,flag):
    nObs = data.shape[0]
    X = np.empty((nObs,2))
    X[:,0] = 1
    X[:,1] = data[:,0]
    Sinv = inv(Sigma)
    res = np.log(data[:,1:])-np.matmul(X, coef[:,[0,2]].conj().transpose())
    A = np.matmul(np.matmul(res, Sinv), coef[:,1])
    B = np.matmul(coef[:,1].conj().transpose(), Sinv)
    C = np.matmul(B, coef[:,1])
    SLAI = np.exp(A/C) #np.matmul(C, A)
    print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    print('after', SLAI) 
    return SLAI

#**** This module needs to be improved *************
#** log-log reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#***************************************************
def log_log_reg(wobs_data, data,flag):    
    (coef,Sigma) = get_regr_coef_VI_LAI(wobs_data)
    print('coef', coef)   # TEST print
    print('Sigma', Sigma) # TEST print
    (SLAI) = get_LAI_from_VI(data, coef,Sigma,flag)
    return SLAI

#** Empirical reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#*****************************************************
def empirical_VIs(df_in, para_a1,para_b1,para_a2,para_b2,
                  para_a3,para_b3,para_a4,para_b4, flag):
    LAI_MTVI = para_a1*np.exp(para_b1*df_in[:,1])
    LAI_NDVI = para_a2*np.exp(para_b2*df_in[:,2])
    LAI_OSAVI = para_a3*np.exp(para_b3*df_in[:,3])
    LAI_RDVI = para_a4*np.exp(para_b4*df_in[:,4])
    SLAI = (LAI_NDVI+LAI_RDVI+LAI_MTVI+LAI_OSAVI)/4.0
    
    SLAI = SLAI.flatten()
    
    print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    print('after', SLAI) 
    
    return SLAI

#** Plot a time-series graph using LAI data ********
#***************************************************
def plot_LAI(DOYs, SLAI):   
    max_y = np.nanmax(SLAI)
    min_x = np.nanmin(DOYs)
    max_x = np.nanmax(DOYs) 
    plt.plot(DOYs, SLAI, 'go')
    plt.xlabel('Day of year')
    plt.ylabel('LAI (m$^2$'+' m$^-$'+'$^2$)')  
    plt.axis([min_x-10, max_x+10, 0, max_y+1]) 
    plt.show()

def main(reg_opt, plot_opt, file_opt, flag, para_FN, df_whole_VIs_n_LAI, df_in,output_FN):

    """ Simulate LAI in relation to vegetation indices (VIs) for wheat
    Input files: Four VIs (MTVI1, NDVI, OSAVI, & RDVI) """

    if(reg_opt == 0):
        DNN_model = DNN_reg(path)
        #log10_log_DOY2 = np.log10(np.log(df_in['DOY'].to_numpy()))
        #data = df_in[['MTVI1','NDVI','OSAVI','RDVI']].to_numpy()
        log10_log_DOY2 = np.log10(np.log(df_in[:,0]))
        data = df_in[:,1:]
        data = np.c_[log10_log_DOY2, data]
        Ypredict = DNN_model.predict(data).flatten()    
        SLAI = Ypredict

    if(reg_opt == 1 or reg_opt == 5):  # ML (Extra Trees) model
        pickle_model = ML_reg(path)
        log10_log_DOY2 = np.log10(np.log(df_in[:,0]))
        data = df_in[:,1:]
        data = np.c_[log10_log_DOY2, data]     
        Ypredict = pickle_model.predict(data).flatten()
        #SLAI = 10**Ypredict
        SLAI = Ypredict
        
        print('before', SLAI)
        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag
        print('after', SLAI)  
    
        if (reg_opt == 5): SLAI_ML = SLAI

    # NDVI-based (2) or our VIs-based (3) regression model
    if(reg_opt == 2 or reg_opt == 5):

        exponential_regression(df_whole_VIs_n_LAI, para_FN)
        
        # read empirical regression parameters
        P = open(para_FN,'r')
        ipara = yaml.load(P, Loader=yaml.FullLoader )
        para_a2 = ipara['para_a2'] # [para_a2*exp(NDVI*para_b2)]
        para_b2 = ipara['para_b2'] # [para_a2*exp(NDVI*para_b2)]
        
        SLAI = empirical_NDVI(df_in, para_a2,para_b2, flag)
        if(reg_opt == 5): SLAI_NDVI = SLAI

    if(reg_opt == 3 or reg_opt == 5):

        exponential_regression(df_whole_VIs_n_LAI, para_FN)
        
        # read empirical regression parameters
        P = open(para_FN,'r')
        ipara = yaml.load(P, Loader=yaml.FullLoader )
        para_a1 = ipara['para_a1'] # [para_a1*exp(NDVI*para_b1)]
        para_b1 = ipara['para_b1'] # [para_a1*exp(NDVI*para_b1)]
        para_a2 = ipara['para_a2'] # [para_a2*exp(RDVI*para_b2)]
        para_b2 = ipara['para_b2'] # [para_a2*exp(RDVI*para_b2)]
        para_a3 = ipara['para_a3'] # [para_a3*exp(MTVI*para_b3)]
        para_b3 = ipara['para_b3'] # [para_a3*exp(MTVI*para_b3)]
        para_a4 = ipara['para_a4'] # [para_a4*exp(OSAVI*para_b4)]
        para_b4 = ipara['para_b4'] # [para_a4*exp(OSAVI*para_b4)]
        SLAI = empirical_VIs(df_in, para_a1,para_b1,para_a2,para_b2,
                             para_a3,para_b3,para_a4,para_b4, flag)
        if(reg_opt == 5): SLAI_VIs = SLAI
    
   
    if (reg_opt == 5): 
        sum_SLAI = [sum(values) for values in zip(SLAI_ML, SLAI_NDVI, SLAI_VIs)]
        divisor = 3
        SLAI = np.divide(sum_SLAI, divisor)
        
    #*** Need to be improved ****
    if(reg_opt == 6):  
        (SLAI) = log_log_reg(df_whole_VIs_n_LAI, df_in,flag)        
    
    np.set_printoptions(precision=2)
    print(SLAI)
       
    if(plot_opt == 1): plot_LAI(df_in[:,0], SLAI)
    
    # Save the arrays in a text file
    if(file_opt == 1):
        file = open(output_FN, "w+")
        SLAI = SLAI.flatten()
        for i in range(df_in.shape[0]):
            line = "%i %5.2f\n" % (df_in[i,0], SLAI[i])
            file.write(line)
        file.close()

#main()
