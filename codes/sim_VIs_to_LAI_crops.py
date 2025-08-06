######################################################################################
# Simulate LAI in relation to vegetation indices (VIs) for crops
# Input files: Four VIs (MTVI1, NDVI, OSAVI, & RDVI) data in csv format
# output files: simulated LAI
# Coded by J Ko and Tim Ng
# Last update: July 01, 2025
######################################################################################
import os
import math
import csv
import pickle
import tensorflow as tf
import keras
from keras import models
from keras import layers
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

#** Read data *************
def read_data(path_n_fn):
    data = np.genfromtxt(path_n_fn, dtype='float')
    return data

#** DNN reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#************************************************************
def DNN_reg(DNN_FN):
    with open(DNN_FN, 'rb') as file:
            DNN_model = keras.models.load_model(DNN_FN, compile=False)   
            
    return DNN_model

#** ML(Extra Trees ) reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#************************************************************
def ML_reg(pkl_FN):
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
    
    """ print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    print('after', SLAI) """
    
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

    """print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    #print('after', SLAI) """

    return SLAI

#** log-log reg model to convert VIs to LAI ********
# function to simulate LAI based on VIs
#***************************************************
def log_log_reg(wobs_data, data,flag):    
    (coef,Sigma) = get_regr_coef_VI_LAI(wobs_data)

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
    
    """print('before', SLAI)
    for i, n in enumerate(SLAI):
           if n > flag:
                SLAI[i] = flag
    #print('after', SLAI) """
    
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

def main(DNN_FN, pkl_FN, pkl_seq_FN, reg_opt, plot_opt, file_opt, flag, para_FN, wobs_FN2, data_FN,output_FN):

    """ Simulate LAI in relation to vegetation indices (VIs) for crops
    Input files: Four VIs (MTVI1, NDVI, OSAVI, & RDVI) """

    df_whole_VIs_n_LAI = pd.read_csv(wobs_FN2)[["DOY","LAI","MTVI1", "NDVI", "OSAVI", "RDVI"]].to_numpy()
    df_in = pd.read_csv(data_FN)[["DOY","MTVI1", "NDVI", "OSAVI", "RDVI"]].to_numpy()

    if(reg_opt == 0 or reg_opt == 6): # DNN model
        DNN_model = DNN_reg(DNN_FN)

        # Produce a scale of the data
        scaler = preprocessing.StandardScaler()

        # change data
        data_standardized = scaler.fit_transform(df_in[:])

        Ypredict = DNN_model.predict(data_standardized).flatten()
        SLAI = Ypredict

        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag        

        if reg_opt == 0:
            print('before', Ypredict)
            print('after', SLAI)

        if (reg_opt == 6):        
            SLAI_DNN = SLAI

    if(reg_opt == 1 or reg_opt == 6 or reg_opt == 7):  # Regular ML regression
        pickle_model = ML_reg(pkl_FN)

        log10_log_DOY2 = np.log10(np.log(df_in[:,0]))
        data = df_in[:,1:]
        data = np.c_[log10_log_DOY2, data]

        Ypredict = pickle_model.predict(data).flatten()
        SLAI = Ypredict

        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag

        if reg_opt == 1:
            print('before',Ypredict)
            print('after', SLAI)  
    
        if (reg_opt == 6 or reg_opt == 7): SLAI_ML = SLAI

    if(reg_opt == 2 or reg_opt == 6 or reg_opt == 7):   ###-- Sequential ML regression -- 
        pickle_model = ML_reg(pkl_FN)

        # Create a sequential variable replacing DOY (1, 2, ..., n)
        df = pd.read_csv(data_FN)
        df = df.reset_index(drop=True)
        seq = np.arange(1, len(df) + 1)

        # Transform the sequential variable: log10(log(seq))
        with np.errstate(divide='ignore'):
            log_seq = np.log(seq)
        # Avoid -inf for seq=1 by shifting index or starting from 2 if needed
        # Here, seq starts at 1 => log(1)=0 => log10(0) is -inf. To handle, add a small epsilon:
        eps = 1e-6
        log_seq_safe = np.log(seq + eps)
        log10_log_seq = np.log10(log_seq_safe)

        # Shrink factor α: smaller α → sequence matters less
        alpha = 1
        log10_log_seq_shrunk = log10_log_seq * alpha

        # Feature matrix: shrunk sequential variable + four vegetation indices
        data = np.column_stack([
            log10_log_seq_shrunk,
            df[['MTVI1', 'NDVI', 'OSAVI', 'RDVI']].values]) 
        
        Ypredict = pickle_model.predict(data).flatten()
        SLAI = Ypredict

        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag

        if reg_opt == 2:
            print('before', Ypredict)
            print('after', SLAI)
    
        if (reg_opt == 6 or reg_opt == 7): SLAI_ML = SLAI

    # NDVI-based (3) or our VIs-based (4) regression model
    if(reg_opt == 3 or reg_opt == 6 or reg_opt == 7):

        exponential_regression(df_whole_VIs_n_LAI, para_FN)
        
        # read empirical regression parameters
        P = open(para_FN,'r')
        ipara = yaml.load(P, Loader=yaml.FullLoader )
        para_a2 = ipara['para_a2'] # [para_a2*exp(NDVI*para_b2)]
        para_b2 = ipara['para_b2'] # [para_a2*exp(NDVI*para_b2)]
        
        SLAI0 = empirical_NDVI(df_in, para_a2,para_b2, flag)
        SLAI = SLAI0

        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag

        if reg_opt == 3:
            print('before', SLAI0)
            print('after', SLAI)
    
        if(reg_opt == 6 or reg_opt == 7): SLAI_NDVI = SLAI

    if(reg_opt == 4 or reg_opt == 6 or reg_opt == 7):

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
        SLAI0 = empirical_VIs(df_in, para_a1,para_b1,para_a2,para_b2,
                             para_a3,para_b3,para_a4,para_b4, flag)
        SLAI = SLAI0
        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag

        if reg_opt == 4:
            print('before', SLAI0)
            print('after', SLAI)
       
        if(reg_opt == 6 or reg_opt == 7): SLAI_VIs = SLAI
        
    if(reg_opt == 5 or reg_opt == 6 or reg_opt == 7):  
        (SLAI0) = log_log_reg(df_whole_VIs_n_LAI, df_in,flag)

        SLAI = SLAI0
        for i, n in enumerate(SLAI):
            if n > flag:
                SLAI[i] = flag

        if reg_opt == 5:
            print('before', SLAI0)
            print('after', SLAI)
        
        if(reg_opt == 6 or reg_opt == 7): SLAI_log_log = SLAI       
    
    if (reg_opt == 6): 
        sum_SLAI = [sum(values) for values in zip(SLAI_DNN, SLAI_ML, SLAI_VIs, SLAI_log_log)]
        divisor = 4
        SLAI = np.divide(sum_SLAI, divisor)

    if (reg_opt == 7): 
        sum_SLAI = [sum(values) for values in zip(SLAI_ML, SLAI_VIs, SLAI_log_log)]
        divisor = 3
        SLAI = np.divide(sum_SLAI, divisor)

    np.set_printoptions(precision=2)
       
    if(plot_opt == 1): plot_LAI(df_in[:,0], SLAI)
    
    # Save the arrays in a text file
    if(file_opt == 1):
        file = open(output_FN, "w+")
        SLAI = SLAI.flatten()
        for i in range(df_in.shape[0]):
            line = "%i %5.2f\n" % (df_in[i,0], SLAI[i])
            file.write(line)
        file.close()

if __name__ == "__main__":
    # parse arguments or set defaults, then call main(...)
    #main(DNN_FN, pkl_FN, pkl_seq_FN, reg_opt, plot_opt, file_opt, flag, para_FN, wobs_FN2, data_FN,output_FN)
    pass
