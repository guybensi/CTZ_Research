#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:05:35 2026

@author: eeytan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:35:42 2026

@author: eeytan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:55:35 2026

@author: eeytan
"""


import datetime
import math
import os
import sys
import time

import pandas as pd
import netCDF4 as nc
import numpy as np
import matplotlib as mpl
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
import matplotlib.colors as colors

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large','figure.figsize': (5, 5),'axes.labelsize': 'x-large','axes.titlesize':'x-large','xtick.labelsize':'x-large','ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import matplotlib
matplotlib.use('Agg')   # <-- must be before importing pyplot
import matplotlib.pyplot as plt

font = {'weight' : 'bold','size'   : 22}
plt.rc('font', **font)

import pickle

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mCTHth=300000   # limit the mean height of clouds
stdCTHth=100000   # limit standard deviation of CTH

# transform_method:
#   0 -> no transform applied to the data (plain MSE loss)
#   1 -> log1p transform + weighted loss (upweights large values)
#   2 -> Yeo-Johnson transform, no weighting (distribution normalised)
transform_method = 1

costf=0   #  cost function: 0-MAE, 1-MSE


outlierL=10
outlierH=1000

train_label_low  = 400 # limit the training to scenes with a minimal all-sky flux.
train_label_high = outlierH   

base_featers_list=['SZA','VZA','RAA','0.47', '0.66', '0.86', '1.375', '1.24','2.13','3.75', 
              '3.96', '4.05', '4.46', '4.5', '6.7', '7.3', '8.5', '9.7', '11.0', '12.0', '13.33', '13.63', '13.93', '14.23']
base_cross_featers_list=['SWIRsRatioA','midIRsst','WVz','Ozone','CO2','SWIRsRatioB','SO2']

visls=[0.47,0.55,0.66,0.86]
nirls=[0.905,0.935,0.940]
swirls=[1.24,1.64,2.13,1.375]
thermalls=np.array([3.75,3.961,3.96,4.05,4.46,4.5,6.7,7.3,8.5,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
solarls=np.concatenate((visls,nirls,swirls),axis=0)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Flags and parameters &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

wavelns_list0=['0.47','0.66','0.86','1.375','2.13','3.96','4.05','4.5', '6.7', '7.3', '9.7','11.0','12.0']

moments_list=['m','s','sk','k']
# moments_list=['m']#,'s','sk']



savenm='Pac_allMnth_allCld_400Nup'

fname='mrchsepjunedec_2014-2015_Pacific.npz'

datapath='projects/F1km/ALLSKY_newDat/'
fname=datapath+fname


txtdy=40
fontsz=25


SolarNorm='0.47'  # wavelengths to use for normalizing the other wavelengths (according to the part of the spectrum; i.e., solar,midIR and thermal)
MixedNorm='4.05'
ThermalNorm='11.0'

featers_list=['SZA','VZA','RAA']
for i in range(0,len(moments_list)):   
    featers_list=featers_list+[moments_list[i]+tmp for tmp in wavelns_list0]

# The following variables are the hyperparameters.
random_seed = 42  # None   # controls shuffeling. None will give a random seed.
learning_rate = 0.001
Nepochs = 1000
batch_size = 10000
internal_test=True
trainprec=0.7
validation_split = 0.1
delta_rmse_threshold=0.00005  # when to cut the training
patience=3 # cut the training only if the rmse remains below the threshold a few times.

linearfit_split = 0
activationf='relu'
Nfeaters1=round(len(featers_list)/2)
Nfeaters2=round(len(featers_list)/4)
Nfeaters3=round(len(featers_list)/8)
Nfeaters4=5

Nlyers=np.sum(np.array([Nfeaters1,Nfeaters2,Nfeaters3,Nfeaters4])>0)

nmextension=savenm+'_m'+str(len(moments_list))+'_Ly'+str(Nlyers)+'_Trns'+str(transform_method)+'_Cost'+str(costf)
savepath=datapath+nmextension + '/'
print('************',nmextension,'*****************')

os.makedirs(savepath+'figs', exist_ok=False)

errorprec=False
linreg=0 # 1- linear regression | 0- ANN model | 2- do both and compare error
Flabel=True
SWf=True
clipf=True  # if oulierL or outlierH !=0 than decide to clip or ignore (ignoring not implemented in linear regression yet)


if costf==0:
    loss_square= "mean_absolute_error" 
elif costf==1:
    loss_square= "mean_squared_error" 

if SWf:
    lim=[50,850]
else:
    lim=[100,340]

dl=lim[1]-lim[0]    
dxtxt=1/35*dl
sytxt=2/35*dl

base_column_names = ['SZA','VZA','RAA','0.47', '0.55', '0.66', '0.86', '0.905', '0.935', '0.940', '1.24', '1.64','2.13', '1.375', '3.75', '3.96a', '3.96', '4.05', '4.46', '4.5', '6.7', '7.3', '8.5', '9.7', '11.0', '12.0', '13.33', '13.63', '13.93', '14.23']
mcols=['m'+tmp for tmp in base_column_names[3:]]
stdcols=['s'+tmp for tmp in base_column_names[3:]]
skewcols=['sk'+tmp for tmp in base_column_names[3:]]
kurtcols=['k'+tmp for tmp in base_column_names[3:]]
base_column_names=['SZA','VZA','RAA']+mcols+stdcols+skewcols+kurtcols

normlist=[SolarNorm,ThermalNorm,MixedNorm]

# %%
# ---------------------- Plotting function ----------------------------------------
def plot_the_loss_curve(epochs, mse_training, mse_validation):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.plot(epochs, mse_training, label="Training Loss")
    plt.plot(epochs, mse_validation, label="Validation Loss")
    merged_mse_lists = mse_training.tolist() + mse_validation
    highest_loss = max(merged_mse_lists)
    lowest_loss = min(merged_mse_lists)
    top_of_y_axis = highest_loss * 1.03
    bottom_of_y_axis = lowest_loss * 0.97
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.legend()
    # plt.show()

print("Defined the plot_the_loss_curve function.")

# ------------ Define linear regression model outputs ----------------------------------------
def get_outputs_linear_regression():
    dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,), name='dense_output')(preprocessing_layers)
    outputs = {'dense_output': dense_output}
    return outputs

# ------------ Define ANN model outputs ----------------------------------------
def get_outputs_dnn():
    regulator=tf.keras.regularizers.L1(l1=0.01)
    dense_output = tf.keras.layers.Dense(units=Nfeaters1, input_shape=(1,),
                                 activation=activationf, kernel_regularizer=regulator,
                                name='hidden_dense_layer_1')(preprocessing_layers)
    if Nfeaters2>0: 
        regulator=tf.keras.regularizers.L1(l1=0.001)
        dense_output = tf.keras.layers.Dense(units=Nfeaters2, input_shape=(1,),
                                       activation=activationf, kernel_regularizer=regulator,
                                       name='hidden_dense_layer_2')(dense_output)
    if Nfeaters3>0:
        regulator=tf.keras.regularizers.L1(l1=0.001)
        dense_output = tf.keras.layers.Dense(units=Nfeaters3, input_shape=(1,),
                                        activation=activationf,kernel_regularizer=regulator,
                                        name='hidden_dense_layer_3')(dense_output)
    if Nfeaters4>0:
        regulator=tf.keras.regularizers.L1(l1=0.001)
        dense_output = tf.keras.layers.Dense(units=Nfeaters4, input_shape=(1,),
                                        activation=activationf,kernel_regularizer=regulator,
                                        name='hidden_dense_layer_4')(dense_output)
    dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,),
                                      name='dense_output')(dense_output)
    outputs = {'dense_output': dense_output}
    return outputs

# =============================================================================
# Label transform / inverse-transform functions
#
#   method=0  no transform, plain data, plain MSE
#   method=1  log1p transform + weighted MSE (upweights large values)
#   method=2  Yeo-Johnson transform, plain MSE (normalises distribution)
# =============================================================================

# Global Yeo-Johnson transformer — fitted once on training labels, reused everywhere
pt = PowerTransformer(method='yeo-johnson')

# def transform_label(y, method):
#     """Apply the chosen label transform before training."""
#     if method == 0:
#         # No transform — return as-is
#         return np.array(y, dtype=np.float32)
#     elif method == 1:
#         # log1p — safe for zeros, compresses right tail
#         return np.log1p(np.array(y, dtype=np.float32))
#     elif method == 2:
#         # Yeo-Johnson — fit on training labels, then transform
#         y = np.array(y, dtype=np.float32).reshape(-1, 1)
#         transformed = pt.fit_transform(y)   # fit AND transform in one step
#         print(f"Yeo-Johnson optimal lambda: {pt.lambdas_[0]:.4f}")
#         return transformed.flatten()
    # -------------------- old ---------------
# def transform_label(y, method):
#     if method == 0:
#         return np.array(y, dtype=np.float32)
#     elif method == 1:
#         return np.log1p(np.array(y, dtype=np.float32))
#     elif method == 2:
#         y = np.array(y, dtype=np.float32).reshape(-1, 1)
#         transformed = pt.fit_transform(y)
#         # Save the actual min/max of transformed training data
#         pt.transformed_min_ = float(transformed.min())
#         pt.transformed_max_ = float(transformed.max())
#         print(f"Yeo-Johnson optimal lambda: {pt.lambdas_[0]:.4f}")
#         print(f"Transformed training range: [{pt.transformed_min_:.4f}, {pt.transformed_max_:.4f}]")
#         return transformed.flatten()

# def untransform_label(y, method):
#     """Invert the label transform to recover original scale."""
#     if method == 0:
#         # No transform was applied, nothing to invert
#         return np.array(y, dtype=np.float32)
#     elif method == 1:
#         # Invert log1p
#         return np.expm1(np.array(y, dtype=np.float32))
#     elif method == 2:
#         # Invert Yeo-Johnson using the already-fitted pt
#         y = np.array(y, dtype=np.float32).reshape(-1, 1)
#         return pt.inverse_transform(y).flatten()




#------- new -----------------

# Add this global alongside pt — fitted once on training labels
keras_label_normalizer = tf.keras.layers.Normalization(axis=None)

def transform_label(y, method):
    if method == 0:
        # Keras normalization — normalize to mean~0, std~1
        y_array = np.array(y, dtype=np.float32)
        keras_label_normalizer.adapt(y_array)  # fit on training labels
        return keras_label_normalizer(y_array).numpy()
    elif method == 1:
        return np.log1p(np.array(y, dtype=np.float32))
    elif method == 2:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        transformed = pt.fit_transform(y)
        pt.transformed_min_ = float(transformed.min())
        pt.transformed_max_ = float(transformed.max())
        print(f"Yeo-Johnson optimal lambda: {pt.lambdas_[0]:.4f}")
        print(f"Transformed training range: [{pt.transformed_min_:.4f}, {pt.transformed_max_:.4f}]")
        return transformed.flatten()

def untransform_label(y, method):
    if method == 0:
        # Invert Keras normalization: original = y * std + mean
        mean = keras_label_normalizer.mean.numpy()[0]
        std  = np.sqrt(keras_label_normalizer.variance.numpy()[0])
        return np.array(y, dtype=np.float32) * std + mean
    elif method == 1:
        return np.expm1(np.array(y, dtype=np.float32))
    elif method == 2:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        return pt.inverse_transform(y).flatten()

# =============================================================================
# Loss functions
# =============================================================================

def weighted_mse(method):
    """Return a Keras-compatible loss function for the chosen method."""
    def loss(y_true, y_pred):
        error = tf.square(y_true - y_pred)
        if method == 0:
            # No transform — plain MSE
            return tf.reduce_mean(error)
        elif method == 1:
            # log1p scale — upweight larger (log-scale) values
            weights = tf.math.log1p(tf.abs(y_true)) + 1.0
            return tf.reduce_mean(weights * error)
        elif method == 2:
            # Yeo-Johnson — distribution is ~normal, plain MSE is optimal
            return tf.reduce_mean(error)
    return loss

def weighted_mae(method):
    """Return a Keras-compatible loss function for the chosen method."""
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        if method == 0:
            # No transform — plain MSE
            return tf.reduce_mean(error)
        elif method == 1:
            # log1p scale — upweight larger (log-scale) values
            weights = tf.math.log1p(tf.abs(y_true)) + 1.0
            return tf.reduce_mean(weights * error)
        elif method == 2:
            # Yeo-Johnson — distribution is ~normal, plain MSE is optimal
            return tf.reduce_mean(error)
    return loss

# =============================================================================
# Model creation
# =============================================================================

def create_model(my_inputs, my_outputs, my_learning_rate, method):
    model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)

    if costf == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
            loss=weighted_mse(method),          # closure called here — returns inner loss fn
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
            loss=weighted_mae(method), 
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
    return model

# =============================================================================
# Early-stopping callback based on delta-RMSE in original scale
# =============================================================================

class DeltaRMSECallback(tf.keras.callbacks.Callback):
    """Stop training when improvement in RMSE (original scale) drops below threshold."""

    def __init__(self, threshold=0.001, patience=3, monitor='val_loss', method=0):
        super().__init__()
        self.threshold = threshold
        self.patience  = patience
        self.monitor   = monitor
        self.method    = method
        self.prev_rmse = None
        self.epochs_below_threshold = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return

        # RMSE in transformed scale, then invert to original scale
        transformed_rmse = np.sqrt(current_loss)
        current_rmse = untransform_label(np.array([transformed_rmse]), self.method)[0]

        if self.prev_rmse is not None:
            delta_rmse = self.prev_rmse - current_rmse
            print(f"\nEpoch {epoch+1}: RMSE={current_rmse:.4f} (original scale), "
                  f"delta_RMSE={delta_rmse:.6f}")

            if abs(delta_rmse) < self.threshold:
                self.epochs_below_threshold += 1
                print(f" -> Below threshold {self.epochs_below_threshold}/{self.patience}")
                if self.epochs_below_threshold >= self.patience:
                    print(f"\nStopping: delta_RMSE < {self.threshold} "
                          f"for {self.patience} consecutive epochs.")
                    self.model.stop_training = True
            else:
                self.epochs_below_threshold = 0

        self.prev_rmse = current_rmse

# =============================================================================
# Training function
# =============================================================================

def train_model(model, dataset, epochs, batch_size, label_name,
                validation_split, delta_rmse_threshold, patience):
    """Feed a dataset into the model, stopping when delta-RMSE criterion is met."""

    features = {name: np.array(value) for name, value in dataset.items()}
    # transform_label fits pt (for method=2) and returns transformed labels
    label = transform_label(np.array(features.pop(label_name)), transform_method)

    early_stop = DeltaRMSECallback(
        threshold=delta_rmse_threshold,
        patience=patience,
        monitor='val_loss' if validation_split > 0 else 'loss',
        method=transform_method
    )

    history = model.fit(
        x=features,
        y=label,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )

    epochs_ran = history.epoch
    hist = pd.DataFrame(history.history)
    mse  = hist[loss_square]
    return epochs_ran, mse, history.history

# =============================================================================
# Prediction wrapper — always returns original-scale predictions
# =============================================================================

# def predict(model, features, method):
#     """Run model.predict and invert the label transform."""
#     raw_preds = model.predict(features)
    
#     # model output is a dict — extract the dense_output array
#     if isinstance(raw_preds, dict):
#         transformed_preds = raw_preds['dense_output'].flatten()
#     else:
#         transformed_preds = raw_preds.flatten()
    
#     return untransform_label(transformed_preds, method)

# def predict(model, features, method):
#     """Run model.predict and invert the label transform."""
#     transformed_preds = model.predict(features).flatten()
#     return untransform_label(transformed_preds, method)


def predict(model, features, method):
    raw_preds = model.predict(features)
    
    if isinstance(raw_preds, dict):
        transformed_preds = raw_preds['dense_output'].flatten()
    else:
        transformed_preds = raw_preds.flatten()
    
    print(f"  Raw predictions range: [{np.nanmin(transformed_preds):.4f}, {np.nanmax(transformed_preds):.4f}]")
    print(f"  NaNs in raw predictions: {np.sum(np.isnan(transformed_preds))}")
    
    if method == 2:
        # Clip to the exact range the inverse transform was fitted on
        clip_min = pt.transformed_min_
        clip_max = pt.transformed_max_
        n_clipped = np.sum((transformed_preds < clip_min) | (transformed_preds > clip_max))
        print(f"  Clipping {n_clipped} values to [{clip_min:.4f}, {clip_max:.4f}]")
        transformed_preds = np.clip(transformed_preds, clip_min, clip_max)
    
    result = untransform_label(transformed_preds, method)
    
    print(f"  Original scale range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
    print(f"  NaNs after inverse transform: {np.sum(np.isnan(result))}")
    
    return result



#------ new ------
# Add this global alongside pt — fitted once on training labels
keras_label_normalizer = tf.keras.layers.Normalization(axis=None)

def transform_label(y, method):
    if method == 0:
        # Keras normalization — normalize to mean~0, std~1
        y_array = np.array(y, dtype=np.float32)
        keras_label_normalizer.adapt(y_array)  # fit on training labels
        return keras_label_normalizer(y_array).numpy()
    elif method == 1:
        return np.log1p(np.array(y, dtype=np.float32))
    elif method == 2:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        transformed = pt.fit_transform(y)
        pt.transformed_min_ = float(transformed.min())
        pt.transformed_max_ = float(transformed.max())
        print(f"Yeo-Johnson optimal lambda: {pt.lambdas_[0]:.4f}")
        print(f"Transformed training range: [{pt.transformed_min_:.4f}, {pt.transformed_max_:.4f}]")
        return transformed.flatten()

def untransform_label(y, method):
    if method == 0:
        # Invert Keras normalization: original = y * std + mean
        mean = keras_label_normalizer.mean.numpy()[0]
        std  = np.sqrt(keras_label_normalizer.variance.numpy()[0])
        return np.array(y, dtype=np.float32) * std + mean
    elif method == 1:
        return np.expm1(np.array(y, dtype=np.float32))
    elif method == 2:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        return pt.inverse_transform(y).flatten()

# -----------------------------


'''

This will print what the actual training range was, so you'll see something like:

Transformed training range: [-3.2, 4.1]
Clipping 15234 values to [-3.2, 4.1]


This will tell you exactly where the NaNs come from. Most likely you will see one of these:
# Case 1 — NaNs already in raw model output
NaNs in raw predictions: 523   ← problem is in the model itself (exploding gradients)

# Case 2 — extreme values breaking inverse transform  
Transformed predictions: min=-847.3, max=1205.6   ← way outside [-3, 4] YJ range
NaNs in raw predictions: 0
NaNs after inverse transform: 523   ← inverse transform can't handle extremes  
'''


# %%
dat=np.load(fname)
column_names=dat["column_names"]
label_normalization_factor=dat["label_normalization_factor"]
SWnorm=dat["SWnorm"]
MIXnorm=dat["MIXnorm"]
LWnorm=dat["LWnorm"]
Norms=dat["Norms"]
df_mean=dat["df_mean"]
df_std=dat["df_std"]
name=fname[:-3]+'pkl'
df = pd.read_pickle(name)

Leni=len(df)

# Filter rows based on a column condition
df = df[ (df['mCTH'] < mCTHth) & (df['stdCTH'] < stdCTHth) ]

# Drop unwanted columns
drop_list=[ind for ind in column_names if ind not in featers_list]
print(drop_list)
print(column_names)
df = df.drop(columns=drop_list)
column_names=featers_list.copy()
print(column_names)

if Flabel:
    label_name = "Flux"

# ---- Distribution plots for all transform options ----
fig, axes = plt.subplots(2, 3, figsize=(30, 13))
transforms = {
    'Original':      df[label_name],
    'log1p':         np.log1p(df[label_name]),
    'sqrt':          np.sqrt(df[label_name]),
    'Box-Cox':       boxcox(df[label_name] + 1)[0],
    'Yeo-Johnson':   PowerTransformer('yeo-johnson').fit_transform(df[[label_name]]).flatten(),
}

for ax, (name, data) in zip(axes.flatten(), transforms.items()):
    ax.hist(data, bins=50)
    ax.set_title(name)
    skew = pd.Series(data).skew()
    ax.set_xlabel(f'skewness: {skew:.2f}')

plt.tight_layout()
# plt.show(block=False)
fig.savefig(savepath+nmextension+'dataHist.png', dpi=150, bbox_inches='tight')
plt.close()

print('##### Number of FOVs initial and final', Leni, len(df))

df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

df_train=df.iloc[0:int(np.ceil(trainprec*len(df))),:]
df_test=df.iloc[int(np.ceil(trainprec*len(df)))+1:,:]

# ------------------------------------------------------------------
#   Focising on particular values for the training
# --------------------------------------------------------------------
if train_label_low is not None or train_label_high is not None:
    lo = train_label_low  if train_label_low  is not None else -np.inf
    hi = train_label_high if train_label_high is not None else  np.inf
    
    mask_train = (df_train[label_name] * label_normalization_factor > lo) & \
                 (df_train[label_name] * label_normalization_factor < hi)
    n_before = len(df_train)
    df_train = df_train[mask_train].reset_index(drop=True)
    print(f'Training label filter [{lo}, {hi}]: {n_before} -> {len(df_train)} samples '
          f'({n_before - len(df_train)} removed, {(n_before-len(df_train))/n_before*100:.1f}%)')
    
    
    mask_test = (df_test[label_name] * label_normalization_factor > lo) & \
                 (df_test[label_name] * label_normalization_factor < hi)
    n_before = len(df_test)
    df_test = df_test[mask_test].reset_index(drop=True)
    print(f'Training label filter [{lo}, {hi}]: {n_before} -> {len(df_test)} samples '
          f'({n_before - len(df_test)} removed, {(n_before-len(df_test))/n_before*100:.1f}%)')
    
# -----------------------------------------------------------------------------------------


lentest=len(df_test)
if linearfit_split>0:
    df_test_lincorr=df_test.iloc[int(np.ceil(linearfit_split*lentest))+1:,:]
    df_test=df_test.iloc[0:int(np.ceil(linearfit_split*lentest)),:]

# %% START ANN — build input and preprocessing layers
inputs=[1]*(df.shape[1]-1)
inputs_list=[1]*(df.shape[1]-1)
for i in range(0,df.shape[1]-1):
    inputs_list[i]= (tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=column_names[i]))
    inputs[i]= (column_names[i], tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=column_names[i]))
inputs_dic=dict(inputs)

norm_inputs=[1]*(df.shape[1]-1)
for i in range(0,df.shape[1]-1):
    print('normalizing layer',i)
    tmpstr='normalization_'+column_names[i]
    tmp = tf.keras.layers.Normalization(name=tmpstr, axis=None)
    tmp.adapt(df_train[column_names[i]])
    norm_inputs[i] = tmp(inputs_dic.get(column_names[i]))

preprocessing_layers = tf.keras.layers.Concatenate()(norm_inputs)
print(preprocessing_layers)
print("Preprocessing layers defined.")

# *********************************************************************************************
# NOTE: Keras Normalization layers for the Flux label are named flux_normalizer /
# test_flux_normalizer to avoid shadowing the transform_label / untransform_label functions.

flux_normalizer = tf.keras.layers.Normalization(axis=None)
flux_normalizer.adapt(np.array(df_train['Flux']))

test_flux_normalizer = tf.keras.layers.Normalization(axis=None)
test_flux_normalizer.adapt(np.array(df_test['Flux']))

if internal_test:
    df_Input=df_test.copy()
    df_Input.drop('Flux',inplace=True, axis=1)
    array_Input=np.array(df_Input)
    
    df_label=df_test['Flux']
    label= np.array(df_label)
    
    if SWf:
        np.savez(savepath+r'/SW_norm_factors_'+nmextension+'.npz',
                 std=np.nanstd(label),mean=np.nanmean(label),
                 norm_factor=label_normalization_factor,
                 DFmean=df_mean,DFstd=df_std,AnglesNorms=Norms,
                 SWnorms=SWnorm,LWnorms=LWnorm,MIXnorms=MIXnorm,Normlist=normlist)
    else:
        np.savez(savepath+r'/LW_norm_factors_'+nmextension+'.npz',
                 std=np.nanstd(label),mean=np.nanmean(label),
                 norm_factor=label_normalization_factor,
                 DFmean=df_mean,DFstd=df_std,AnglesNorms=Norms,
                 SWnorms=SWnorm,LWnorms=LWnorm,MIXnorms=MIXnorm,Normlist=normlist)

    LABEL_test=label*label_normalization_factor

    if linearfit_split:
        df_Input_lincorr=df_test_lincorr.copy()
        df_Input_lincorr.drop('Flux',inplace=True, axis=1)
        array_Input_lincorr=np.array(df_Input_lincorr)
        df_label_lincorr=df_test_lincorr['Flux']
        label_lincorr= np.array(df_label_lincorr)
        LABEL_test_lincorr=label_lincorr*label_normalization_factor

# %% ------------------------------------Linear regression----------------------------------------------------------------
if linreg!=0:

    label_name = "Flux"
    outputs = get_outputs_linear_regression()
    Lin_model = create_model(inputs_dic, outputs, learning_rate, transform_method)
    epochs, mse, history = train_model(Lin_model, df_train, Nepochs, batch_size,
                                       label_name, validation_split,
                                       delta_rmse_threshold, patience)

    test_features = {name:np.array(value) for name, value in df_test.items()}
    # Use test_flux_normalizer (not test_label) to avoid name shadowing
    lin_test_label = test_flux_normalizer(test_features.pop(label_name))
    print("\n Evaluate the linear regression model against the test set:")
    Score=Lin_model.evaluate(x=test_features, y=lin_test_label,
                             batch_size=batch_size, return_dict=True)

    if SWf:
        Lin_model.save(savepath+"/Linear_model_SW_"+nmextension+".keras")
    else:
        Lin_model.save(savepath+"/Linear_model_LW_"+nmextension+".keras")

    lin_normMSE=Score[loss_square]
    tmp=df_test['Flux']*label_normalization_factor
    lin_Ymean=np.mean(tmp)
    lin_Ystd=np.std(tmp)
    lin_N=np.size(tmp)
    lin_RMSE=np.sqrt(lin_normMSE)*lin_Ystd
    print('\n')
    print('@@@@@ LINEAR REGRESSION @@@@@@@@@')
    print(lin_Ymean,lin_Ystd,lin_N,lin_normMSE)
    print('##########################')
    print('RMSE= ',np.sqrt(lin_RMSE), ' W/m^2')
    print('##########################')

    Lin_weights=Lin_model.get_weights()[(df.shape[1] - 1)*3]
    fig=plt.figure(figsize=(30, 13))
    plt.set_cmap('bwr')
    ax=plt.subplot(1,1,1)    
    vmin = np.nanmin(Lin_weights)
    vmax = np.nanmax(Lin_weights)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.pcolor(Lin_weights, norm=norm)
    plt.colorbar()
    ax.set_ylabel('Input features')
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.5,len(featers_list)+0.5))
    ax.set_yticklabels(featers_list)
    ax.axes.set_aspect('equal')
    # plt.show(block=False)
    fig.savefig(savepath+'/figs/'+nmextension+'Linear_weights.png', dpi=150, bbox_inches='tight')
    with open(savepath+'/figs/Linear_weights.pkl', 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.close()

    if internal_test:
        test_features = {name:np.array(value) for name, value in df_Input.items()}
        Lin_Yhat = predict(Lin_model, test_features, transform_method)
        # Lin_Yhat=Lin_Yhat['dense_output']
        Lin_Yhat = Lin_Yhat*label_normalization_factor  #(np.nanstd(label)*Lin_Yhat+np.nanmean(label))*label_normalization_factor
        
        N_tooH=np.nansum(Lin_Yhat>outlierH)
        N_tooL=np.nansum(Lin_Yhat<outlierL)
        
        if outlierH != 0:
            # if clipf:
            Lin_Yhat=np.where(Lin_Yhat>outlierH,outlierH,Lin_Yhat)
            print('!!!!!!!!!  Clipping high values >',outlierH, ' N,f= ',N_tooH,N_tooH/len(Lin_Yhat), '  !!!!!!!!!!!!!!!!' )
            # else:
                # tmp_ind=np.nonzero(Lin_Yhat<outlierH)
                # Lin_Yhat=Lin_Yhat[tmp_ind]
                # LABEL_test=LABEL_test[tmp_ind]
                # print('!!!!!!!!!  Ignoring high values >',outlierH, ' N,f= ',N_tooH,N_tooH/len(Lin_Yhat), '  !!!!!!!!!!!!!!!!' )
        if outlierL !=0: 
            # if clipf:
            Lin_Yhat=np.where(Lin_Yhat<outlierL,outlierL,Lin_Yhat)
            print('!!!!!!!!!  Clipping low values <',outlierL, ' N,f= ',N_tooL,N_tooL/len(Lin_Yhat), '  !!!!!!!!!!!!!!!!' )
            # else:
                # tmp_ind=np.nonzero(Lin_Yhat>outlierL)
                # Lin_Yhat=Lin_Yhat[tmp_ind]
                # LABEL_test=LABEL_test[tmp_ind]
                # print('!!!!!!!!!  Ignoring high values >',outlierH, ' N,f= ',N_tooH,N_tooH/len(Lin_Yhat), '  !!!!!!!!!!!!!!!!' )
        
        tmp0=np.reshape(LABEL_test,np.shape(Lin_Yhat))
        Lin_reg = LinearRegression().fit(tmp0.reshape(-1, 1),Lin_Yhat.reshape(-1, 1))
        Lin_R2=Lin_reg.score(tmp0.reshape(-1, 1),Lin_Yhat.reshape(-1, 1))
        Lin_a=Lin_reg.coef_[0]
        Lin_b=Lin_reg.intercept_
        Lin_txt='Y='+str("{0:2.1f}".format(Lin_a[0]))+'X+'+str("{0:2.2f}".format(Lin_b[0]))
        Lin_Ytag = Lin_reg.predict(Lin_Yhat.reshape(-1, 1)).flatten()
        Lin_Yhat=np.squeeze(Lin_Yhat)
        Lin_Ytag=np.squeeze(Lin_Ytag)

# ---------------------- ANN -------------------------------------------------------------------------------
if linreg!=1:

    label_name = "Flux"
    dnn_outputs = get_outputs_dnn()
    ANN_model = create_model(inputs_dic, dnn_outputs, learning_rate, transform_method)

    epochs, mse, history = train_model(ANN_model, df_train, Nepochs, batch_size,
                                       label_name, validation_split,
                                       delta_rmse_threshold, patience)

    test_features = {name:np.array(value) for name, value in df_test.items()}
    # Use test_flux_normalizer (not test_label) to avoid name shadowing
    ANN_test_label = test_flux_normalizer(np.array(test_features.pop(label_name)))
    print("\n Evaluate the new model against the test set:")
    Score=ANN_model.evaluate(x=test_features, y=ANN_test_label,
                             batch_size=batch_size, return_dict=True)

    if SWf:
        ANN_model.save(savepath+"/ANN_model_SW_"+nmextension+".keras")
    else:
        ANN_model.save(savepath+"/ANN_model_LW_"+nmextension+".keras")

    normMSE=Score[loss_square]
    tmp=df_test['Flux']*label_normalization_factor
    Ymean=np.mean(tmp)
    Ystd=np.std(tmp)
    N=np.size(tmp)
    RMSE=np.sqrt(normMSE)*Ystd
    print('\n')
    print('@@@@@ ANN model @@@@@@@@@')
    print(Ymean,Ystd,N,normMSE)
    print('##########################')
    print('RMSE= ',RMSE, ' W/m^2')
    print('##########################')

    weights=ANN_model.get_weights()[(df.shape[1] - 1)*3]
    weights_out=ANN_model.get_weights()[(df.shape[1] - 1)*3+2]

    fig = plt.figure(figsize=(30, 13))
    gs = fig.add_gridspec(2, 2, width_ratios=(4,1), height_ratios=(1,8),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_out = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_out.tick_params(axis="x", labelbottom=False)

    plt.set_cmap('bwr')
    plt.title('ANN Regression weights')
    vmin = np.nanmin(weights)
    vmax = np.nanmax(weights)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    main=ax.pcolor(weights, norm=norm)
    plt.colorbar(main)
    ax.set_yticks(np.arange(0.5,len(featers_list)+0.5))
    ax.set_yticklabels(featers_list)
    ax.set_xticks(np.arange(0.5,len(featers_list)+0.5))
    ax.set_xticklabels(np.arange(1,len(featers_list)+1,1))
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Input featers')

    vmin = np.nanmin(weights_out)
    vmax = np.nanmax(weights_out)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    outs=ax_out.pcolor(weights_out.T, norm=norm)
    plt.colorbar(outs)
    ax_out.set_yticklabels([])
    ax_out.set_yticks([])

    # plt.show(block=False)
    fig.savefig(savepath+'/figs/'+nmextension+'ANN_weights.png', dpi=150, bbox_inches='tight')
    with open(savepath+'/figs/ANN_weights.pkl', 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.close()

    if internal_test:
        test_features = {name:np.array(value) for name, value in df_Input.items()}
        Yhat = predict(ANN_model, test_features, transform_method)
        # Yhat=Yhat['dense_output']
        Yhat = Yhat*label_normalization_factor  #(np.nanstd(label)*Yhat+np.nanmean(label))*label_normalization_factor
        
        
        N_tooH=np.nansum(Yhat>outlierH)
        N_tooL=np.nansum(Yhat<outlierL)
        
        if outlierH != 0:   
            if clipf:
                Yhat=np.where(Yhat>outlierH,outlierH,Yhat)
                print('!!!!!!!!!  Clipping high values >',outlierH, ' N,f= ',N_tooH,N_tooH/len(Yhat), '  !!!!!!!!!!!!!!!!' )
            else:
                tmp_ind=np.nonzero(Yhat<outlierH)
                Yhat=Yhat[tmp_ind]
                LABEL_test=LABEL_test[tmp_ind]
                print('!!!!!!!!!  Ignoring high values >',outlierH, ' N,f= ',N_tooH,N_tooH/len(Yhat), '  !!!!!!!!!!!!!!!!' )
        if outlierL !=0: 
            if clipf:
                Yhat=np.where(Yhat<outlierL,outlierL,Yhat)
                print('!!!!!!!!!  Clipping low values <',outlierL, ' N,f= ',N_tooL,N_tooL/len(Yhat), '  !!!!!!!!!!!!!!!!' )
            else:
                tmp_ind=np.nonzero(Yhat<outlierL)
                Yhat=Yhat[tmp_ind]
                LABEL_test=LABEL_test[tmp_ind]
                print('!!!!!!!!!  Ignoring high values >',outlierL, ' N,f= ',N_tooL,N_tooL/len(Yhat), '  !!!!!!!!!!!!!!!!' )
        
              
        tmp0=np.reshape(LABEL_test,np.shape(Yhat))
        reg = LinearRegression().fit(tmp0.reshape(-1, 1),Yhat.reshape(-1, 1))
        R2=reg.score(tmp0.reshape(-1, 1),Yhat.reshape(-1, 1))
        a=reg.coef_[0]
        b=reg.intercept_
        txt='Y='+str("{0:2.2f}".format(a[0]))+'X+'+str("{0:2.2f}".format(b[0]))

        if linearfit_split:
            test_features_lincorr = {name:np.array(value) for name, value in df_Input_lincorr.items()}
            Yhat_lincorr= predict(ANN_model, test_features_lincorr, transform_method)
            # Yhat_lincorr=Yhat_lincorr['dense_output']
            Yhat_lincorr = Yhat_lincorr*label_normalization_factor #(np.std(label)*Yhat_lincorr+np.nanmean(label))*label_normalization_factor
            Ytag = reg.predict(Yhat_lincorr.reshape(-1, 1)).flatten()
            Yhat_lincorr=np.squeeze(Yhat_lincorr)
            Ytag=np.squeeze(Ytag)
        Yhat=np.squeeze(Yhat)

if linreg>1:
    dnormMSE=normMSE-lin_normMSE
    dRMSE=RMSE-lin_RMSE
    print('\n')
    print('########### ANN - LINEAR REGRESSION ###########')
    print('delta normalized MSE= ',dnormMSE)
    print('delta RMSE= ',dRMSE, ' W/m^2')
    print('##########################')

#%%
if internal_test:
# $$$$$$$$$$$$$$$$$$$$$$$$ FIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if linreg!=1:
        fig = plt.figure(figsize=(30, 13))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)

        ax_histx.tick_params(axis="x", labelbottom=False)

        # ax.scatter(LABEL_test,Yhat,label='ANN vs. Truth')
        
        h = ax.hist2d(LABEL_test, Yhat, bins=200, range=[[lim[0],lim[1]],[lim[0],lim[1]]],
              cmap='turbo', norm=mpl.colors.LogNorm())
        
        pos = ax.get_position()  # returns [x0, y0, width, height] in figure fraction
        ax_cb = fig.add_axes([pos.x1 + 0.01,  # just to the right of ax
                      pos.y0,           # same bottom as ax
                      0.015,             # colorbar width
                      pos.height])      # same height as ax
        
        plt.colorbar(h[3], cax=ax_cb, label='Count')
        # divider = make_axes_locatable(ax)
        # ax_cb = divider.append_axes("right", size="3%", pad=0.1)  # size controls width
        # plt.colorbar(h[3], cax=ax_cb, label='Count')
        
        
        ax.plot([lim[0],lim[1]],[lim[0],lim[1]],'k',label='X=Y',linewidth=3)
        if linearfit_split:
            ax.scatter(Ytag, LABEL_test_lincorr,color='g',label='Fit vs. Truth',alpha=0.2)
            ax.plot(Yhat_lincorr,Ytag,'r',label='Linear Fit',linewidth=3)
        ax.legend(loc='lower right')

        if errorprec:
            error_factor=LABEL_test
        else:
            error_factor=1

        ANN_RMSE=np.sqrt(np.nanmean((Yhat-LABEL_test)**2/error_factor))
        ANN_MAE_perc=np.nanmean(np.abs(Yhat-LABEL_test)/LABEL_test)
        ANN_ME=np.nanmean(Yhat-LABEL_test)/error_factor
        perL=np.percentile(LABEL_test,10); perH=np.percentile(LABEL_test,90)
        err_perL=np.nanmean(np.abs(Yhat-np.where(LABEL_test<perL,LABEL_test,np.nan))/error_factor)
        err_perM=np.nanmean(np.abs(Yhat-np.where((LABEL_test>perL)&(LABEL_test<perH),LABEL_test,np.nan))/error_factor)
        err_perH=np.nanmean(np.abs(Yhat-np.where(LABEL_test>perH,LABEL_test,np.nan))/error_factor)
        ANN_MAE=np.nanmean(np.abs(Yhat-LABEL_test)/error_factor)
        ANN_R2=1-(np.nansum((Yhat-LABEL_test)**2)/np.nansum((Yhat-np.nanmean(LABEL_test))**2))
        if linearfit_split:
            ANN_MAE_lincorr=np.nanmean(np.abs(Ytag-LABEL_test_lincorr)/error_factor)

        # txtdy=20
        # fontsz=25
        ax.text(lim[0]+dxtxt, lim[1]-txtdy,    '$RMSE (Wm^{-2})$= '+str("{0:2.2f}".format(ANN_RMSE)),   fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*2,  '$MAE (Wm^{-2})$= ' +str("{0:2.2f}".format(ANN_MAE)),    fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*3,  '$ME (Wm^{-2})$= '  +str("{0:2.2f}".format(ANN_ME)),     fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*4,  '$R^2 $= '          +str("{0:2.2f}".format(ANN_R2)),     fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*5,  '$MAE (\%)$= '      +str("{0:2.2f}".format(ANN_MAE_perc)),fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*6,  '$MAE_{10\%} (Wm^{-2})$= ' +str("{0:2.2f}".format(err_perL)), fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*7,  '$MAE_{center} (Wm^{-2})$= '+str("{0:2.2f}".format(err_perM)), fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*8,  '$MAE_{90\%} (Wm^{-2})$= '  +str("{0:2.2f}".format(err_perH)), fontsize=fontsz)
        
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*9, txt,fontsize=fontsz, color='r')
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*10, '$R^2$='+str("{0:2.2f}".format(R2)), fontsize=fontsz, color='r')
        if linearfit_split:
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*11,'$MAE_{linear-correction} (Wm^{-2})$= '+str("{0:2.2f}".format(ANN_MAE_lincorr)),fontsize=fontsz)

        ax.set_ylabel('ANN/Fit ($Wm^{-2}$)')
        ax.set_xlabel('CERES ($Wm^{-2}$)')
        ax.set_xlim((lim[0],lim[1]))
        ax.set_ylim((lim[0],lim[1]))

        binwidth = 0.25
        xymax = max(np.max(np.abs(Yhat)), np.max(np.abs(LABEL_test)))

        bins = np.arange(lim[0], lim[1] + binwidth, binwidth)
        ax_histx.hist(Yhat, bins=bins, alpha=0.9,  label='ANN')
        ax_histx.hist(LABEL_test, bins=bins, alpha=0.65, label='CERES')
        
        # Update dashed percentile lines to span the full log y-axis
        ymin, ymax = ax_histx.get_ylim()
        ax_histx.plot([perL,perL],[ymin,ymax],'--k')
        ax_histx.plot([perH,perH],[ymin,ymax],'--k')

        if linearfit_split:
            ax_histx.hist(Ytag, bins=bins, alpha=0.4, label='Fit')
        ax_histx.legend(loc='lower right')
        ax_histx.set_ylabel('Count')
        ax_histx.set_yscale('log')
        ax_histx.set_xlim((lim[0],lim[1]))
        
        if SWf:
            np.savez(savepath+r'/SW_ANNtestarray_'+nmextension+'.npz',
                     model=Yhat,Label=LABEL_test,RMSE=ANN_RMSE,MAE=ANN_MAE,
                     ME=ANN_ME,R2=ANN_R2,MAEperc=ANN_MAE_perc,
                     Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,
                     fiteq=txt,fitR2=R2)
        else:
            np.savez(savepath+r'/LW_ANNtestarray_'+nmextension+'.npz',
                     model=Yhat,Label=LABEL_test,RMSE=ANN_RMSE,MAE=ANN_MAE,
                     ME=ANN_ME,R2=ANN_R2,MAEperc=ANN_MAE_perc,
                     Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,
                     fiteq=txt,fitR2=R2)

        print('########## ANN RMSE=',np.sqrt(np.nanmean((Yhat-LABEL_test)**2)), 'Wm^-2 ###############')
        print('^^^^^^^^^^^^^^^^^^^^ mean: [lable,test,ANN]',np.nanmean(label),np.nanmean(LABEL_test),np.nanmean(Yhat))
        print('^^^^^^^^^^^^^^^^^^^^ STD: [lable,test,ANN]',np.nanstd(label),np.nanstd(LABEL_test),np.nanstd(Yhat))
        if linearfit_split:
            print('########## corrected RMSE=',np.sqrt(np.nanmean((Ytag-LABEL_test_lincorr)**2)), 'Wm^-2 ##############')

        # plt.show(block=False)
        fig.savefig(savepath+'/figs/'+nmextension+'ANN.png', dpi=150, bbox_inches='tight')
        with open(savepath+'/figs/ANN.pkl', 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.close()

    if linreg!=0:
        fig = plt.figure(figsize=(30, 13))
        gs = fig.add_gridspec(2, 2, width_ratios=(4,1), height_ratios=(1,4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)

        ax_histx.tick_params(axis="x", labelbottom=False)

        # ax.scatter(LABEL_test,Lin_Yhat,label='Linear reg vs. Truth')
        
        h = ax.hist2d(LABEL_test, Lin_Yhat, bins=200, range=[[lim[0],lim[1]],[lim[0],lim[1]]],
              cmap='turbo', norm=mpl.colors.LogNorm())
        
        pos = ax.get_position()  # returns [x0, y0, width, height] in figure fraction
        ax_cb = fig.add_axes([pos.x1 + 0.01,  # just to the right of ax
                      pos.y0,           # same bottom as ax
                      0.015,             # colorbar width
                      pos.height])      # same height as ax
        
        plt.colorbar(h[3], cax=ax_cb, label='Count')
        
        # divider = make_axes_locatable(ax)
        # ax_cb = divider.append_axes("right", size="3%", pad=0.1)  # size controls width
        # plt.colorbar(h[3], cax=ax_cb, label='Count')
        
        
        ax.plot([lim[0],lim[1]],[lim[0],lim[1]],'k',label='X=Y',linewidth=3)
        ax.legend(loc='lower right')

        if errorprec:
            error_factor=LABEL_test
        else:
            error_factor=1

        Lin_RMSE=np.sqrt(np.nanmean((Lin_Yhat-LABEL_test)**2/error_factor))
        Lin_MAE_perc=np.nanmean(np.abs(Lin_Yhat-LABEL_test)/LABEL_test)
        Lin_ME=np.nanmean(Lin_Yhat-LABEL_test)
        perL=np.percentile(LABEL_test,10); perH=np.percentile(LABEL_test,90)
        err_perL=np.nanmean(np.abs(Lin_Yhat-np.where(LABEL_test<perL,LABEL_test,np.nan))/error_factor)
        err_perM=np.nanmean(np.abs(Lin_Yhat-np.where((LABEL_test>perL)&(LABEL_test<perH),LABEL_test,np.nan))/error_factor)
        err_perH=np.nanmean(np.abs(Lin_Yhat-np.where(LABEL_test>perH,LABEL_test,np.nan))/error_factor)
        Lin_MAE=np.nanmean(np.abs(Lin_Yhat-LABEL_test)/error_factor)
        Lin_R2=1-(np.nansum((Lin_Yhat-LABEL_test)**2)/np.nansum((Lin_Yhat-np.nanmean(LABEL_test))**2))
        if linearfit_split:
            Lin_MAE_lincorr=np.nanmean(np.abs(Lin_Ytag-LABEL_test_lincorr)/error_factor)

        ax.text(lim[0]+dxtxt, lim[1]-txtdy,   '$RMSE (Wm^{-2})$= '+str("{0:2.2f}".format(Lin_RMSE)),    fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*2, '$MAE (Wm^{-2})$= ' +str("{0:2.2f}".format(Lin_MAE)),     fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*3, '$ME (Wm^{-2})$= '  +str("{0:2.2f}".format(Lin_ME)),      fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*4, '$R^2 $= '          +str("{0:2.2f}".format(Lin_R2)),      fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*5, '$MAE (\%)$= '      +str("{0:2.2f}".format(Lin_MAE_perc)),fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*6, '$MAE_{10\%} (Wm^{-2})$= ' +str("{0:2.2f}".format(err_perL)), fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*7, '$MAE_{center} (Wm^{-2})$= '+str("{0:2.2f}".format(err_perM)), fontsize=fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*8, '$MAE_{90\%} (Wm^{-2})$= '  +str("{0:2.2f}".format(err_perH)), fontsize=fontsz)
        
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*9, Lin_txt,fontsize=fontsz, color='r')
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*10, '$R^2$='+str("{0:2.2f}".format(Lin_R2)), fontsize=fontsz, color='r')
        if linearfit_split:
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*11,'$MAE_{linear-correction} (Wm^{-2})$= '+str("{0:2.2f}".format(Lin_MAE_lincorr)),fontsize=fontsz)

        ax.set_ylabel('Linear reg/Fit ($Wm^{-2}$)')
        ax.set_xlabel('CERES ($Wm^{-2}$)')
        ax.set_xlim((lim[0],lim[1]))
        ax.set_ylim((lim[0],lim[1]))

        binwidth = 0.25
        xymax = max(np.max(np.abs(Lin_Yhat)), np.max(np.abs(LABEL_test)))

        bins = np.arange(lim[0], lim[1] + binwidth, binwidth)
        ax_histx.hist(Lin_Yhat, bins=bins, alpha=0.9,  label='Linear reg')
        ax_histx.hist(LABEL_test, bins=bins, alpha=0.65, label='CERES')

        # Update dashed percentile lines to span the full log y-axis
        ymin, ymax = ax_histx.get_ylim()
        ax_histx.plot([perL,perL],[ymin,ymax],'--k')
        ax_histx.plot([perH,perH],[ymin,ymax],'--k')

        ax_histx.legend(loc='lower right')
        ax_histx.set_ylabel('Count')
        ax_histx.set_yscale('log')
        ax_histx.set_xlim((lim[0],lim[1]))
        
        # plt.show(block=False)
        fig.savefig(savepath+'/figs/'+nmextension+'ALinear.png', dpi=150, bbox_inches='tight')
        with open(savepath+'/figs/Linear.pkl', 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.close()

        if SWf:
            np.savez(savepath+r'/SW_Lintestarray_'+nmextension+'.npz',
                     model=Yhat,Label=LABEL_test,RMSE=Lin_RMSE,MAE=Lin_MAE,
                     R2=Lin_R2,MAEperc=Lin_MAE_perc,
                     Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,
                     Lims=lim,fiteq=txt,fitR2=R2)
        else:
            np.savez(savepath+r'/LW_Lintestarray_'+nmextension+'.npz',
                     model=Yhat,Label=LABEL_test,RMSE=Lin_RMSE,MAE=Lin_MAE,
                     R2=Lin_R2,MAEperc=Lin_MAE_perc,
                     Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,
                     Lims=lim,fiteq=txt,fitR2=R2)

        print('########## Linear Reg RMSE=',np.sqrt(np.nanmean((Lin_Yhat-LABEL_test)**2)), 'Wm^-2 ###############')
        print('########## corrected RMSE=',np.sqrt(np.nanmean((Lin_Ytag-LABEL_test)**2)), 'Wm^-2 ##############')