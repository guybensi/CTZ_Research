# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:05:32 2024

@author: eeytan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:08:17 2024

@author: eeytan
"""

import datetime
import argparse
import json
import math
import os
import sys
import time

import pandas as pd
import netCDF4 as nc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
# import multiprocessing
# import multiprocessing.pool as pool 
import tensorflow as tf
from tensorflow.keras import layers
# import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.colors as colors

try:
    import Twilight_module as tw
except ImportError:
    tw = None
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large','figure.figsize': (5, 5),'axes.labelsize': 'x-large','axes.titlesize':'x-large','xtick.labelsize':'x-large','ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

font = {'weight' : 'bold','size'   : 22}
plt.rc('font', **font)
#import h5py
#import gdal

parser = argparse.ArgumentParser(description='All_Sky_AIflux regression workflow')
parser.add_argument('--data-path', type=str, default='', help='Override the default data directory')
parser.add_argument('--data-file', type=str, default='', help='Override the default .npz filename')
parser.add_argument('--preload-tuning', action='store_true', help='Run hyperparameter tuning before the data file is loaded')
parser.add_argument('--preload-only', action='store_true', help='Run the preload tuning step and exit before loading data')
parser.add_argument('--tuning-trials', type=int, default=8, help='Number of synthetic tuning candidates to test')
parser.add_argument('--tuning-samples', type=int, default=1024, help='Synthetic samples used for pre-load tuning')
parser.add_argument('--tuning-output', type=str, default='', help='Optional path for a tuning summary JSON file')
parser.add_argument('--seed', type=int, default=42, help='Random seed used for the synthetic tuning data')
ARGS, _UNKNOWN_ARGS = parser.parse_known_args()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mCTHth=3000
stdCTHth=1000



# base_featers_list=['SZA','VZA','RAA','0.47', '0.66', '0.86', '0.905', '1.375', '1.24','2.13','3.75', 
#               '3.96', '4.05', '4.46', '4.5', '6.7', '7.3', '8.5', '9.7', '11.0', '12.0', '13.33', '13.63', '13.93', '14.23']
# base_cross_featers_list=['SWIRsRatioA','midIRsst','WVz','Ozone','CO2','SWIRsRatioB','SO2']

visls=[0.47,0.55,0.66,0.86]
nirls=[0.905,0.935,0.940]
swirls=[1.24,1.64,2.13,1.375]
thermalls=np.array([3.75,3.961,3.96,4.05,4.46,4.5,6.7,7.3,8.5,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
solarls=np.concatenate((visls,nirls,swirls),axis=0)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Flags and parameters &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

wavelns_list0=['0.47','0.66','0.86','1.24','1.375','2.13','3.96','4.05','4.5', '6.7', '7.3', '8.5', '9.7','11.0','12.0', '13.33','14.23']
moments_list=['m','sk']# ['m','s','sk','k']

# featers_list=['SZA','VZA','RAA','0.47','0.935','1.24','1.375','3.96','4.05','4.5', '6.7', '7.3', '8.5', '9.7','11.0','12.0', '13.33','14.23']
# featers_list=['SZA','VZA','RAA','0.47','0.935','1.24','1.375','8.5', '9.7','11.0','12.0']


# YEAR=['2014']#,'2015']
# REGION=['Pacific'] # 'Atlantic']#
# MONTH=['june','sep']#,'mrch', 'sep','dec'] #,
fname='mrch_2014_Pacific.npz'

# datapath=r'C:/Users/eeytan/Documents/Projects/F1km'
# fname=datapath+'/DFs/'+fname

DEFAULT_DATAPATH=r'C:\Users\eeytan\OneDrive - UCB-O365\Documents\New stuff after mv to ahrddrive\projects\F1km'
datapath=ARGS.data_path if ARGS.data_path else DEFAULT_DATAPATH
fname=ARGS.data_file if ARGS.data_file else fname
fname=os.path.join(datapath, fname)

nmextension='3LY_Pac_mrch_tst'
SolarNorm='0.47'
MixedNorm='4.05'
ThermalNorm='11.0'
# Nmoms=4  # number of moments to use

# momsnms=['m','s','sk','k']
featers_list=['SZA','VZA','RAA']
for i in range(0,len(moments_list)):   
    featers_list=featers_list+[moments_list[i]+tmp for tmp in wavelns_list0]

AODlim=0  # limit learning to spesific AOD
# The following variables are the hyperparameters.
learning_rate = 0.001
Nepochs = 200
batch_size = 5000
internal_test=True
trainprec=0.7  # HOW MUCH OF THE DATA TAKE FOR TRAINING (THE REST IS FOR TESTING) 
validation_split = 0.05 # Split the original training set into a reduced training set and a validation set. 
linearfit_split = 0  # if >0 take 1 minus that fraction (1-linearfit_split) from the test dataset (1-trainprec) to do linear fit correction
loss_square= "mean_squared_error" #  "mean_absolute_error" #
activationf='relu'  # 'tanh' # 'sigmoid' #
Nfeaters1=round(len(featers_list)/2)
Nfeaters2=0 #round(len(featers_list)/4)
Nfeaters3=0 #round(len(featers_list)/8)
Nfeaters4=0

errorprec=False # present error in (%) or (Wm^-2)
linreg=0 # 1- linear regression | 0- ANN model | 2- do both and compare error
# crossfeat=False
cross_featers_list=[]
Flabel=True
SWf=True

if SWf:
    lim=[50,850]
else:
    lim=[100,340]

dl=lim[1]-lim[0]    
dxtxt=1/35*dl
sytxt=2/35*dl





base_column_names = ['SZA','VZA','RAA','0.47', '0.55', '0.66', '0.86', '0.905', '0.935', '0.940', '1.24', '1.64','2.13', '1.375', '3.75', '3.96a', '3.96', '4.05', '4.46', '4.5', '6.7', '7.3', '8.5', '9.7', '11.0', '12.0', '13.33', '13.63', '13.93', '14.23']
# base_column_names = ['SZA','VZA','RAA',
                     # 'm0.47', 'm0.55', 'm0.66', 'm0.86', 'm0.905', 'm0.935', 'm0.940', 'm1.24', 'm1.64','m2.13', 'm1.375', 'm3.75', 'm3.96a', 'm3.96', 'm4.05', 'm4.46', 'm4.5', 'm6.7', 'm7.3', '8.5', '9.7', '11.0', '12.0', '13.33', '13.63', '13.93', '14.23']
mcols=['m'+tmp for tmp in base_column_names[3:]]
stdcols=['s'+tmp for tmp in base_column_names[3:]]
skewcols=['sk'+tmp for tmp in base_column_names[3:]]
kurtcols=['k'+tmp for tmp in base_column_names[3:]]
base_column_names=['SZA','VZA','RAA']+mcols+stdcols+skewcols+kurtcols

normlist=[SolarNorm,ThermalNorm,MixedNorm]

# %%
#---------------------- title Define the plotting function. ---------------------------------------
def plot_the_loss_curve(epochs, mse_training, mse_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse_training, label="Training Loss")
  plt.plot(epochs, mse_validation, label="Validation Loss")
  
  # mse_training is a pandas Series, so convert it to a list first.
  merged_mse_lists = mse_training.tolist() + mse_validation
  highest_loss = max(merged_mse_lists)
  lowest_loss = min(merged_mse_lists)
  top_of_y_axis = highest_loss * 1.03
  bottom_of_y_axis = lowest_loss * 0.97 

  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.legend()
  plt.show()  

print("Defined the plot_the_loss_curve function.")


def build_preload_candidate_space(feature_count):
    """Build a compact hyperparameter search space for the synthetic pre-load tuning step."""

    base_width = max(8, round(feature_count / 2))
    return [
            {'learning_rate': 0.001, 'batch_size': 64, 'validation_split': 0.10, 'activationf': 'relu', 'Nfeaters1': base_width, 'Nfeaters2': 0, 'Nfeaters3': 0, 'Nfeaters4': 0},
            {'learning_rate': 0.0005, 'batch_size': 128, 'validation_split': 0.10, 'activationf': 'relu', 'Nfeaters1': base_width * 2, 'Nfeaters2': 0, 'Nfeaters3': 0, 'Nfeaters4': 0},
            {'learning_rate': 0.001, 'batch_size': 64, 'validation_split': 0.05, 'activationf': 'tanh', 'Nfeaters1': base_width, 'Nfeaters2': base_width // 2, 'Nfeaters3': 0, 'Nfeaters4': 0},
            {'learning_rate': 0.0005, 'batch_size': 32, 'validation_split': 0.05, 'activationf': 'relu', 'Nfeaters1': base_width * 2, 'Nfeaters2': base_width, 'Nfeaters3': base_width // 2, 'Nfeaters4': 0},
            {'learning_rate': 0.002, 'batch_size': 128, 'validation_split': 0.10, 'activationf': 'relu', 'Nfeaters1': base_width, 'Nfeaters2': 0, 'Nfeaters3': 0, 'Nfeaters4': 0},
            {'learning_rate': 0.001, 'batch_size': 256, 'validation_split': 0.15, 'activationf': 'tanh', 'Nfeaters1': base_width * 2, 'Nfeaters2': base_width, 'Nfeaters3': 0, 'Nfeaters4': 0},
            {'learning_rate': 0.0003, 'batch_size': 64, 'validation_split': 0.10, 'activationf': 'relu', 'Nfeaters1': base_width * 2, 'Nfeaters2': base_width, 'Nfeaters3': base_width // 2, 'Nfeaters4': base_width // 4},
            {'learning_rate': 0.001, 'batch_size': 32, 'validation_split': 0.05, 'activationf': 'sigmoid', 'Nfeaters1': base_width, 'Nfeaters2': 0, 'Nfeaters3': 0, 'Nfeaters4': 0},
    ][:max(1, min(ARGS.tuning_trials, 8))]


def run_preload_hyperparameter_tuning(feature_count):
    """Search a small ANN hyperparameter set before the main data file is loaded."""

    synthetic_rng = np.random.default_rng(ARGS.seed)
    sample_count = max(64, int(ARGS.tuning_samples))
    x = synthetic_rng.normal(size=(sample_count, feature_count)).astype(np.float32)
    y = (
            0.65 * x[:, 0]
            - 0.25 * np.square(x[:, 1])
            + 0.20 * np.sin(x[:, 2])
            + 0.10 * x[:, 3] * x[:, 4]
            + synthetic_rng.normal(scale=0.5, size=sample_count)
    ).astype(np.float32)

    split_point = int(np.ceil(0.8 * len(x)))
    x_train = x[:split_point]
    x_val = x[split_point:]
    y_train = y[:split_point]
    y_val = y[split_point:]

    best_score = None
    best_config = None

    for config in build_preload_candidate_space(feature_count):
        tf.keras.backend.clear_session()
        tuning_inputs = tf.keras.layers.Input(shape=(feature_count,), dtype=tf.float32, name='tuning_input')
        tuning_norm = tf.keras.layers.Normalization(axis=-1, name='tuning_norm')
        tuning_norm.adapt(x_train)
        tuned_x = tuning_norm(tuning_inputs)

        tuned = tf.keras.layers.Dense(units=config['Nfeaters1'], activation=config['activationf'], name='tuning_dense_1')(tuned_x)
        if config['Nfeaters2'] > 0:
            tuned = tf.keras.layers.Dense(units=config['Nfeaters2'], activation=config['activationf'], name='tuning_dense_2')(tuned)
        if config['Nfeaters3'] > 0:
            tuned = tf.keras.layers.Dense(units=config['Nfeaters3'], activation=config['activationf'], name='tuning_dense_3')(tuned)
        if config['Nfeaters4'] > 0:
            tuned = tf.keras.layers.Dense(units=config['Nfeaters4'], activation=config['activationf'], name='tuning_dense_4')(tuned)
        tuned = tf.keras.layers.Dense(units=1, name='tuning_output')(tuned)

        tuning_model = tf.keras.Model(inputs=tuning_inputs, outputs=tuned)
        tuning_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                loss=loss_square,
                metrics=[tf.keras.metrics.MeanSquaredError() if loss_square == 'mean_squared_error' else tf.keras.metrics.MeanAbsoluteError()],
        )
        tuning_history = tuning_model.fit(
                x_train,
                y_train,
                epochs=min(20, Nepochs),
                batch_size=config['batch_size'],
                validation_data=(x_val, y_val),
                verbose=0,
                shuffle=True,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
        )

        score = float(np.min(tuning_history.history['val_loss']))
        if best_score is None or score < best_score:
            best_score = score
            best_config = config

    if best_config is None:
        return {}

    print('##### Preload hyperparameter tuning selected:', best_config)
    return best_config


def apply_preload_tuning_results(best_config):
    """Apply the synthetic tuning result to the global ANN hyperparameters."""

    global learning_rate, batch_size, validation_split, activationf
    global Nfeaters1, Nfeaters2, Nfeaters3, Nfeaters4

    if not best_config:
        return

    learning_rate = best_config['learning_rate']
    batch_size = best_config['batch_size']
    validation_split = best_config['validation_split']
    activationf = best_config['activationf']
    Nfeaters1 = best_config['Nfeaters1']
    Nfeaters2 = best_config['Nfeaters2']
    Nfeaters3 = best_config['Nfeaters3']
    Nfeaters4 = best_config['Nfeaters4']


def _resolve_test_data_root(base_path):
    """Find a directory that contains moments/ and scalars/ NPZ subfolders."""

    candidates = [
            base_path,
            os.path.join(base_path, 'test') if base_path else '',
            os.path.join(os.getcwd(), 'test'),
            os.path.join(os.getcwd(), 'Data_toGuy', 'test'),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        moments_dir = os.path.join(candidate, 'moments')
        scalars_dir = os.path.join(candidate, 'scalars')
        if os.path.isdir(moments_dir) and os.path.isdir(scalars_dir):
            return candidate
    return ''


def _build_dataframe_from_npz_pairs(data_root, wavelength_names, sw_mode=True):
    """Build the expected training dataframe from moments/scalars NPZ pairs."""

    moments_dir = os.path.join(data_root, 'moments')
    scalars_dir = os.path.join(data_root, 'scalars')
    moment_files = sorted([fname for fname in os.listdir(moments_dir) if fname.endswith('.npz')])

    rows = []
    used_pairs = 0
    for moment_file in moment_files:
        moment_path = os.path.join(moments_dir, moment_file)
        scalar_path = os.path.join(scalars_dir, moment_file)
        if not os.path.exists(scalar_path):
            continue

        with np.load(moment_path, allow_pickle=True) as moments_data, np.load(scalar_path, allow_pickle=True) as scalars_data:
            sw_wavelengths = np.asarray(moments_data['SWwavlngs'], dtype=np.float32)
            lw_wavelengths = np.asarray(moments_data['LWwavlngs'], dtype=np.float32)
            all_wavelengths = np.concatenate((sw_wavelengths, lw_wavelengths), axis=0)

            mean_spec = np.concatenate((moments_data['mSWspec'], moments_data['mLWspec']), axis=1)
            skew_spec = np.concatenate((moments_data['skewSWspec'], moments_data['skewLWspec']), axis=1)

            sza = np.asarray(scalars_data['SZA'], dtype=np.float32)
            vza = np.asarray(scalars_data['VZA'], dtype=np.float32)
            raa = np.asarray(scalars_data['RAA'], dtype=np.float32)
            mean_cth = np.asarray(scalars_data['meanCTH'], dtype=np.float32)
            std_cth = np.asarray(scalars_data['stdCTH'], dtype=np.float32)
            flux = np.asarray(scalars_data['Fsw' if sw_mode else 'Flw'], dtype=np.float32)

            target_wavelengths = np.asarray([float(wl) for wl in wavelength_names], dtype=np.float32)
            nearest_indices = [int(np.argmin(np.abs(all_wavelengths - wl))) for wl in target_wavelengths]

            sample_count = min(len(sza), len(vza), len(raa), len(mean_cth), len(std_cth), len(flux), mean_spec.shape[0], skew_spec.shape[0])
            for row_idx in range(sample_count):
                row = {
                        'SZA': float(sza[row_idx]),
                        'VZA': float(vza[row_idx]),
                        'RAA': float(raa[row_idx]),
                        'mCTH': float(mean_cth[row_idx]),
                        'stdCTH': float(std_cth[row_idx]),
                        'Flux': float(flux[row_idx]),
                }

                for wl_name, channel_idx in zip(wavelength_names, nearest_indices):
                    row['m' + wl_name] = float(mean_spec[row_idx, channel_idx])
                    row['sk' + wl_name] = float(skew_spec[row_idx, channel_idx])
                rows.append(row)

            used_pairs += 1

    if not rows:
        raise ValueError('No valid moments/scalars NPZ pairs were found under ' + data_root)

    print('##### Built dataframe from', used_pairs, 'NPZ file pairs')
    return pd.DataFrame(rows)


preload_tuning_config = {}
if ARGS.preload_tuning:
    preload_tuning_config = run_preload_hyperparameter_tuning(len(featers_list))
    apply_preload_tuning_results(preload_tuning_config)

    tuning_summary = {
        'selected_config': preload_tuning_config,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'activationf': activationf,
        'Nfeaters1': Nfeaters1,
        'Nfeaters2': Nfeaters2,
        'Nfeaters3': Nfeaters3,
        'Nfeaters4': Nfeaters4,
    }
    tuning_output = ARGS.tuning_output if ARGS.tuning_output else os.path.join(os.getcwd(), nmextension + '_preload_tuning.json')
    try:
        os.makedirs(os.path.dirname(tuning_output), exist_ok=True)
        with open(tuning_output, 'w', encoding='utf-8') as tuning_file:
            json.dump(tuning_summary, tuning_file, indent=2)
        print('##### Preload hyperparameter tuning summary saved to', tuning_output)
    except Exception as tuning_error:
        print('##### Could not save preload tuning summary:', tuning_error)

    if ARGS.preload_only:
        print('##### Exiting after preload tuning as requested')
        sys.exit(0)

# ------------------- Define functions to create and train a linear regression model ----------------------------------------
def create_model(my_inputs, my_outputs, my_learning_rate):
  """Create and compile a simple linear regression model."""
  model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)
  
  if loss_square=="mean_squared_error":
  # Construct the layers into a model that TensorFlow can execute.
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.MeanSquaredError()])
      
  else:
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate), loss="mean_absolute_error", metrics=[tf.keras.metrics.MeanAbsoluteError()])

  return model


def train_model(model, dataset, epochs, batch_size, label_name, validation_split=0.1):
        """Feed a dataset into the model in order to train it."""

        # Split the dataset into features and label.
        features = {name: np.array(value) for name, value in dataset.items()}
        label_raw = np.array(features.pop(label_name), dtype=np.float32).reshape(-1, 1)
        label = train_label(label_raw)
        history = model.fit(
                        x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_split=validation_split,
        )

        # Get details that will be useful for plotting the loss curve.
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        mse = hist[loss_square]

        return epochs, mse, history.history

print("Defined the create_model and train_model functions.")

# ------------ Define linear regression model outputs -----------------------------------------------
def get_outputs_linear_regression():
  # Create the Dense output layer.
  dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,), name='dense_output')(preprocessing_layers)

  # Define an output dictionary we'll send to the model constructor.
  outputs = {'dense_output': dense_output}
  return outputs

# ------------ Define ANN model outputs -----------------------------------------------
def get_outputs_dnn():
  # Dense layer 1.
  regulator=tf.keras.regularizers.L1(l1=0.01)
  dense_output = tf.keras.layers.Dense(units=Nfeaters1, input_shape=(1,),
                               activation=activationf, kernel_regularizer=regulator,
                              name='hidden_dense_layer_1')(preprocessing_layers)
  if Nfeaters2>0: 
    # Dense layer 2.
    regulator=tf.keras.regularizers.L1(l1=0.001)
    dense_output = tf.keras.layers.Dense(units=Nfeaters2, input_shape=(1,),
                                   activation=activationf, kernel_regularizer=regulator,
                                   name='hidden_dense_layer_2')(dense_output)
  if Nfeaters3>0:
   # Dense layer 3.
    regulator=tf.keras.regularizers.L1(l1=0.001)
    dense_output = tf.keras.layers.Dense(units=Nfeaters3, input_shape=(1,),
                                    activation=activationf,kernel_regularizer=regulator,
                                    name='hidden_dense_layer_3')(dense_output)
  if Nfeaters4>0:
   # Dense layer 4.
    regulator=tf.keras.regularizers.L1(l1=0.001)
    dense_output = tf.keras.layers.Dense(units=Nfeaters4, input_shape=(1,),
                                    activation=activationf,kernel_regularizer=regulator,
                                    name='hidden_dense_layer_4')(dense_output)

 # Dense output layer.
  dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,),
                                    name='dense_output')(dense_output)

 # Define an output dictionary we'll send to the model constructor.
  outputs = {'dense_output': dense_output}

  return outputs


# %%
# count=0
# for Iyear in YEAR:
#     for Iregion in REGION:
#         for Imonth in MONTH:
#             print(Iyear,Iregion,Imonth)
#             count=count+1
#             tmppath=datapath+r"/"+Iyear+"/"+Iregion+"/"+Imonth+"/"
            
#             if count==1:
#                 df,column_names,label_normalization_factor,SWnorm,mixnorm,LWnorm,Norms,df_mean,df_std=make_df(tmppath,Flabel,SWf,featers_list,cross_featers_list,normlist)
#             else:
#                 tmpdf,column_names,label_normalization_factor,SWnorm,mixnorm,LWnorm,Norms,df_mean,df_std=make_df(tmppath,Flabel,SWf,featers_list,cross_featers_list,normlist)
#                 df=df.append(tmpdf)

legacy_ready = False
if os.path.exists(fname):
    with np.load(fname, allow_pickle=True) as dat:
        needed_keys = ['column_names', 'label_normalization_factor', 'SWnorm', 'MIXnorm', 'LWnorm', 'Norms', 'df_mean', 'df_std']
        if all(key in dat.files for key in needed_keys):
            legacy_ready = True

if legacy_ready and os.path.exists(fname[:-3] + 'pkl'):
    with np.load(fname, allow_pickle=True) as dat:
        column_names=dat['column_names']
        label_normalization_factor=dat['label_normalization_factor']
        SWnorm=dat['SWnorm']
        MIXnorm=dat['MIXnorm']
        LWnorm=dat['LWnorm']
        Norms=dat['Norms']
        df_mean=dat['df_mean']
        df_std=dat['df_std']
    name=fname[:-3]+'pkl'
    df = pd.read_pickle(name)
else:
    test_data_root = _resolve_test_data_root(datapath)
    if not test_data_root:
        raise FileNotFoundError('Could not locate test moments/scalars folders. Set --data-path to a folder containing moments/ and scalars/.')

    datapath = test_data_root
    df = _build_dataframe_from_npz_pairs(test_data_root, wavelns_list0, SWf)
    column_names = list(df.columns)
    label_normalization_factor = 1.0
    SWnorm = np.array([1.0], dtype=np.float32)
    MIXnorm = np.array([1.0], dtype=np.float32)
    LWnorm = np.array([1.0], dtype=np.float32)
    Norms = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    df_mean = df[featers_list].mean().to_numpy(dtype=np.float32)
    df_std = df[featers_list].replace([np.inf, -np.inf], np.nan).std().fillna(0.0).to_numpy(dtype=np.float32)

# df = df[:2000000] 
# print('################## NOTE ############  DF cut by half')

# Filter rows based on a column condition
df = df[ (df['mCTH'] < mCTHth) & (df['stdCTH'] < stdCTHth) ]

# Keep only model features and label after cloud-height filtering.
required_columns = featers_list + ['Flux']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError('Missing required columns in dataframe: ' + ', '.join(missing_columns))
df = df[required_columns]
column_names=featers_list  #np.concatenate((column_names[0:3], column_names[9:]))

print('##### Number of FOVs', len(df)) 

# plt.figure()
# tw.plthist(df['VZA']*60)
# input('stop')
# if SolarNorm==0:
#     for ch in featers_list[3:]:
#         tmp=float(ch)
#         if tmp<3.0:
#             df[ch]=df[ch]/df[ch].quantile(0.99)
# if MixedNorm==0:
#     for ch in featers_list[3:]:
#         tmp=float(ch)
#         if tmp>3.0 and tmp<5:
#             df[ch]=df[ch]/df[ch].quantile(0.99)
# if ThermalNorm==0:
#     for ch in featers_list[3:]:
#         tmp=float(ch)
#         if tmp>5.0:
#             df[ch]=df[ch]/df[ch].quantile(0.99) 


# df = df.sample(frac=1).reset_index(drop=True)
# df=df.dropna() 

df = df.sample(frac=1.0, random_state=ARGS.seed).reset_index(drop=True)
split_idx = int(np.ceil(trainprec * len(df)))
split_idx = max(1, min(split_idx, len(df) - 1))
df_train=df.iloc[0:split_idx,:]
df_test=df.iloc[split_idx:,:]

if len(df_train) < 2 or len(df_test) < 1:
    raise ValueError('Not enough samples after preprocessing. Need at least 3 rows to train/test split.')

batch_size = max(1, min(batch_size, len(df_train)))
max_val_split = (len(df_train) - 1) / len(df_train)
validation_split = min(validation_split, max(0.0, max_val_split - 1e-6))

lentest=len(df_test)
if linearfit_split>0:
    df_test_lincorr=df_test.iloc[int(np.ceil(linearfit_split*lentest))+1:,:]
    df_test=df_test.iloc[0:int(np.ceil(linearfit_split*lentest)),:]
# %% START ANN
# Keras Input tensors of float values.
inputs=[1]*(df.shape[1]-1)
inputs_list=[1]*(df.shape[1]-1)
for i in range(0,df.shape[1]-1):
    inputs_list[i]= (tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=column_names[i])) 
    inputs[i]= (column_names[i], tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=column_names[i])) 
inputs_dic=dict(inputs)

# # Create a Normalization layer to normalize the features data.
norm_inputs=[1]*(df.shape[1]-1)
for i in range(0,df.shape[1]-1):
    print('normalizing layer',i)
    tmpstr='normalization_'+column_names[i]
    tmp = tf.keras.layers.Normalization(name=tmpstr, axis=None)
    tmp.adapt(np.array(df_train[column_names[i]], dtype=np.float32).reshape(-1, 1))
    norm_inputs[i] = tmp(inputs_dic.get(column_names[i]))


# Concatenate our inputs into a single tensor.
preprocessing_layers = tf.keras.layers.Concatenate()(norm_inputs)
print(preprocessing_layers)
print("Preprocessing layers defined.")

# *********************************************************************************************

# Create Normalization layers to normalize the Flux data. Because Flux is our label, these layers won't be added to our model.
train_label= tf.keras.layers.Normalization(axis=None)#,invert=True)
train_label.adapt(np.array(df_train['Flux'], dtype=np.float32).reshape(-1, 1))
test_label = tf.keras.layers.Normalization(axis=None)#,invert=True)
test_label.adapt(np.array(df_test['Flux'], dtype=np.float32).reshape(-1, 1))


if internal_test:
    df_Input=df_test.copy()
    df_Input.drop('Flux',inplace=True, axis=1)
    array_Input=np.array(df_Input)
    
    df_label=df_test['Flux']
    label= np.array(df_label)
    
    if SWf:
        np.savez(datapath+r'/SW_norm_factors_'+nmextension+'.npz',std=np.nanstd(label),mean=np.nanmean(label),norm_factor=label_normalization_factor,DFmean=df_mean,DFstd=df_std,AnglesNorms=Norms,SWnorms=SWnorm,LWnorms=LWnorm,MIXnorms=MIXnorm,Normlist=normlist)
    else:
        np.savez(datapath+r'/LW_norm_factors_'+nmextension+'.npz',std=np.nanstd(label),mean=np.nanmean(label),norm_factor=label_normalization_factor,DFmean=df_mean,DFstd=df_std,AnglesNorms=Norms,SWnorms=SWnorm,LWnorms=LWnorm,MIXnorms=MIXnorm,Normlist=normlist)

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
    
    # Establish the model's topography.
    Lin_model = create_model(inputs_dic, outputs, learning_rate)
    
    # Train the model on the normalized training set.
    epochs, mse, history = train_model(Lin_model, df_train, Nepochs, batch_size, label_name, validation_split)
    
    # plot_the_loss_curve(epochs, mse, history["val_mean_squared_error"])
    
    test_features = {name:np.array(value) for name, value in df_test.items()}
    lin_test_label = test_label(np.array(test_features.pop(label_name), dtype=np.float32).reshape(-1, 1)) # isolate the label
    print("\n Evaluate the linear regression model against the test set:")
    Score=Lin_model.evaluate(x = test_features, y = lin_test_label, batch_size=batch_size, return_dict=True)
    
    if SWf:
        Lin_model.save(datapath+"/Linear_model_SW_"+nmextension+".keras")
    else:
        Lin_model.save(datapath+"/Linear_model_LW_"+nmextension+".keras")
    
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
    
    Lin_weights=Lin_model.get_weights()[(df.shape[1] - 1) * 3]
    plt.figure()
    plt.set_cmap('bwr')
    ax=plt.subplot(1,1,1)    
    vmin = np.nanmin(Lin_weights) 
    vmax = np.nanmax(Lin_weights)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.pcolor(Lin_weights, norm=norm) 
    plt.colorbar()
    ax.set_ylabel('Input featers')
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
    ax.set_yticklabels(featers_list+cross_featers_list)
    ax.axes.set_aspect('equal')
    
    if internal_test:
        test_features = {name:np.array(value) for name, value in df_Input.items()}  
        Lin_Yhat=Lin_model.predict(test_features) #(array_Input[:,0],array_Input[:,1],array_Input[:,2],array_Input[:,3],array_Input[:,4],array_Input[:,5],array_Input[:,6],array_Input[:,7],array_Input[:,8],array_Input[:,9],array_Input[:,10],array_Input[:,11],array_Input[:,12],array_Input[:,13],array_Input[:,14],array_Input[:,15],array_Input[:,16],array_Input[:,17],array_Input[:,18],array_Input[:,19],array_Input[:,20]))
        Lin_Yhat=Lin_Yhat['dense_output']
        Lin_Yhat=(np.nanstd(label)*Lin_Yhat+np.nanmean(label))*label_normalization_factor  # de-normalizing the normalized label distribution (sigam*Y+mu)*normalization_factor   
        # Linear fitting to correct
        tmp0=np.reshape(LABEL_test,np.shape(Lin_Yhat))
        Lin_reg = LinearRegression().fit(tmp0,Lin_Yhat)
        Lin_R2=Lin_reg.score(tmp0,Lin_Yhat)
        Lin_a=Lin_reg.coef_[0]
        Lin_b=Lin_reg.intercept_
        Lin_txt='Y='+str("{0:2.1f}".format(Lin_a[0]))+'X+'+str("{0:2.2f}".format(Lin_b[0]))
        Lin_Ytag=Lin_reg.predict(Lin_Yhat)
        Lin_Yhat=np.squeeze(Lin_Yhat)
        Lin_Ytag=np.squeeze(Lin_Ytag)
# ---------------------- ANN -------------------------------------------------------------------------------
if linreg!=1:


    # Specify the label
    label_name = "Flux"


    dnn_outputs = get_outputs_dnn()
    
    # Establish the model's topography.
    ANN_model = create_model(inputs_dic,dnn_outputs,learning_rate)
    
    # Train the model on the normalized training set. We're passing the entire 
    # normalized training set, but the model will only use the features
    # defined in our inputs.
    epochs, mse, history = train_model(ANN_model, df_train, Nepochs, batch_size, label_name, validation_split)
    
    # plot_the_loss_curve(epochs, mse, history["val_mean_squared_error"])
    
    # After building a model against the training set, test that model against the test set.
    test_features = {name:np.array(value) for name, value in df_test.items()}
    ANN_test_label = test_label(np.array(test_features.pop(label_name), dtype=np.float32).reshape(-1, 1)) # isolate the label
    print("\n Evaluate the new model against the test set:")
    Score=ANN_model.evaluate(x = test_features, y = ANN_test_label, batch_size=batch_size, return_dict=True)
    
    if SWf:
        ANN_model.save(datapath+"/ANN_model_SW_"+nmextension+".keras")
    else:
        ANN_model.save(datapath+"/ANN_model_LW_"+nmextension+".keras")
    
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
    
    Nfeaturestot=Nfeaters1*Nfeaters2+Nfeaters2*Nfeaters3+Nfeaters3*Nfeaters4
    
    weights=ANN_model.get_weights()[(df.shape[1] - 1) * 3 ]  #[len(featers_list+cross_featers_list)*3]
    weights_out=ANN_model.get_weights()[(df.shape[1] - 1) * 3 +2] #[len(featers_list+cross_featers_list)*3+2]
    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2,  width_ratios=(4,1), height_ratios=(1,8),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    # ax.axes.set_aspect('equal')
    ax_out = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_out.tick_params(axis="x", labelbottom=False)
    # ax_out.axes.set_aspect('equal')
    
    plt.set_cmap('bwr')
    plt.title('ANN Regression weights')
    vmin = np.nanmin(weights) 
    vmax = np.nanmax(weights)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    main=ax.pcolor(weights, norm=norm) 
    plt.colorbar(main)
    ax.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
    ax.set_yticklabels(featers_list+cross_featers_list)
    ax.set_xticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
    ax.set_xticklabels(np.arange(1,len(featers_list+cross_featers_list)+1,1))
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Input featers')
    
    plt.title('ANN Regression weights')
    vmin = np.nanmin(weights_out) 
    vmax = np.nanmax(weights_out)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    outs=ax_out.pcolor(weights_out.T, norm=norm) 
    plt.colorbar(outs)
    ax_out.set_yticklabels([])
    ax_out.set_yticks([])
      
    
    # plt.figure()
    # plt.set_cmap('bwr') 
    # ax=plt.subplot(1,1,1)
    # # plt.pcolor(weights,cmap=mpl.colormaps['bwr'])
    # plt.title('ANN Regression weights')
    # vmin = np.nanmin(weights) 
    # vmax = np.nanmax(weights)
    # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # plt.pcolor(weights, norm=norm) 
    # plt.colorbar()
    # # ax.set_xticklabels([])
    # # ax.set_xticks([])
    # ax.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
    # ax.set_yticklabels(featers_list+cross_featers_list)
    # ax.axes.set_aspect('equal')
    
    
    if internal_test:
        test_features = {name:np.array(value) for name, value in df_Input.items()}  
        Yhat=ANN_model.predict(test_features) #(array_Input[:,0],array_Input[:,1],array_Input[:,2],array_Input[:,3],array_Input[:,4],array_Input[:,5],array_Input[:,6],array_Input[:,7],array_Input[:,8],array_Input[:,9],array_Input[:,10],array_Input[:,11],array_Input[:,12],array_Input[:,13],array_Input[:,14],array_Input[:,15],array_Input[:,16],array_Input[:,17],array_Input[:,18],array_Input[:,19],array_Input[:,20]))
        Yhat=Yhat['dense_output']
        Yhat=(np.nanstd(label)*Yhat+np.nanmean(label))*label_normalization_factor  # de-normalizing the normalized label distribution (sigam*Y+mu)*normalization_factor
        
        
            # Linear fitting to correct
        tmp0=np.reshape(LABEL_test,np.shape(Yhat))
        reg = LinearRegression().fit(tmp0,Yhat)
        R2=reg.score(tmp0,Yhat)
        a=reg.coef_[0]
        b=reg.intercept_
        txt='Y='+str("{0:2.2f}".format(a[0]))+'X+'+str("{0:2.2f}".format(b[0]))
        if linearfit_split:
            test_features_lincorr = {name:np.array(value) for name, value in df_Input_lincorr.items()}  
            Yhat_lincorr=ANN_model.predict(test_features_lincorr) #(array_Input[:,0],array_Input[:,1],array_Input[:,2],array_Input[:,3],array_Input[:,4],array_Input[:,5],array_Input[:,6],array_Input[:,7],array_Input[:,8],array_Input[:,9],array_Input[:,10],array_Input[:,11],array_Input[:,12],array_Input[:,13],array_Input[:,14],array_Input[:,15],array_Input[:,16],array_Input[:,17],array_Input[:,18],array_Input[:,19],array_Input[:,20]))
            Yhat_lincorr=Yhat_lincorr['dense_output']
            Yhat_lincorr=(np.std(label)*Yhat_lincorr+np.nanmean(label))*label_normalization_factor
            Ytag=reg.predict(Yhat_lincorr)
            Yhat_lincorr=np.squeeze(Yhat_lincorr)
            Ytag=np.squeeze(Ytag)
        Yhat=np.squeeze(Yhat)  # align dimentions with LABEL_test
        
if linreg>1:
   dnormMSE=normMSE-lin_normMSE
   dRMSE=RMSE-lin_RMSE
   print('\n')
   print('########### ANN - LINEAR REGRESSION ###########')
   print('delta normalized MSE= ',dnormMSE)
   print('delta RMSE= ',dRMSE, ' W/m^2')
   print('##########################') 

# ^^^^^^^ subpltos weights together
   # plt.figure()
   # ax1=plt.subplot(1,2,1)
   # # gs = fig.add_gridspec(2, 2,  width_ratios=(4,1), height_ratios=(1,8),
   # #                       left=0.1, right=0.9, bottom=0.1, top=0.9,
   # #                       wspace=0.05, hspace=0.05)
   # # # Create the Axes.
   # # ax = fig.add_subplot(gs[1, 0])
   # # # ax.axes.set_aspect('equal')
   # # ax_out = fig.add_subplot(gs[0, 0], sharex=ax)
   # # ax_out.tick_params(axis="x", labelbottom=False)
   # # # ax_out.axes.set_aspect('equal')
   # #  weights=ANN_model.get_weights()[len(featers_list+cross_featers_list)*3]
   # #  weights_out=ANN_model.get_weights()[len(featers_list+cross_featers_list)*3+2]  
   # #  plt.set_cmap('bwr')
   # #  plt.title('ANN Regression weights')
   # #  vmin = np.nanmin(weights) 
   # #  vmax = np.nanmax(weights)
   # #  norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
   # #  main=ax.pcolor(weights, norm=norm) 
   # #  plt.colorbar(main)
   # #  ax.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
   # #  ax.set_yticklabels(featers_list+cross_featers_list)
   # #  ax.set_xticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
   # #  ax.set_xticklabels(np.arange(1,len(featers_list+cross_featers_list)+1,1))
   # #  ax.set_xlabel('Neurons')
   # # # ax.set_ylabel('Input featers')
   
   # plt.title('ANN Regression weights')
   # vmin = np.nanmin(weights_out) 
   # vmax = np.nanmax(weights_out)
   # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
   # outs=ax_out.pcolor(weights_out.T, norm=norm) 
   # plt.colorbar(outs)
   # ax_out.set_yticklabels([])
   # ax_out.set_yticks([])
   # # plt.pcolor(weights,cmap=mpl.colormaps['bwr'])
   # plt.title('ANN Regression weights')
   # vmin = np.nanmin(weights) 
   # vmax = np.nanmax(weights)
   # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
   # plt.pcolor(weights, norm=norm) 
   # plt.colorbar()
   # # ax.set_xticklabels([])
   # # ax.set_xticks([])
   # # ax1.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
   # ax1.set_yticklabels(featers_list+cross_featers_list)
   # ax1.axes.set_aspect('equal')
   # ax2=plt.subplot(1,2,2)
   # # plt.pcolor(weights,cmap=mpl.colormaps['bwr'])
   # plt.title('Linear Regression weights')
   # vmin = np.nanmin(Lin_weights) 
   # vmax = np.nanmax(Lin_weights)
   # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
   # plt.pcolor(Lin_weights, norm=norm) 
   # plt.colorbar()
   # ax2.set_xticklabels([])
   # ax2.set_xticks([])
   # ax2.set_yticks(np.arange(0.5,len(featers_list+cross_featers_list)+0.5))
   # ax2.set_yticklabels(featers_list+cross_featers_list)
   # ax2.axes.set_aspect('equal')
  

    
#%%   
 
if internal_test:   
# $$$$$$$$$$$$$$$$$$$$$$$$ FIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    if linreg!=1:
        fig = plt.figure(figsize=(6, 6))    
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        # Draw the scatter plot and marginals.
        # scatter_hist(Yhat, LABEL,Ytag, ax, ax_histx)#, ax_histy)
        
        ax_histx.tick_params(axis="x", labelbottom=False)
        # ax_histy.tick_params(axis="y", labelleft=False)
        
        # the scatter plot:
        ax.scatter(LABEL_test,Yhat,label='ANN vs. Truth')
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
        # rmse_perM=np.sqrt(np.nanmean((Yhat-np.where((LABEL_test>perL)&(LABEL_test<perH),LABEL_test,np.nan))**2/error_factor))
        if linearfit_split:
            ANN_MAE_lincorr=np.nanmean(np.abs(Ytag-LABEL_test_lincorr)/error_factor)
        
        txtdy=20
        fontsz=25
        ax.text(lim[0]+dxtxt, lim[1]-txtdy, '$RMSE (Wm^{-2})$= '+ str("{0:2.2f}".format(ANN_RMSE)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*2, '$MAE (Wm^{-2})$= '+ str("{0:2.2f}".format(ANN_MAE)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*3, '$ME (Wm^{-2})$= '+ str("{0:2.2f}".format(ANN_ME)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*4, '$R^2 $= '+ str("{0:2.2f}".format(ANN_R2)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*5, '$MAE (\%)$= '+ str("{0:2.2f}".format(ANN_MAE_perc)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*6, '$MAE_{10\%} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perL)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*7, '$MAE_{center} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perM)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*8, '$MAE_{90\%} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perH)), fontsize = fontsz)
        if linearfit_split:
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*9, '$MAE_{linear-correction} (Wm^{-2})$= '+ str("{0:2.2f}".format(ANN_MAE_lincorr)), fontsize = fontsz)
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*10, txt, fontsize = fontsz,color='r')
        ax.text(lim[0]+dxtxt, lim[1]-txtdy*11, '$R^2$='+str("{0:2.2f}".format(R2)), fontsize = fontsz,color='r')
        
        ax.set_ylabel('ANN/Fit ($Wm^{-2}$)')
        ax.set_xlabel('CERES ($Wm^{-2}$)')
        ax.set_xlim((lim[0],lim[1]))
        ax.set_ylim((lim[0],lim[1]))
        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(Yhat)), np.max(np.abs(LABEL_test)))
        # lim = (int(xymax/binwidth) + 1) * binwidth
        
        bins = np.arange(lim[0], lim[1] + binwidth, binwidth)
        ax_histx.hist(Yhat, bins=bins,alpha=0.9,label='ANN')
        ax_histx.hist(LABEL_test, bins=bins,alpha=0.65,label='CERES')
        plt.plot([perL,perL],[0, 100],'--k')
        plt.plot([perH,perH],[0,100],'--k')
        if linearfit_split:
            ax_histx.hist(Ytag, bins=bins,alpha=0.4,label='Fit')
        ax_histx.legend(loc='lower right')
        ax_histx.set_ylabel('Count')
        # ax_histy.hist(y, bins=bins, orientation='horizontal')
        
        if SWf:
            np.savez(datapath+r'/SW_ANNtestarray_'+nmextension+'.npz',model=Yhat,Label=LABEL_test,RMSE=ANN_RMSE,MAE=ANN_MAE,ME=ANN_ME,R2=ANN_R2,MAEperc=ANN_MAE_perc,Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,fiteq=txt,fitR2=R2)
        else:
            np.savez(datapath+r'/LW_ANNtestarray_'+nmextension+'.npz',model=Yhat,Label=LABEL_test,RMSE=ANN_RMSE,MAE=ANN_MAE,ME=ANN_ME,R2=ANN_R2,MAEperc=ANN_MAE_perc,Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,fiteq=txt,fitR2=R2)
        
        print('########## ANN RMSE=',np.sqrt(np.nanmean((Yhat-LABEL_test)**2)), 'Wm^-2 ###############')
        
        print('^^^^^^^^^^^^^^^^^^^^ mean: [lable,test,ANN]',np.nanmean(label),np.nanmean(LABEL_test),np.nanmean(Yhat))
        print('^^^^^^^^^^^^^^^^^^^^ STD: [lable,test,ANN]',np.nanstd(label),np.nanstd(LABEL_test),np.nanstd(Yhat))
        if linearfit_split:
            print('########## corrected RMSE=',np.sqrt(np.nanmean((Ytag-LABEL_test_lincorr)**2)), 'Wm^-2 ##############')
        
    if linreg!=0:
            fig = plt.figure(figsize=(6, 6))
       
            # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
            # the size of the marginal axes and the main axes in both directions.
            # Also adjust the subplot parameters for a square plot.
            gs = fig.add_gridspec(2, 2,  width_ratios=(4,1), height_ratios=(1,4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            # Create the Axes.
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            
            # Draw the scatter plot and marginals.
            # scatter_hist(Yhat, LABEL,Ytag, ax, ax_histx)#, ax_histy)
            
            ax_histx.tick_params(axis="x", labelbottom=False)
            # ax_histy.tick_params(axis="y", labelleft=False)
            
            # the scatter plot:
            ax.scatter(LABEL_test,Lin_Yhat,label='Linear reg vs. Truth')
            # ax.scatter(Lin_Ytag, LABEL,color='g',label='Fit vs. Truth',alpha=0.2)
            ax.plot([lim[0],lim[1]],[lim[0],lim[1]],'k',label='X=Y',linewidth=3)
            # ax.plot(Lin_Yhat,Lin_Ytag,'r',label='Linear Fit',linewidth=3)
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
            # rmse_perM=np.sqrt(np.nanmean((Yhat-np.where((LABEL_test>perL)&(LABEL_test<perH),LABEL_test,np.nan))**2/error_factor))
            if linearfit_split:
                Lin_MAE_lincorr=np.nanmean(np.abs(Lin_Ytag-LABEL_test_lincorr)/error_factor)
            
            txtdy=20
            fontsz=25
            ax.text(lim[0]+dxtxt, lim[1]-txtdy, '$RMSE (Wm^{-2})$= '+ str("{0:2.2f}".format(Lin_RMSE)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*2, '$MAE (Wm^{-2})$= '+ str("{0:2.2f}".format(Lin_MAE)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*3, '$ME (Wm^{-2})$= '+ str("{0:2.2f}".format(Lin_ME)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*4, '$R^2 $= '+ str("{0:2.2f}".format(Lin_R2)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*5, '$MAE (\%)$= '+ str("{0:2.2f}".format(Lin_MAE_perc)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*6, '$MAE_{10\%} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perL)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*7, '$MAE_{center} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perM)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*8, '$MAE_{90\%} (Wm^{-2})$= '+ str("{0:2.2f}".format(err_perH)), fontsize = fontsz)
            if linearfit_split:
                ax.text(lim[0]+dxtxt, lim[1]-txtdy*9, '$MAE_{linear-correction} (Wm^{-2})$= '+ str("{0:2.2f}".format(Lin_MAE_lincorr)), fontsize = fontsz)
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*10, Lin_txt, fontsize = fontsz,color='r')
            ax.text(lim[0]+dxtxt, lim[1]-txtdy*11, '$R^2$='+str("{0:2.2f}".format(Lin_R2)), fontsize = fontsz,color='r')
            
            ax.set_ylabel('Linear reg/Fit ($Wm^{-2}$)')
            ax.set_xlabel('CERES ($Wm^{-2}$)')
            ax.set_xlim((lim[0],lim[1]))
            ax.set_ylim((lim[0],lim[1]))
            # now determine nice limits by hand:
            binwidth = 0.25
            xymax = max(np.max(np.abs(Lin_Yhat)), np.max(np.abs(LABEL_test)))
            # lim = (int(xymax/binwidth) + 1) * binwidth
            
            bins = np.arange(lim[0], lim[1] + binwidth, binwidth)
            ax_histx.hist(Lin_Yhat, bins=bins,alpha=0.9,label='Linear reg')
            ax_histx.hist(LABEL_test, bins=bins,alpha=0.65,label='CERES')
            plt.plot([perL,perL],[0, 100],'--k')
            plt.plot([perH,perH],[0,100],'--k')
            # ax_histx.hist(Lin_Ytag, bins=bins,alpha=0.4,label='Fit')
            ax_histx.legend(loc='lower right')
            ax_histx.set_ylabel('Count')
            # ax_histy.hist(y, bins=bins, orientation='horizontal')
            
            if SWf:
                np.savez(datapath+r'/SW_Lintestarray_'+nmextension+'.npz',model=Yhat,Label=LABEL_test,RMSE=Lin_RMSE,MAE=Lin_MAE,R2=Lin_R2,MAEperc=Lin_MAE_perc,Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,Lims=lim,fiteq=txt,fitR2=R2)
            else:
                np.savez(datapath+r'/LW_Lintestarray_'+nmextension+'.npz',model=Yhat,Label=LABEL_test,RMSE=Lin_RMSE,MAE=Lin_MAE,R2=Lin_R2,MAEperc=Lin_MAE_perc,Errlow=err_perL,Errmed=err_perM,Errhigh=err_perH,Lims=lim,fiteq=txt,fitR2=R2)
            
            print('########## Linear Reg RMSE=',np.sqrt(np.nanmean((Lin_Yhat-LABEL_test)**2)), 'Wm^-2 ###############')
            print('########## corrected RMSE=',np.sqrt(np.nanmean((Lin_Ytag-LABEL_test)**2)), 'Wm^-2 ##############')
        
            
            # tmp=Yhat.copy()
            # tmp=np.where(np.abs(Yhat-LABEL)<10,np.nan,tmp)
            # print('########## corrected RMSE=',np.sqrt(np.nanmean((tmp-LABEL)**2)), 'Wm^-2 ##############')
            # plt.figure()
            # tw.plthist(np.abs(Yhat-LABEL),[])
            # plt.yscale('log')
            
            
            # tmp=Yhat.copy()
            # err=tmp-LABEL
            # data=LABEL
            # big_err=np.nonzero(err>20)
            # data2=data[big_err[0],big_err[1]]            
            # count, bins_count = np.histogram(data)
              
            # # finding the PDF of the histogram using count values
            # pdf = count / sum(count)
              
            # # using numpy np.cumsum to calculate the CDF
            # # We can also find using the PDF values by looping and adding
            # cdf = np.cumsum(pdf)
              
            # # plotting PDF and CDF
            # plt.figure()
            # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
            # plt.plot(bins_count[1:], cdf, label="CDF")
            # plt.legend()
            
            
            
            
            
            
            
            
            