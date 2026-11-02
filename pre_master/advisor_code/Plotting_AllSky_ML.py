#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:32:02 2026

@author: eeytan
"""

if __name__ == '__main__':
    import datetime
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
    #import tensorflow as tf
    #from tensorflow.keras import layers
    import esh_fun_module as tw
    import matplotlib.pylab as pylab
    from matplotlib.patches import Polygon
    import matplotlib.gridspec as gridspec
    params = {'legend.fontsize': 'large','figure.figsize': (5, 5),'axes.labelsize': 'x-large','axes.titlesize':'x-large','xtick.labelsize':'x-large','ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)
    
    font = {'weight' : 'bold','size'   : 12}
    plt.rc('font', **font)

    fntsz=12
    
    
    #%%
    
    scatplot=False
    logscalef=True
    
    mainpath='/export/home/eeytan/projects/F1km/ALLSKY_newDat'
    # datapath=mainpath+'/mrchsepjunedec_2014-2015_Pacific.npz'
    
    
    modelA='Pac_allMnth_allCld_m3_Ly2_Trns1_Cost0'
    modelB='Pac_allMnth_allCld_400Nup_m3_Ly2_Trns1_Cost0'
    
    # dat=np.load(datapath)
    
    fig = plt.figure(figsize=(30, 13)) 
    outer = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.3,left=0.3, right=0.68, bottom=0.1, top=0.95)
    
    txspaceSW=60
    
    ANNpath=mainpath+'/'+modelA
    
    ANNdatsw=np.load(ANNpath+r'/SW_ANNtestarray_'+modelA+'.npz')
    
    swmodelA=ANNdatsw['model']
    swlabelA=ANNdatsw['Label']
    maxF=np.nanmax(swlabelA)
    lims=[50,850]
    
    inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                    subplot_spec=outer[0], wspace=0.1, hspace=0.1,width_ratios=(4, 1), height_ratios=(1, 4))
    lab=['CERES $F_{all-sky}^{solar} (Wm^{-2})$','model $F_{all-sky}^{solar} (Wm^{-2})$','Truth','Model']
    ANN_RMSE,ANN_MAE,ANN_ME,ANN_R2,[a,b],ax,ax_histx=tw.pltmodeltest(swmodelA,swlabelA,lims,lab,False,[fig,inner],8000,txspaceSW,scatplot,logscalef)
    ax_histx.text(-0.03, 1.1, '(a)', fontsize = fntsz,transform=ax_histx.transAxes)
    
    ax.axvspan(400, 845, alpha=0.2, color='gray')
    # ax.plot([400,400],lims,'gray',linewidth=3,alpha=0.5)
    # ax.plot([845,845],lims,'gray',linewidth=3,alpha=0.5)
    
    # **********************************************************************
    txspaceSW=40
    
    ANNpath=mainpath+'/'+modelB
    
    ANNdatsw=np.load(ANNpath+r'/SW_ANNtestarray_'+modelB+'.npz')
    
    swmodelB=ANNdatsw['model']
    swlabelB=ANNdatsw['Label']
    
    inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                    subplot_spec=outer[1], wspace=0.1, hspace=0.1,width_ratios=(4, 1), height_ratios=(1, 4))
    lab=['CERES $F_{all-sky}^{solar} (Wm^{-2})$','model $F_{all-sky}^{solar} (Wm^{-2})$','Truth','Model']
    ANN_RMSE,ANN_MAE,ANN_ME,ANN_R2,[a,b],ax,ax_histx=tw.pltmodeltest(swmodelB,swlabelB,[350,850],lab,False,[fig,inner],200,txspaceSW,scatplot,logscalef)
    ax_histx.text(-0.03, 1.1, '(b)', fontsize = fntsz,transform=ax_histx.transAxes)
    
    ax.axvspan(400, 845, alpha=0.2, color='gray')
    # ax.plot([400,400],lims,'gray',linewidth=4,alpha=0.35)
    # ax.plot([845,845],lims,'gray',linewidth=4,alpha=0.35)

    # plt.tight_layout()
    plt.show(block=False)
    fig.savefig(mainpath+'/BERESHIT_fig2.png', dpi=150, bbox_inches='tight')
    # plt.close()
