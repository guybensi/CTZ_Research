#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:13:58 2024

@author: eeytan
"""
# %%
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
    from scipy import signal
    import multiprocessing
    from multiprocessing import Pool
    import json
    #import tensorflow as tf
    #from tensorflow.keras import layers
    # sys.path.append('/export/home/eeytan/codes/python/')
    import Twilight_module as tw
    #import h5py
    #import gdal
    
    # %matplotlib qt5
    # %%
    starttime=time.time()
    #------------------------- FLAGS ----------------------------------------
    
    cpupool=40
    Delos=True
    print('Delos',Delos,cpupool)
    
    if Delos:
        	main_spath="/export/home/eeytan/projects/F1km/ALLSKY_newDat"  # path to save data to
    else:    
        main_spath=r'C:/Users/eeytan/Documents/Projects/F1km/Colocated_data'
    
    SATf='TERRA'  # satellite name AQUA/TERRA
    
    GenIm=False
    cldmsktyp='relaxed'   # cloud mask strickness
    PSFf=True # use point spread function of CERES
    NoColdCld=False  # If true don't consieder cold clouds
    plotfov=False
    plotsubdom=False
    savefigs=False
    BroadLW=True
    Batchf=1
    Nsvfile=7
    
    fovsc=20    # CERES FOV DIAMETER
    mfovp=100
    minN=50
    
    
    Granulei=0
    totpiece=-1
    datanum=0
    foviswmax=0
    fovilwmax=0
    
    YEAR=['2014']#,'2015']
    REGION=['Pacific']#,'Atlantic']
    MONTH=['june','sep','mrch','dec']
    
    
    
    # *************************** Wavelengths *****************************
    visls=[0.47,0.55,0.66,0.85]
    nirls=[0.905,0.935,0.940]
    swirls=[1.24,1.64,2.13,1.375]
    thermalls=np.array([3.75,3.96,3.96,4.05,4.46,4.5,6.7,7.3,8.55,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
    solarls=np.concatenate((visls,nirls,swirls),axis=0)
    
    
    exDict = {'Delos': Delos, 'cldmsktyp':cldmsktyp, 'NoColdCld': NoColdCld,'PSFf': PSFf, 'BroadLW':BroadLW, 'Batchf':Batchf, 'Nsvfile':Nsvfile, 'YEAR': list(YEAR),
              'REGION':list(REGION), 'MONTH': list(MONTH), 'visls':list(visls), 'nirls':list(nirls), 'swirls':list(swirls), 'thermalls':list(thermalls)}  
    with open(main_spath+'/INPUT.txt', 'w') as file:
         file.write(json.dumps(exDict)) # use `json.loads` to do the reverse
    
       
    # ----------- CERES FOV mask -------------------
    if PSFf:
        if Delos:    
            tmp = pd.read_csv('/export/home/eeytan/projects/Twilight_MODIS_CERES/ceres_psf.csv')
        else:
            tmp = pd.read_csv(r'C:/Users/eeytan/Documents/Projects/Twilight_RE_CERES/ceres_psf.csv')
        
        PSF=np.array(tmp) 
        PSF=PSF[0:-1,:]
        tmp=np.zeros((np.shape(PSF)[0],1))
        PSF=np.concatenate((PSF,tmp),axis=1)
        # PSF0=PSF
        # PSF0=np.where(np.isnan(PSF0),0.0,PSF0)
        PSF=np.where(np.isnan(PSF),0.0,PSF)
                                
        PSF2=signal.resample(PSF,55,axis=1)  # converting  from grid of the CERES-NOAA20 (altitude of 824km) with 250m to 1km and factoring to AQUA FOV (z=702km) by dividing by  4*824/702=4.7
        PSF2=signal.resample(PSF2,27,axis=0) # this dimention should be odd for analysis
        # PSF2=PSF2[:,5:50]
        PSF2=PSF2/np.sum(PSF2)
        FOVmsk=PSF2.copy()
        
        # #$$$$$$$$ aggregating NOAA-20 PSF from 250m to 1km (for instument at z=824km) by summing every 4x4 box to be one MODIS 1km pixel
        # PSF=np.array(tmp) 
        # PSF=PSF[0:-1,:]
        # tmp=np.zeros((np.shape(PSF)[0],1))
        # PSF=np.concatenate((PSF,tmp),axis=1)
        # # PSF=PSF/np.sum(PSF) 
        # window=np.ones((4,4))
        # PSF=tw.filter2D(PSF,window,False)
        # PSF=PSF[:-1,5:50]  # cutting PSF to be odd, keeping 98% of the energy
        # PSF=PSF/np.sum(PSF) 
        # FOVmsk=PSF.copy()

    else:
        xx,yy=np.meshgrid(range(0,fovsc+1),range(0,fovsc+1))
        distfrmcntr=np.sqrt((xx-fovsc/2)**2+(yy-fovsc/2)**2) 
        FOVmsk=distfrmcntr<fovsc/2
    
        
# %%
    for Iyear in YEAR:
        for Iregion in REGION:          
            for Imonth in MONTH:
#%%               
                #---------------------------- paths -----------------------------------------
                # tmp="/"+Iyear+"/"+Iregion+"/"+Imonth
                # spath = os.path.join(main_spath, tmp)
                spath=main_spath+'/'+Iyear+"/"+Iregion+"/"+Imonth
                print(spath)
                # os.makedirs(spath)
                os.makedirs(spath+r"/scalars")
                os.makedirs(spath+r"/moments")
                if GenIm:
                    os.makedirs(spath+r"/images")
                datanum=0
                
                if Delos:
                    datapath=r"/export/home/eeytan/Data/AQUA/"+Iyear+"/"+Iregion+"/"+Imonth+"/" 
                else:
                    datapath = r"C:/Users/eeytan/Documents/Data/Twilight/MODIS/"+SATf+"/"
                print(datapath)
                Dirlist=os.listdir(datapath)
                CERESnm = str([tmp for tmp in Dirlist if 'CERES' in tmp])
                CERESnm=CERESnm[2:-2]
                CERESpath=datapath+CERESnm

                CERESds = nc.Dataset(CERESpath)
                # print(CERESds.variables.keys()) # 
                
                temp="Cloud_mask_clear_strong_percent_coverage"
                Clrfov = CERESds.variables[temp]
                # print(Clrfov)
                fillV=CERESds.variables[temp].getncattr('_FillValue')
                Clrfov=np.array(Clrfov)
                Clrfov=np.where(Clrfov==fillV,np.nan,Clrfov) # mtrx[mtrx==fillV]=np.nan
                
                # temp="PSF_wtd_MOD04_cloud_fraction_ocean"
                # fovCF4 = CERESds.variables[temp]
                # # print(Fsw)
                # fillV=CERESds.variables[temp].getncattr('_FillValue')
                # fovCF4=np.array(fovCF4)
                # fovCF4=np.where(fovCF4==fillV,np.nan,fovCF4) # mtrx[mtrx==fillV]=np.nan
                
                temp="PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_"
                AODfov = CERESds.variables[temp]
                # print(AODfov)
                fillV=CERESds.variables[temp].getncattr('_FillValue')
                AODfov=np.array(AODfov)
                AODfov=np.where(AODfov==fillV,np.nan,AODfov)
                
                
                temp="CERES_SW_TOA_flux___upwards"
                Fsw = CERESds.variables[temp]
                # print(Fsw)
                fillV=CERESds.variables[temp].getncattr('_FillValue')
                Fsw=np.array(Fsw)
                Fsw=np.where(Fsw==fillV,np.nan,Fsw) # mtrx[mtrx==fillV]=np.nan
                
                if BroadLW:           
                    temp="CERES_LW_TOA_flux___upwards"
                else:
                    temp="CERES_WN_TOA_flux___upwards"
                
                Flw = CERESds.variables[temp]
                # print(Flw)
                fillV=CERESds.variables[temp].getncattr('_FillValue')
                Flw=np.array(Flw)
                Flw=np.where(Flw==fillV,np.nan,Flw)
                
                
                temp="CERES_SW_radiance___upwards"
                Isw = CERESds.variables[temp]
                # print(Isw)
                fillV=CERESds.variables[temp].getncattr('_FillValue')
                Isw=np.array(Isw)
                Isw=np.where(Isw==fillV,np.nan,Isw)
                
                if BroadLW:
                    temp="CERES_LW_radiance___upwards"
                    Ilw = CERESds.variables[temp]
                    # print(Ilw)
                    fillV=CERESds.variables[temp].getncattr('_FillValue')
                    Ilw=np.array(Ilw)
                    Ilw=np.where(Ilw==fillV,np.nan,Ilw)
                else:
                    temp="CERES_WN_radiance___upwards"
                    Ilw = CERESds.variables[temp]
                    # print(Ilw)
                    fillV=CERESds.variables[temp].getncattr('_FillValue')
                    Ilw=np.array(Ilw)
                    Ilw=np.where(Ilw==fillV,np.nan,Ilw)
                
                times = (CERESds.variables['time'])
                time_unit = CERESds.variables["time"].getncattr('units') # first read the 'units' attributes from the variable time
                #time_cal = CERESds.variables["time"].getncattr('calendar') # read calendar type
                local_time = nc.num2date(times, units=time_unit) # convert time
                
                CERESlon = (CERESds.variables['Longitude_of_CERES_FOV_at_surface'])
                CERESlat = (CERESds.variables['Colatitude_of_CERES_FOV_at_surface'])
                CERESvza = (CERESds.variables['CERES_viewing_zenith_at_surface'])
                CERESraa = (CERESds.variables['CERES_relative_azimuth_at_surface'])
                CERESsza = (CERESds.variables['CERES_solar_zenith_at_surface'])
                # print(times,CERESlon,CERESlat)
                #times=np.array(times)
                CERESsza=np.array(CERESsza).astype(np.float32)
                CERESraa=np.array(CERESraa).astype(np.float32)
                CERESvza=np.array(CERESvza).astype(np.float32)
                CERESlat=(np.array(CERESlat)-90)*(-1)
                CERESlat=CERESlat.astype(np.float32)
                CERESlon=np.array(CERESlon).astype(np.float32)
                CERESlen=int(Fsw.shape[0])
                CERESlon=np.where(CERESlon>180,CERESlon-360,CERESlon) #CERESlon[CERESlon>180]=CERESlon-360
                
                CERESjdate=np.ones(CERESlen)*np.nan
                for i in range(CERESlen):
                    year=local_time[i].year
                    month=local_time[i].month
                    day=local_time[i].day
                    hour=str(local_time[i].hour)
                    if len(hour)==1:
                        hour='0'+str(hour)
                    # minute=str(local_time[i].minute)
                    # if len(minute)==1:
                        # minute='0'+str(minute)
                    datestr=str(year)+'-'+str(month)+'-'+str(day)
                    dayj=tw.datestdtojd(datestr)
                    if dayj<100:
                        dayj='0'+str(dayj)
                    CERESjdate[i]=str(year)+str(dayj)+'.'+hour#+minute
                    # if i%100000==0:
                        # time.sleep(0.5)
                    
                del CERESds
                
                ceres_data=np.stack((CERESjdate,CERESlat,CERESlon,CERESsza,CERESvza,CERESraa,Fsw,Isw,Flw,Ilw,AODfov,Clrfov),axis=0)
                # %% --------------------------------------- MODIS ----------------------------------------------
                
                # if SATf=='AQUA':
                lstMOD02=os.listdir(datapath+'MOD02/')
                lstMOD03=os.listdir(datapath+'MOD03/')
                lstMOD06=os.listdir(datapath+'MOD06/')
                lstMOD35=os.listdir(datapath+'MOD35/')
                
            
                #%% ####################################################### MODIS granules iteration #####################################################################
                GranuletotN=len(lstMOD02)
                filestep=0
                
                if Batchf>0:
                    BatchN=Batchf
                    Nfiles_fBtch=GranuletotN #Nsvfile
                else:
                    BatchN=int(np.ceil(GranuletotN/Nsvfile))
                    Nfiles_fBtch=GranuletotN-(BatchN-1)*Nsvfile
                
                granI=0
                for ggg in range(0,BatchN): 
                    print('Batch ',ggg) 
                        
                    if ggg==BatchN-1:
                        granF=granI+Nfiles_fBtch
                        ARGS=[np.nan]*Nfiles_fBtch
                    else:
                        granF=granI+Nsvfile
                        ARGS=[np.nan]*Nsvfile
                        
                    grancount=-1
                    for grani in range(granI,granF):
                        grancount=grancount+1
                        MOD02=lstMOD02[grani] 
                        dtNtm=MOD02[9:22]
                        
                        MOD03=str([tmp for tmp in lstMOD03 if dtNtm in tmp])
                        MOD03=MOD03[2:-2]
                        MOD06=str([tmp for tmp in lstMOD06 if dtNtm in tmp])
                        MOD06=MOD06[2:-2]
                        MOD35=str([tmp for tmp in lstMOD35 if dtNtm in tmp])
                        MOD35=MOD35[2:-2]
                        
                        args=(datapath,MOD02,MOD03,MOD06,MOD35,ceres_data,cldmsktyp,FOVmsk,spath,GenIm)
                        # input('args ready')
                        ARGS[grancount]=args
                    granI=grani+1
                    
                    print('Started MPI...')  
                    if __name__ == '__main__':
                        # freeze_support()
                        with Pool(processes=cpupool) as pool:
                            results = pool.starmap(tw.MOD_CERES_allsky, ARGS)
                            time.sleep(0.5)
            
                        # report a message
                    print('Finished MPI')
# %% ###############################  OLD way to generate data ###################################################################                                   
                    # found=True
                    # i=0
                    # while found & (i<len(results)):
                    #     tmp=results[i]
                    #     i=i+1
                    #     if np.size(tmp)>1:
                    #         found=False

                    # if found==True:
                    #     print('NO DATA IN THE  BATCH')
                    #     continue
                    
                    # if ggg==0:
                    #     Nscalars=len(results[0])
                    #     NswChannels=np.shape(tmp[1]["mSolarspec"])[1] # get number of SW dimention
                    #     NlwChannels=np.shape(tmp[1]["mThermalspec"])[1] # get number of LW dimention
                    #     FOVsz=[np.shape(tmp[2]["ImSW"])[2],np.shape(tmp[2]["ImSW"])[3]]
                    #     print('Nscalars=',Nscalars,'Nswchannels=',NswChannels,'Nlwchannels=',NlwChannels, 'FOVsz=',FOVsz)
                        
                    # Fsw=np.ones((1),dtype=np.float32)*np.nan
                    # Isw=Fsw.copy()
                    # Flw=Fsw.copy()
                    # Ilw=Fsw.copy()
                    # meanSumIsw=Fsw.copy() 
                    # stdSumIsw=Fsw.copy()
                    # meanSumIlw=Fsw.copy() 
                    # stdSumIlw=Fsw.copy() 
                    
                    # meanDFNC=Fsw.copy()
                    # stdDFNC=Fsw.copy()
                    # meanCTH=Fsw.copy()
                    # stdCTH=Fsw.copy()
                    # CF=Fsw.copy()
                    # LiqPhsf=Fsw.copy()
                    
                    # CERESVZA=Fsw.copy()
                    # CERESSZA=Fsw.copy()
                    # CERESRAA=Fsw.copy()
                    # CERESLat=Fsw.copy()
                    # CERESLon=Fsw.copy()
                    
                    # CERESAODsw=Fsw.copy()
                    # CERESClrfsw=Fsw.copy()
                    # glintf=Fsw.copy()
                    # landf=Fsw.copy()
                          
                    
                    # mSWspec=np.ones((1,NswChannels),dtype=np.float32)*np.nan
                    # stdSWspec=mSWspec.copy()
                    # skewSWspec=mSWspec.copy()
                    # kurtSWspec=mSWspec.copy()
                    # mLWspec=np.ones((1,NlwChannels),dtype=np.float32)*np.nan
                    # stdLWspec=mLWspec.copy()
                    # skewLWspec=mLWspec.copy()
                    # kurtLWspec=mLWspec.copy()
                    
                    # Imsw=np.ones((1,NswChannels,FOVsz[0],FOVsz[1]),dtype=np.float32)*np.nan
                    # Imlw=np.ones((1,NlwChannels,FOVsz[0],FOVsz[1]),dtype=np.float32)*np.nan
                    # Imviews=np.ones((1,3,FOVsz[0],FOVsz[1]),dtype=np.float32)*np.nan
                    
                    # if ggg==BatchN-1:
                    #     granF=Nfiles_fBtch
                    # else:
                    #     granF=Nsvfile
                           
                    # for grani in range(0,granF):
                        
                    #     output=results[grani]
                    #     # print('Granulei',grani)    
                    #     if output=='nothing_found':
                    #         # print('nothing_found')
                    #         continue
                    #     Fsw=np.concatenate((Fsw,output[0]["Fsw"]))
                    #     Isw=np.concatenate((Isw,output[0]["Isw"]))
                    #     Flw=np.concatenate((Flw,output[0]["Flw"]))
                    #     Ilw=np.concatenate((Ilw,output[0]["Flw"]))
                    #     meanSumIsw=np.concatenate((meanSumIsw,output[0]["mSumIsw"]))
                    #     stdSumIsw=np.concatenate((stdSumIsw,output[0]["stdSumIsw"]))
                    #     meanSumIlw=np.concatenate((meanSumIlw,output[0]["mSumIlw"]))
                    #     stdSumIlw=np.concatenate((stdSumIlw,output[0]["stdSumIlw"]))
                                 
                    #     meanDFNC=np.concatenate((meanDFNC,output[0]["mDFNC"]))
                    #     stdDFNC=np.concatenate((stdDFNC,output[0]["stdDFNC"]))
                    #     meanCTH=np.concatenate((meanCTH,output[0]["meanCTH"]))
                    #     stdCTH=np.concatenate((stdCTH,output[0]["stdCTH"]))
                    #     CF=np.concatenate((CF,output[0]["CF"]))
                    #     LiqPhsf=np.concatenate((LiqPhsf,output[0]["LiqPhsf"]))
                        
                        
                    #     CERESVZA=np.concatenate((CERESVZA,output[0]["VZA"]))
                    #     CERESSZA=np.concatenate((CERESSZA,output[0]["SZA"]))
                    #     CERESRAA=np.concatenate((CERESRAA,output[0]["RAA"]))
                    #     CERESLat=np.concatenate((CERESLat,output[0]["Lat"]))
                    #     CERESLon=np.concatenate((CERESLon,output[0]["Lon"]))
                        
                    #     CERESAODsw=np.concatenate((CERESAODsw,output[0]["AOD"]))
                    #     CERESClrfsw=np.concatenate((CERESClrfsw,output[0]["Clrf"]))
                    #     glintf=np.concatenate((glintf,output[0]["glintf"]))
                    #     landf=np.concatenate((landf,output[0]["landf"]))
                        
                    #     mSWspec=np.concatenate((mSWspec,output[1]["mSolarspec"]))  
                    #     stdSWspec=np.concatenate((stdSWspec,output[1]["stdSolarspec"])) 
                    #     skewSWspec=np.concatenate((skewSWspec,output[1]["skewSolarspec"]))  
                    #     kurtSWspec=np.concatenate((kurtSWspec,output[1]["kurtSolarspec"])) 
                    #     mLWspec=np.concatenate((mLWspec,output[1]["mThermalspec"]))
                    #     stdLWspec=np.concatenate((stdLWspec,output[1]["stdThermalspec"]))
                    #     skewLWspec=np.concatenate((skewLWspec,output[1]["skewThermalspec"]))
                    #     kurtLWspec=np.concatenate((kurtLWspec,output[1]["kurtThermalspec"]))
                        
                        
                    #     Imsw=np.concatenate((Imsw,output[2]["ImSW"])) 
                    #     Imlw=np.concatenate((Imlw,output[2]["ImLW"])) 
                    #     Imviews=np.concatenate((Imviews,output[2]["Imviews"])) 
                        

                    #     # swallnanraw=np.nonzero(np.nansum(np.isnan(Fsw),axis=1)<np.shape(Fsw)[1])[0]                  
                    #     # lwallnanraw=np.nonzero(np.nansum(np.isnan(lwCERES_MOD_F),axis=1)<np.shape(lwCERES_MOD_F)[1])[0] 
                        
                    # datanum=datanum+1
                    # print('saving',datanum)
                    # print('N Fsw FOVs- ',np.sum(~np.isnan(Fsw.ravel())))
                    # print('N Flw FOVs- ',np.sum(~np.isnan(Flw.ravel())))
                    # print('@@@@@@@@@@@@@@@@@@@@@@@')
                    # print(np.shape(Fsw))
                    # print(np.shape(mSWspec))
                    
                    # # str(datanum) to save batches
                    # np.savez(spath+r'/AllskyData_N'+dtNtm+'.npz',Fsw=Fsw,Isw=Isw,Flw=Flw,Ilw=Ilw,mSumIsw=meanSumIsw,mSumIlw=meanSumIlw,stdSumIsw=stdSumIsw,stdSumIlw=stdSumIlw,
                    #          mDFNC=meanDFNC,stdDFNC=stdDFNC,mCTH=meanCTH,stdCTH=stdCTH,CF=CF,LiqPhsf=LiqPhsf,
                    #          VZA=CERESVZA,RAA=CERESRAA,SZA=CERESSZA,Lon=CERESLon,Lat=CERESLat,
                    #          AODfov=CERESAODsw, CLRfov=CERESClrfsw,glintf=glintf,landf=landf,
                    #          mSWspec=mSWspec,stdSWspec=stdSWspec,skewSWspec=skewSWspec,kurtSWspec=kurtSWspec,mLWspec=mLWspec,stdLWspec=stdLWspec,skewLWspec=skewLWspec,kurtLWspec=kurtLWspec )
                    
                    # np.savez(spath+r'/AllskyImages_N'+dtNtm+'.npz',Imsw=Imsw,Imlw=Imlw,Imviews=Imviews)
                        
                             
                # del CERESlat,CERESlon,CERESraa,CERESsza,CERESvza
    print('elapsed time ' + str((time.time()-starttime)/60))
