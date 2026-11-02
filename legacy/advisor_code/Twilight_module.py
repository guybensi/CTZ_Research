# Twilight Module
"""
Created on Thu May 11 15:04:52 2023

@author: eeytan
"""
import datetime
import os
import sys
import time
import pandas as pd
import netCDF4 as nc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
from string import whitespace
from skimage.measure import label as label_skim, regionprops

#%%
def readMODIS (fname,var,sclsNoffs):
    dataset = nc.Dataset(fname, 'r')
    # print(dataset.variables.keys()) # 
    mtrx = dataset.variables[var]
    # print(mtrx) # 
    MODsz=np.shape(mtrx)           
    mtrx=np.array(mtrx) 
    mtrx=mtrx.astype('float')
    fillV=dataset.variables[var].getncattr('_FillValue')
    mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
    if sclsNoffs !=0:
        scalesn=sclsNoffs+'_scales'
        offsn=sclsNoffs+'_offsets'
        scales=dataset.variables[var].getncattr(scalesn)
        offset=dataset.variables[var].getncattr(offsn)
        # print(scales)
        # print(offset)
        scales = np.tile(scales[:, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2]))
        offset = np.tile(offset[:, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2]))
        mtrx=(mtrx-offset)*scales
    # mtrx=np.reshape(mtrx,()) 
    return(mtrx)    
 
# --------------------------------------------------------------------------

def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)

# ----------------------------------------------------------------------------
def Planckfunc(Lamda,Dinput,temp):
    #Radiance=double(Radiance);
    c=np.float64(299792458.0) # m/s speed of light in vaccum
    h=np.float64(6.62607004e-34) # J*s Planck constant 
    k=np.float64(1.38064852e-23) #  J/K Boltzmann constant
    MODsz=np.shape(Dinput) 
    c1=2.0*h*(c**2.0)
    c2=h*c/k
    k1=(c1/(Lamda**5))*10**(-6)
    k2=c2/Lamda
    if len(MODsz)==2:
        k1=np.tile(k1[:, np.newaxis], (1,MODsz[1])) 
        k2=np.tile(k2[:, np.newaxis], (1,MODsz[1]))
    elif len(MODsz)==3:
        k1=np.tile(k1[:, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2])) 
        k2=np.tile(k2[:, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2]))
    elif len(MODsz)==4:
        k1=np.tile(k1[:, np.newaxis, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2],MODsz[3])) 
        k2=np.tile(k2[:, np.newaxis, np.newaxis, np.newaxis], (1,MODsz[1],MODsz[2],MODsz[3])) 
    if temp:
        Doutput=k2/np.log((k1/Dinput)+1.0);
    else:
        Doutput=k1/(np.exp(k2/Dinput)-1.0);
    return(Doutput)

#%%

def miecalc(wl,r,subst,nk_path,plotPhs):
    
    import miepython as mp

    # wl=0.55 # micron
    # r=5  # microns
    # subst='water'

    data=np.load(nk_path+subst+'.npz') #r'C:\Users\eeytan\Documents\New stuff after mv to ahrddrive\codes\refractive indices/'+subst+'.npz')
    data=data["data"]

    ind=findmindiff(wl, data[:,0])[0][0]
    
    if abs(wl-data[ind,0])>0.5:     
        print('##########################################################################################')
        print('!!!!!!!!!!!!!! Refractive index wavelength is > 0.5 micron than wl !!!!!!!!!!!!!!!!!!!!!!')
        input('!!!!!!!!!!!!!! asked for '+str(wl)+ ' microns and got '+ str(data[ind,0])+ ' microns !!!!!!!!!!!!!!!!!!!!!!!!!!!'   )
    
    n=data[ind,1]
    k=data[ind,2]

    m = n + k*1j

    x = 2 * np.pi * r / wl


    Qext, Qsca, Qabs, g = mp.efficiencies_mx(m, x)
    ssa = Qsca / Qext

    print(Qext, Qsca, Qabs, g, ssa)

    # Scattering angle (0 to pi)
    mu = np.linspace(-1, 1, 20000)
    theta = np.arccos(mu)   # [0, pi]

    # Mie scattering amplitudes
    S1, S2 = mp.S1_S2(m, x, mu)

    # Phase function
    P = np.abs(S1)**2 + np.abs(S2)**2

    # Normalize
    dmu = mu[1] - mu[0]
    P /= 2 * np.pi * np.sum(P * dmu)

    # --- Make it axisymmetric ---
    theta_full = np.concatenate([theta, 2*np.pi - theta[::-1]])
    P_full = np.concatenate([P, P[::-1]])
    
    if plotPhs:    
        # --- Polar plot ---
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
    
        ax.plot(theta_full, P_full)
        ax.set_theta_zero_location("E")   # forward scattering up
        ax.set_theta_direction(-1)        # clockwise
        ax.set_title("Axisymmetric Phase Function for "+subst+ " with $\lambda$="+str(wl)+" r="+str(r),pad=30)
        ax.set_rscale("log")
        plt.show()
    
    return(Qext,Qsca,Qabs,g,[P_full,theta_full])
#%%
def LognormSD(Rs,N,Rg,sig):
    
    #  rg - geometric mean
    # sig - geometric standard deviation
    
    DS=N/np.sqrt(2*np.pi)/Rs/np.log(sig)*np.exp(-(np.log(Rs/Rg))**2/(np.log(sig)**2))
    return(DS)


def AerGrow(RH,pres,temp,rm,aercom):
    
   if np.shape(RH)==():
        RH=RH.reshape(1)
        pres=pres.reshape(1)
        temp=temp.reshape(1)
    
# RH is a vector of the relative humidity for each size distribution in fraction and not percents!
# pres- same but for pressure
# temp- same but for temperature
# PSD- particle size distributions concentration (#/cm3), sum(PSD,axis=1)=Na first dimention as for RH, second dimation is the radii size 
# rm- the radii of the PSD same size as PSD
# drm- width of each rm bin same size as PSD
# aercom- aerosol composition: nacl or ammunium sulfatre (amsul). STRING    
    
#-----------------------------------------------------------------------------
#  Will need to read in # of timesteps, T (C), P(cgs), rh (e.g., 0.85), x, y, z,size distribution (#/cc in each bin)
#-----------------------------------------------------------------------------
#
# mass fraction of soluble material: epsilon (0 < epsilon <= 1)
# Thiss code defines size grid based on UHSAS aerosol size grid
# density of soluble material: rhosol
# molecular weight of soluble material: rMs
# pressure: pres (cgs: e.g., 900 mb = 900,000)
# temperature: temp (Kelvin)
# Assumptions: Density of various aerosol components is the same

   if np.nansum(RH.ravel()>2)>0:
       input('!!!!!!!!!!  RH given in percents !!!!! Press SPACE botton to continue by correcting with RH/100 !!!!!! ')
       RH=RH/100
       
       
   RADnew=np.ones(np.shape(rm))
   itbin= np.shape(rm)[1] 
   nt= np.shape(rm)[0] 
  
   rad=np.ones((itbin))*np.nan
   # rm=rad.copy()
   rm0=rad.copy()
   diameter=rad.copy() 
   dlogD=rad.copy()  # mid points of UHSAS bins in nm (i.e., 2 x rm ) and dlogD
   conc=np.ones((nt,itbin))*np.nan
# rh=np.ones((nt)) 
# wup=rh(nt), zpos(nt), time(nt)
# real  xlwc(nt),temp1(nt),pres1(nt)
   rhow=1
   rMw=18.016
   rr=8.3144e7
   rd=2.8704e6
   eps=0.62197
   epsilon=1.00
  
   if aercom=='amsul':
        rMs=132.13
        rhosol=1.769
   elif aercom=='nacl':
        rMs=58.44
        rhosol=2.165
    
# rm(ic)=0.5*diameter(ic)*1.e-07   
   tmass=4./3.*np.pi*rhosol*rm**3.
  
   # for ic in range(0,itbin):
   #       dbl=rm[ic]*2-drm[ic]  # 9.7 is bin width
   #       dbu=rm[ic]*2+drm[ic]
   #       dlogD[0,ic]=np.log10(dbu)-np.log10(dbl)  # for calculating pure # conc in each bin
         
   for it in range(0,nt):
       
      # PSD=PSD*np.matlib.repmat(dlogD,nt,itbin)
       
      rho=pres[it] / (rd * temp[it]) # air density (g/cm3)
      # xntot=np.sum(PSD,axis=1)
      satr=min(RH[it],0.99)   # Q- why only below 0.99, should I not treat 1?
      
      if (np.isnan(pres[it])) | (np.isnan(temp[it])) | (np.isnan(RH[it])):
            continue
      
      pass1 = (2.* rMw) / (rr * temp[it] * rhow)
      pass2 = 76.1 - 0.155*(temp[it]-273.15)  
      
      for ic in range(0,itbin):
         f1 = 3./(4.*np.pi*rhosol)
         errabs=0.0
         rel1=1e-9
         itmax=50

         jop = 2

         

         xl  = ((f1 * tmass[it,ic])**(1/3)) * 1.01  # 1.01*CCN radius as lower limit
         xr = 5e-3                              # 50 microns
         tms0=tmass[it,ic]  # mass of *soluble* particle in grams
         tms=epsilon*tmass[it,ic]  # mass of *soluble* particle in grams
         
         r0=rm[it,ic]  # radius of total dry particle in grams
    
         sofain=[jop,satr,tms,tms0,r0,pass1,pass2,epsilon,aercom]
         xr=ZZBREN(errabs,rel1,xl,xr,itmax,sofain)
         rad[ic]  =  xr
         if (rm[it,ic]>=1e-05): rad[ic]=0.97*xr    # scale largest particles for sub-eqm
         if (satr<=0.05): rm0[ic]=rad[ic]
#        if(satr.lt.0.98)print *,'g(rh) ',rad(ic)/rm(ic), satr

         # for ic in range(0,itbin):  # refradtive index of soot

             # xmas=tmass(ic)*epsilon   # mass of A.S.
             # vas=xmas/rhosol          #vol of A.S

             # xmsoot=tmass(ic)*(1.-epsilon) # mass of soot
             # vsoot=xmsoot/rhosol          # volume of soot in wet particle

             # vwet = 4./3.*np.pi * rad[ic]**3. # vol of wet particle
             # vw=vwet-vas-vsoot              # volume of water

# Note, if you set nacl to .true., then replace refreas and refimas with the nacl equivalents
# Also, the physical constants (density, molecular wt) shoud be set accordingly.

             # refre=(vas*refreas+vsoot*refresoot+vw*refrew)/vwet
             # refim=(vas*refimas+vsoot*refimsoot+vw*refimw)/vwet
         rad[ic]=rad[ic]*1e4  # in microns
         RADnew[it,ic]=rad[ic]
         
      
   return(RADnew)


def ZZBREN(errabs, errrel, a, b, maxfn, SOFAin):
      is10=True
      is20=True
      i=1
      jop=SOFAin[0]; satr=SOFAin[1]; tms=SOFAin[2]; tms0=SOFAin[3]; r0=SOFAin[4]; pass1=SOFAin[5]; pass2=SOFAin[6]; epsilon=SOFAin[7]; aercom=SOFAin[8];
# c  modification to IMSL subroutine ZBREN to find root of an equation.
      t = errrel
      ic = 2
      s = a
      fa = SOFA(s,jop,satr,tms,tms0,r0,pass1,pass2,epsilon,aercom)
      s = b
      fb = SOFA(s,jop,satr,tms,tms0,r0,pass1,pass2,epsilon,aercom)
# c                                  test for same sign
      if (fa*fb > 0.0):
        print(' ERROR IN ARGUMENTS TO ZZBREN')
# c        write (100,*) ' fa, fb =',fa,fb
        return
      while is10:
          if is20:
              c = a  # 10
              fc = fa
              d = b - c
              e = d
          if (abs(fc) < abs(fb)):  # 20
             a = b
             b = c
             c = a
             fa = fb
             fb = fc
             fc = fa
          i=i+1   
          is20=True
          tol = t*max(abs(b),0.1)
          rm = (c-b)*0.5
    # c                                  test for first convergence criteria
          if (abs(fb) <= errabs): break
    # c                                  test for second convergence criteria
          if (abs(c-b) <= tol): break
    # c                                  check evaluation counter
          if (ic >= maxfn): 
             print('failure to converge in zzbren')
    # c        write (100,*) ' f1, f2 =',c,b
             break
        
    # c                                  is bisection forced
          if (abs(e) < tol):
             e = rm
             d = e
          else:
             if (abs(fa) <= abs(fb)):
                e = rm
                d = e
             else:
                s = fb/fa
                if (a != c):
    # c                                  inverse quadratic interpolation
                   q = fa/fc
                   r = fb/fc
                   rone = r - 1.0
                   p = s*((c-b)*q*(q-r)-(b-a)*rone)
                   q = (q-1.0)*rone*(s-1.0)
                else:
    # c                                  linear interpolation
                   p = (c-b)*s
                   q = 1.0 - s
               
                if (p > 0.0): q = -q
                if (p < 0.0): p = -p
                s = e
                e = d
    # c                                  if abs(p/q).ge.75*abs(c-b) then
    # c                                     force bisection
                if (p+p >= 3.0*rm*q):
    # c                                  bisection
                   e = rm
                   d = e
                else:
    # c                                  if abs(p/q).ge..5*abs(s) then force
    # c                                     bisection. s = the value of p/q
    # c                                     on the step before the last one
                   if (p+p >= abs(s*q)):
    # c                                  bisection
                      e = rm
                      d = e
                   else:
                      d = p/q
      
    # c                                  increment b
          a = b
          fa = fb
          temp = d
          if (abs(temp) <= 0.5*tol): temp = abs(0.5*tol)*rm/abs(rm)
          b = b + temp
          s = b
          fb = SOFA(s,jop,satr,tms,tms0,r0,pass1,pass2,epsilon,aercom)
          ic = ic + 1
          if (fb*fc <= 0.0): is20=False #go to 20
          # go to 10
          if i==100000:
              print('no convergence')
              break
      a = c

      return(b)
      

# ------------------------------------------------------------------------

def SOFA(ra,jop,satr,tms,tms0,r0,pass1,pass2,epsilon,aercom):

# c  purpose:  This function calculates one of three options related to
# c            calculation of equilibrium diameters via P&K 6-26.
# c               jop = 1:   calculate S = S(ra,xms) where ra = equilibrium
# c                          droplet radius and xms = nucleus mass.
# c               jop = 2:   calculate S(ra,xms) - S (to find the zero)
# c               jop = 3:   calculate dS(ra,xms)/d(ra) (to find s*)

        pi43=4.188790205
        rr=8.3144e7
        rhow=1.0
        rMw=18.016
      
        if (aercom=='amsul'):
            rMs=132.13
            rhosol=1.769
        elif (aercom=='nacl'):
            rMs=58.44
            rhosol=2.165

        bfac   = (tms * rMw) / rMs
        d      = 2.* ra                        # droplet diameter (cm)
        vdrop  = pi43 * ra**3                  # droplet volume

# c assume rho of non-hygrscopic part is same as that or hygroscopic part
        xmw    = rhow * (vdrop - (tms0/rhosol))
# c Include entire mass for density calc
        rhos   = (tms0 + xmw) / vdrop           # droplet density
# c Use only soluble part for xnuphi calc
        xnuphi,z = FNUPHI(d,rhos,tms,aercom)
        # if (aercom=='amsul'): xnuphi = FNUPHI(d,rhos,tms,z)  # for ammonium sulfate
        # if (aercom=='nacl'): xnuphi = FNUPHIsc(d,rhos,tms,z)      # for NaCl
        sigsa  = pass2 + (2.17*z)              # sigma s/a P&K 5-19
        f1     = (pass1 * sigsa) / ra
# c     f2     = (bfac * xnuphi) / xmw
# c (Assumes density of insoluble part is same as that of soluble part)
# c Gives same result as the f2 used above!

        f2=xnuphi*epsilon*rMw*rhosol*r0**3./(rhow*rMs*(ra**3.-r0**3.))


        if ((jop == 1) or (jop == 2)):
           answer = np.exp(f1-f2)                    # Option 1:  S(a,m)
        if (jop == 2): answer = answer - satr      # Option 2:  S(a,m) - s

        if (jop == 3):                   # Option 3:  dS(a,m)/da = 0
           drhos = (3./ra) * (rhow - rhos)     # drhos/da
           f3    = (vdrop/xmw) * (drhos + (3.* rhos/ra))
           dz    = -z * f3                     # d/da (molality)
           dsig  = 2.17 * dz                   # d/da (sigma s/a)
           f4    = (pass1 / ra) * (dsig - (sigsa/ra))
           f5    = f2 * f3
           answer= f4 + f5
      
        return(answer)
      
#******************************************************************************
def FNUPHI(d,rho,rmass,aercom):
# c+ purpose:  Calculate the value for nu*phi for ammonium sulfate as a function+
# c+           of molality, using a two-part fourth order fit to Kunkel's       +
# c+           (JAS 26,1344) correction to Low (JAS 26, 608).                   +
    if aercom=='amsul':
      bb=[2.44675,1.714755245,2.69934406,-2.231643357,0.721153846]
      cc=[2.139970678, -0.338048456, 0.135358693,-0.019287787, 0.001016937]
      rMs=132.13
    elif aercom=='nacl':
        bb=[1.87109,-0.13563,0.20419,-0.09126, 0.02278,-0.00287,0.00014]
        rMs=58.44
      
    rMw=18.016
    pi6= 0.523598776
      

# c kg water
    xmw = ((pi6 * d**3 * rho) - rmass) * 0.001
# c moles solute /kg water
    x   = rmass / (rMs * xmw)
# c saturation
    if ((x <0.) or (x > 5.5)): x = 5.5   # Check this with Graham !!!
    
    if aercom=='amsul':    
        if (x < 1.0):
             nuphi = bb + x*(bb[1] + x*(bb[2] + x*(bb[3] + x*bb[4])))
        else:
             nuphi = cc[0] + x*(cc[1] + x*(cc[2] + x*(cc[3] + x*cc[4])))
    elif aercom=='nacl':  
        nuphi = bb[0] + x*bb[1] + (x**2.)*bb[2] + (x**3.)*bb[3] + (x**4.)*bb[4] + (x**5.)*bb[5] + (x**6.)*bb[6]
    
    return(nuphi,x)




#%% ------------------------------------------------------

def Iorgcalc(cldmsk,resolution,conn):
    resolution = 2 # in km
    conn = 2 # connectivity
    
    # Calculate LvL score and the Iorg + NNCDF
    
    # Loop for the greenCu
    dy,dx = np.shape(cldmsk)
           
    NearestNeighborVec = np.arange(0,np.sqrt(dx**2+dy**2)*resolution,0.1) #Vpixels, dependent on the domain size!
    
    L, N = label_skim(cldmsk, connectivity=conn, return_num=True)
    # Get region properties (Centroid and Area)
    props = regionprops(L)
    # [NNdist, centroidsX, centroidsY, Areas, L, nn_green] = Centroid_distances(GOES_cldmsk), resolution, conn);
    Areas=np.ones(N)*np.nan
    centroidsY=Areas.copy()
    centroidsX=Areas.copy()
    NNdist=Areas.copy()
    surface_length=Areas.copy()
    for rr in range(N):
        region=props[rr]
        Areas[rr]=region.area
        surface_length[rr]=region.perimeter_crofton
        centroidsY[rr]=region.centroid[0]
        centroidsX[rr]=region.centroid[1]
    
    for rr in range(N):
        tmpx=centroidsX.copy()
        tmpx[rr]=np.nan
        distances=np.sqrt((centroidsX[rr]-tmpx)**2+(centroidsY[rr]-centroidsY)**2)
        NNdist[rr]=np.nanmin(distances)
  
    # Compute CF
    CLDf = np.nansum(cldmsk) / np.nansum(~np.isnan(cldmsk))  #(dx * dy)
    # COREf = np.nansum(F_msk) / np.nansum(~np.isnan(GOES_cldmsk))
    Srfc_length=np.nansum(surface_length)*resolution
    
    if N > 2: #if more than 2 clouds in the image
        #  Calculate Iorg
        Ntemp,edges = np.histogram(NNdist, NearestNeighborVec,density=False)
        obsNNCDF_green = np.nancumsum(Ntemp / np.nansum(Ntemp))
        binav = (edges[1:] + edges[:-1]) / 2
        # For Poisson Calculations:
        Lambda = N / (dx * dy * resolution**2); # number of cloud points per unit area [#/km^2]
        PoissonNNCDF_green = 1 - np.exp(-Lambda * np.pi * binav**2)
        Iorg = np.trapz(obsNNCDF_green,PoissonNNCDF_green)
        # plt.figure(), plt.plot(PoissonNNCDF_green,obsNNCDF_green), plt.plot([0,1],[0,1]) #plt.plot(binav,obsNNCDF_green,label='Obs'),plt.plot(binav,PoissonNNCDF_green,label='Poisso'),plt.legend()
        
        # Calculate LvL score
        # [KS1, KS2] = LVL(I_Nbox_green_CM)
    else:
        print('error calc. Iorg') # in case there are fewer than 2 clouds
        Iorg = np.nan # fill value
        # KS1 = np.nan # fill value
        # KS2 = np.nan # fill value
    
    return(Iorg,CLDf,Srfc_length,Areas)
    
 # ---------------------------------------------------------
def LvL(cloud_mask):
    
    import numpy as np
    # from scipy.stats import ks_2samp
    # from PIL import Image
    from scipy.ndimage import label
    from scipy.ndimage import sum_labels
    # import cv2
    
    # Convert cloud_mask to double
    cloud_mask = cloud_mask.astype(float)
    sz = cloud_mask.shape
    mnsz = min(sz)  # minimum scale of the field
    sz_cloud_mask = sz[0] * sz[1]
    p = np.sum(cloud_mask) / sz_cloud_mask  # cloud fraction

    # estimating the ideal length to capture almost 100% of the histogram
    # we want that the error < exp(-12) for the perfect rand case
    mxln = int(np.floor(abs(12 / np.log(p))) + 1)
    mxln = min(mxln, mnsz)

    # The cloud part
    # Flatenning along the two directions
    # then we need to divide c1 and c2 by two

    B = cloud_mask.flatten()  # rows
    C = cloud_mask.flatten(order='F') #columns
    B = np.concatenate((C, B))

    L, num_labels = label(B)
    # Label connected components in the binary image
    areas = sum_labels(B, L, index=np.arange(1, num_labels + 1))
    mx_ar = np.max(areas) + 1  
    # -------------- original --------------------------
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # # Extract area of each connected component
    # areas = stats[1:, cv2.CC_STAT_AREA]
    # # Adding one to the max area
    # mx_ar = np.max(areas) + 1  
#  ---------------------------------------------------------------------
    # To avoid single histogram
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    # Get the cloud chord length counts - c1    
    c1, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    # Correct for flattening in two directions
    c1 = c1 / 2
    s1 = np.sum(c1)

    # Now, the theortical calculations ct, nt for a given cloud fraction (p)
    nt1 = np.arange(1, mxln + 1)
    ct1 = (sz_cloud_mask * (1 - p) ** 2) * p ** nt1
    st1 = np.sum(ct1)
    nt1 = nt1.astype(int)


    # Get the KS score for the cloud part
    adf1 = np.abs(np.cumsum(ct1 / st1) - np.cumsum(c1 / s1))
    KS1 = np.max(adf1)


    # The void part
    q = 1 - p # void fraction
    mxln = int(np.floor(abs(12 / np.log(q))) + 1)
    mxln = min(mxln, mnsz)
    B = -B + 1

    L, num_labels = label(B)
    # Label connected components in the binary image
    
    areas = sum_labels(B, L, index=np.arange(1, num_labels + 1))
    mx_ar = np.max(areas) + 1  
    
    # -------------- original -----------------------------------
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # # Extract area of each connected component
    # areas = stats[1:, cv2.CC_STAT_AREA]
    # # Adding one to the max area
    # mx_ar = np.max(areas) + 1  
# --------------------------------------------------------------------------
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    c2, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    c2 = c2 / 2
    s2 = np.sum(c2)

    nt2 = np.arange(1, mxln + 1)
    ct2 = (sz_cloud_mask * (1 - q) ** 2) * q ** nt2
    st2 = np.sum(ct2)
    nt2 = nt2.astype(int)

    # Get the KS score for the void part
    adf2 = np.abs(np.cumsum(ct2 / st2) - np.cumsum(c2 / s2))
    KS2 = np.max(adf2)

    return (KS1,KS2)

# -----------------------------------------------------------------------------
def bindata(mtrx,metricmtrx,bins,minN):
    canelsNo=np.shape(mtrx)[0]
    binedmtrx=np.zeros((canelsNo,len(bins)))
    binNo=np.zeros(len(bins))
    binstd=np.zeros((canelsNo,len(bins)))
    for i in range(0,canelsNo):
        tmp=mtrx[i,:,:].ravel().copy()
        tmp2=metricmtrx.ravel().copy()
        binmsk=np.where(np.array(tmp2)<=bins[0])
        binedmtrx[i,0]=np.nanmean(tmp[binmsk])
        binNo[0]=np.count_nonzero(binmsk)
        binstd[i,0]=np.nanstd(tmp[binmsk])
        for j in range(1,len(bins)-1):
            binmsk=np.where((tmp2>(bins[j-1])) & (tmp2<=bins[j]))
            Nbin=np.count_nonzero(binmsk)
            if Nbin<minN:
                binedmtrx[i,j]=np.nan
                binNo[j]=Nbin
                binstd[i,j]=np.nan
            else:          
                binedmtrx[i,j]=np.nanmean(tmp[binmsk])
                binNo[j]=Nbin
                binstd[i,j]=np.nanstd(tmp[binmsk])
        j=j+1
        binmsk=np.where(tmp2>bins[j])
        Nbin=np.count_nonzero(binmsk)
        if Nbin<minN:
            binedmtrx[i,j]=np.nan
            binNo[j]=Nbin
            binstd[i,j]=np.nan
        else:          
            binedmtrx[i,j]=np.nanmean(tmp[binmsk])
            binNo[j]=Nbin
            binstd[i,j]=np.nanstd(tmp[binmsk])
    return(binedmtrx,binstd,binNo)

def bindata1D(mtrx,metricmtrx,bins,minN):
    canelsNo=np.shape(mtrx)[0]
    binedmtrx=np.zeros((canelsNo,len(bins)))
    binNo=np.zeros(len(bins))
    binstd=np.zeros((canelsNo,len(bins)))
    for i in range(0,canelsNo):
        tmp=mtrx[i,:].ravel().copy()
        tmp2=metricmtrx[i,:].ravel().copy()
        binmsk=np.where(np.array(tmp2)<=bins[0])
        binedmtrx[i,0]=np.nanmean(tmp[binmsk])
        binNo[0]=np.count_nonzero(binmsk)
        binstd[i,0]=np.nanstd(tmp[binmsk])
        for j in range(1,len(bins)-1):
            binmsk=np.where((tmp2>(bins[j-1])) & (tmp2<=bins[j]))
            Nbin=np.count_nonzero(binmsk)
            if Nbin<minN:
                binedmtrx[i,j]=np.nan
                binNo[j]=Nbin
                binstd[i,j]=np.nan
            else:          
                binedmtrx[i,j]=np.nanmean(tmp[binmsk])
                binNo[j]=Nbin
                binstd[i,j]=np.nanstd(tmp[binmsk])
            j=j+1
            binmsk=np.where(tmp2>bins[j])
            Nbin=np.count_nonzero(binmsk)
            if Nbin<minN:
                binedmtrx[i,j]=np.nan
                binNo[j]=Nbin
                binstd[i,j]=np.nan
            else:          
                binedmtrx[i,j]=np.nanmean(tmp[binmsk])
                binNo[j]=Nbin
                binstd[i,j]=np.nanstd(tmp[binmsk])
    return(binedmtrx,binstd,binNo)

# ------------------------------------------------------------------------
def findsat(mtrx,noisfac,plotf): 
    # find the saturation value and poistion of for each row vector in the mtrix (mtrx), where columns are the vector elements
    # noisfac- determines what portion of the largest change can be considered as insignificant to determine saturation
    #plotf- plot the curves with the determined saturation point. If plotf==0 no plot, if plotf=[bins] plot according to the bins (x axes)
    # foundsat- quality flag: 1- good saturation was found, 0-reached the final element withou defining saturation,
    # -1 - same as 0 but note that there are points with higher or lower values inbwtween the first and last points (concaved curve).
    w=5
    length=np.shape(mtrx)[0]
    foundsat=np.ones(length)    
    satval=np.zeros(length)
    satdist=np.copy(satval)
    for i in range(0,length):
        tmp=mtrx[i,:]
        # tmp=np.convolve(tmp, np.ones(w), 'valid') / w
        mintmp=np.nanmin(tmp) 
        maxtmp=np.nanmax(tmp) 
        maxmin=maxtmp-mintmp # the difference between the max and min values in the vector, to scale the saturation (stop changing) value
        noisetol=noisfac*maxmin  # noise tolerance as a fraction of the max differenrce between elements
        lastbin=np.array(np.where(1-np.isnan(tmp)))
        lastbin=lastbin[0,-1]
        for j in range(0,lastbin): 
            if (np.nanmax(tmp[j:])-np.nanmin(tmp[j:]))<=noisetol:
                satval[i]=tmp[j]
                satdist[i]=j+1
                # print(satval[i],satdist[i])
                break 
            elif j==lastbin-1: # reaching the final element without saturation sets the last element as the saturation point and value
                satval[i]=tmp[j+1]
                satdist[i]=j+2
                foundsat[i]=0
        sat_strt_diff=satval[i]-tmp[0]
        meangrad=np.nanmean(np.gradient(tmp))
        if  sat_strt_diff*meangrad<=0: # if mean gradient is zero so concaived curve / if start and sat different and mean gradient are different in sign 
            foundsat[i]=-1
        else:
            mintmp=np.nanmin(tmp[0:j])
            maxtmp=np.nanmax(tmp[0:j])
            absmax=np.nanmax([abs(mintmp-tmp[0]),abs(maxtmp-tmp[0])])
            if abs(sat_strt_diff)/absmax<0.8:
                foundsat[i]=-1  # if the curve is non monotonic so no real saturation
        if np.size(plotf)>1:
            plt.figure() 
            plt.plot(plotf[1:], tmp)    
            plt.plot(satdist[i],satval[i],'.' )
    # print(foundsat)
    return(satval,satdist,foundsat) 

def plthist(data,bins):
    # fig = plt.figure()    # histogram
    if len(bins)==0:
        plt.hist(data.ravel(),alpha=0.5)
    else:
        plt.hist(data.ravel(),bins,alpha=0.5)
    # plt.yscale("log")
    # plt.show() 
    
def save_multi_image(filename):
   from matplotlib.backends.backend_pdf import PdfPages 
   pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()

def clearall(): 
    d = dir() #initializing d with dir(),This will store a list of all the variables in the program   
    #You'll need to check for user-defined variables in the directory
    for obj in d:
        #checking for built-in variables/functions
        if not obj.startswith('__'):
            #deleting the said obj, since a user-defined function
            del globals()[obj]
    return()   

def filter2D(mtrx,window,running):
    sz0=np.shape(mtrx)[0]
    sz1=np.shape(mtrx)[1]
    wind0=np.shape(window)[0]
    wind1=np.shape(window)[1]
    if running==False:
        # if np.size(interp)!=0:
        #     interp0=interp[0]
        #     interp1=interp[1]
        #     rem0=round(np.modf(interp0)[0],2)
        #     rem1=round(np.modf(interp1)[0],2)
        #     box0=np.floor(interp0)
        #     box1=np.floor(interp1)
        #     szout0=np.floor(sz0/interp0)
        #     szout1=np.floor(sz1/interp1)
        #     output = np.ones((szout0,szout1))
        #     iin=0
        #     imtrx=0
        #     for i in range(0,szout0):
        #         ifn=iin+interp0
        #         jin=0
        #         jmtrx=0
        #         BOX0=np.arange(imtrx,imtrx+box0)
        #         if np.modf(iin)[0]==0:
        #             ii=1
        #         elif round(1-rem0,2)==iin:
        #             ii=2
        #         else:
        #             ii=3
                    
        #         for j in range(0,szout1):
        #             jfn=jin+interp0
        #             BOX1=np.arange(jmtrx,jmtrx+box1)
        #             sumbox=np.nansum(mtrx[BOX0,BOX1])
        #             if np.modf(jin)[0]==0:
        #                 if ii==1:                           
        #                     output[i,j] = sumbox+np.nansum(mtrx[imtrx+box0,BOX1]*rem0)+np.nansum(mtrx[BOX0,jmtrx+box1]*rem1)+mtrx[imtrx+box0,jmtrx+box1]*rem0*rem1
        #                 elif ii==2:
        #                     output[i,j] = sumbox+np.nansum(mtrx[imtrx-1,BOX1]*rem0)+np.nansum(mtrx[BOX0,jmtrx+box1]*rem1)+mtrx[imtrx+box0,jmtrx+box1]*rem0*rem1
        #             elif round(1-rem1,2)==jin:
        #                 jj=2
        #             else:
        #                 jj=3
        #             jin=jfn
        #             jmtrx=jmtrx+box1
        #         iin=ifn
        #         imtrx=imtrx+box0
        # else:
        if sz0%wind0!=0:
            mtrx=np.concatenate((mtrx,np.ones((wind0-sz0%wind0,sz1))*np.nan),axis=0)
        if sz1%wind1!=0:
            mtrx=np.concatenate((mtrx,np.ones((wind1-sz1%wind1,sz0))*np.nan),axis=1)
        sz0=np.shape(mtrx)[0]
        sz1=np.shape(mtrx)[1]
        output = np.zeros((int(sz0/wind0),int(sz1/wind1)))
        ii=-1
        for i in range(0,sz0,wind0):
            ii+=1
            jj=-1
            for j in range(0,sz1,wind1):
                jj+=1
                output[ii,jj] = np.nansum(mtrx[i:i+wind0,j:j+wind1]*window) 
    else:
        print('No other method')
    return output  

def txt2numbyLines(path):
    file = open(path,'r')
    lines = file.readlines()
    N=len(lines)
    results=[np.nan]*N
    for i in range(0,N):
        L=lines[i]
        spacelist=np.array([i for i, char in enumerate(L) if char in whitespace])
        tmp=np.ones((len(spacelist)))
        tmp[1:-1]=spacelist[2:]
        tmp2=spacelist[np.nonzero(tmp-spacelist>1)[0]]
        nn=len(tmp2)
        if nn<2:
            continue
        vec=np.ones((nn))*np.nan
        for j in range(0,nn-1):
            vec[j]=float(L[tmp2[j]+1:tmp2[j+1]+1])
        vec[j+1]=float(L[tmp2[j+1]:])
        results[i]=vec
    file.close()
    return(results)

def clall():
    plt.close('all')
    
def shp(A):
    a=np.shape(A)
    print(a)
    
def findmindiff(x,X):
    y=np.nonzero(np.min(np.abs(x-X))==np.abs(x-X))
    return (y)#[0][0])#,X[y[0][0]])

def calc4moms(x,w):
    N=np.sum(~np.isnan(x))
    mean=np.nansum(x*w)/np.nansum(w)
    var=np.nansum(w*(x - mean)**2)/np.nansum(w)#*N/(N - 1)
    skew=np.sum(w*((x - mean)/np.sqrt(var))**3)/np.nansum(w)
    kurt=np.sum(w*((x - mean)/np.sqrt(var))**4)/np.nansum(w)-3
    return(N,mean,var,skew,kurt)

# %%

def MOD_CERES_tw(path,MOD02,MOD03,MOD06,MOD35,ceres_data,cldmsktyp,NoColdCld,BroadLW,MaxDFNC,channel_adjusting,weightTSM,Npices,thermal_channels,FOVmsk_t,PSFf,TSMth):
     
     # PSFf=True  
     CERESnanprec=0.001 # precent of CERES fov (energy wise) to allow for nan values and use this FOV
     consider_thinCirr=False
     # print(MOD02)
     CERESjdate=ceres_data[0,:]
     CERESlat=ceres_data[1,:]
     CERESlon=ceres_data[2,:]
     CERESsza=ceres_data[3,:]
     CERESvza=ceres_data[4,:]
     CERESraa=ceres_data[5,:]
     Fsw=ceres_data[6,:]
     Isw=ceres_data[7,:]
     Flw=ceres_data[8,:]
     Ilw=ceres_data[9,:]
     AODfov=ceres_data[10,:]
     Clrfov=ceres_data[11,:]
     
     del ceres_data
     
     foviswmax=0; fovilwmax=0
     maxbin=30
     minN=50
     
     if PSFf: 
         # if np.shape(FOVmsk)[0]%2==0:
         PSF=FOVmsk_t.copy()
         tmp=np.shape(PSF)[0]-1
         # if np.shape(FOVmsk)[1]%2!=0:
         tmp1=np.shape(PSF)[1]
         fovsc=[tmp,tmp1]
         tmp=np.sum(PSF,axis=0)
         PSFc1=np.nonzero(tmp==np.max(tmp))[0][0]   
         PSFc2=tmp1-PSFc1-1
         # tmp=np.fliplr(PSF)
         # tmp=np.sum(tmp,axis=0)
         # PSFc2=np.nonzero(tmp==np.max(tmp))[0][0]  
     else:
         fovsc=np.shape(FOVmsk_t)[0]-1
         
     
     totpiece=-1  # cloud domains with CERES  FOVs
     CloudDomain_i=-1 # all cloud domains with MODIS data
     NopxFOVmsk=np.size(PSF)    # number of modis pixels in CERES FOV


     visls=[0.47,0.66,0.85] # ,0.55
     nirls=[0.905,0.935] #,0.940
     swirls=[1.24,2.13,1.375]  # 1.24 out ,1.64 is out because its bad in AQUA
     thermalls=np.array([3.75,3.96,3.96,4.05,4.46,4.5,6.7,7.32,8.55,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
     
     estimated_Npieces=Npices
     mfovp=100  # max number of CERES FOVs to analize in an image
     mpixp=5000 # max number of MODIS pixels for clear sky (usually around ~10,000 but can get as high as 20,000). The smaplying is random mpixp pixels
     
     
     sw_F=np.ones((estimated_Npieces,mfovp),dtype=np.float32)*np.nan
     sw_TSM1f=sw_F.copy()
     sw_TSM2f=sw_F.copy()
     sw_meanRHO=sw_F.copy()
     # sw_Fclr=sw_F.copy()
     sw_meanDFNC=sw_F.copy()
     sw_meanTSM=sw_F.copy()
     # swCERES_MOD_name=['nan']*Nsamples
     sw_meanSumI=sw_F.copy()
     sw_I=sw_F.copy()
     sw_AODfov=sw_F.copy()
     sw_Clrfov=sw_F.copy()
     # sw_Fclrspec=np.ones((estimated_Npieces,estimated_Npieces*mfovp,len(lessthermalls)+len(solarls)),dtype=np.float32)*np.nan
     
     
     
     lw_F=sw_F.copy()
     lw_TSM1f=sw_F.copy()
     lw_TSM2f=sw_F.copy()
     lw_meanRHO=sw_F.copy()
     # lw_Fclr=sw_F.copy()
     lw_meanDFNC=sw_F.copy()
     lw_meanTSM=sw_F.copy()
     # lwCERES_MOD_name=['nan']*Nsamples
     lw_meanSumI=sw_F.copy()
     lw_I=sw_F.copy()
     # lw_Fclrspec=np.ones((estimated_Npieces,estimated_Npieces*mfovp,len(lessthermalls)+len(solarls)),dtype=np.float32)*np.nan

     CERESVZAsw=sw_F.copy() #np.ones((estimated_Npieces*mfovp),dtype=np.float32)*np.nan
     CERESRAAsw=CERESVZAsw.copy()
     CERESSZAsw=CERESVZAsw.copy()
     CERESSZAlw=CERESRAAsw.copy()
     CERESRAAlw=CERESRAAsw.copy()
     CERESVZAlw=CERESVZAsw.copy()
     # coloc=CERESVZAsw.copy()
     # cotime=CERESVZAsw.copy()
     
     
     
     
     # EPSsw=np.ones((estimated_Npieces,1))*np.nan
     # EPSlw=np.ones((estimated_Npieces,1))*np.nan
     # meanKsw=np.ones((estimated_Npieces,1))*np.nan
     # meanKlw=np.ones((estimated_Npieces,1))*np.nan
     # meanFswclr=np.ones((estimated_Npieces,1))*np.nan
     # meanFlwclr=np.ones((estimated_Npieces,1))*np.nan
     CloudFraction=np.ones((estimated_Npieces,1))*np.nan
     mean_SZA=CloudFraction.copy()
     mean_RAA=CloudFraction.copy()
     swDom_mean_eps=CloudFraction.copy()
     lwDom_mean_eps=CloudFraction.copy()
     
     dtNtm=MOD02[9:22]
     
     MODISday=float(dtNtm[1:11])  
     
# ------------------------ Reading MODIS L2 images ------------------------------------
     temp="EV_500_Aggr1km_RefSB" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     Rsw500=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_250_Aggr1km_RefSB" #"EV_1KM_RefSB"
     Rsw250=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_1KM_RefSB" #"EV_1KM_RefSB"
     Rswkm=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     # vis=np.append(Rsw500[0:2],Rsw250,axis=0) # ORIG
     vis=np.stack((Rsw500[0],Rsw250[0],Rsw250[1]),axis=0)
     
     nir=np.stack((Rswkm[-4],Rswkm[-3]),axis=0)
     # nir=Rswkm[-4:-1]  # ORIG
     # nir=np.expand_dims(Rswkm[-4],axis=0).copy()
     
     # tmp=np.expand_dims(Rswkm[-1], axis=0).copy()  # readded 164
     # swir=np.append(Rsw500[-3:],tmp,axis=0)
     swir=np.stack((Rsw500[-3],Rsw500[-1],Rswkm[-1]),axis=0) # ORIG,  
     # swir=np.expand_dims(Rsw500[-3],axis=0).copy()
     # swir=np.reshape(swir,(1,np.shape(swir)[0],np.shape(swir)[1]))
     # swir=np.transpose(swir, (2, 0, 1)) # ORIG
     
     temp="EV_1KM_Emissive" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     thermal=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     thermalfull=thermal #PlanckTemp(thermalls*1e-6,thermal)
     thermal=thermalfull[thermal_channels]
     del Rswkm,Rsw250,Rsw500
     
     sw_meanSolarspec=np.ones((estimated_Npieces,mfovp,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]),dtype=np.float32)*np.nan
     lw_meanSolarspec=sw_meanSolarspec.copy()
     sw_meanThermalspec=np.ones((estimated_Npieces,mfovp,np.shape(thermal)[0]),dtype=np.float32)*np.nan
     lw_meanThermalspec=sw_meanThermalspec.copy() 
     
     MODclrsky_sw=np.ones((estimated_Npieces,mpixp,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]+np.shape(thermal)[0]),dtype=np.float32)*np.nan
     MODclrsky_lw=MODclrsky_sw.copy()
     
     MODangles_sw=np.ones((estimated_Npieces*3,mpixp,3),dtype=np.float32)*np.nan
     MODangles_lw=MODangles_sw.copy()
     
     CERES_CldDomain_isw=np.ones((estimated_Npieces*3),dtype=np.float32)*np.nan
     CERES_CldDomain_ilw=CERES_CldDomain_isw.copy()
 #---------------------------------------Reading cloud mask ------------------------------
     temp = nc.Dataset(path+'/MOD35/'+MOD35, 'r')
     # print(temp.variables.keys()) # 
     MOD35D = temp.variables["Cloud_Mask"]
     # print(MOD35D) # 
     
     # fillV=temp.variables[MOD35D].getncattr('_FillValue')
     # CldMsk[CldMsk==fillV]=np.nan
     
     MODsz=np.shape(thermal)
     MODsz=MODsz[1:3]
     CldMsk=np.zeros(MODsz)
     glint=np.zeros(MODsz)
     sfcice=np.zeros(MODsz)
     land=np.zeros(MODsz)
     thinCirr=np.zeros(MODsz)
     NonCld=np.zeros(MODsz)
     byte1=np.array(MOD35D[0,:,:])
     byte1[byte1<0]=256+byte1[byte1<0]
     byte2=np.array(MOD35D[1,:,:])
     byte2[byte2<0]=256+byte2[byte2<0]  # correct for negative values
     for i in range(0,MODsz[0]):
         for j in range(0,MODsz[1]):  
             # print(bin(int(byte1[i,j])))
             binary=format(int(byte1[i,j]),'#010b').replace("0b","")
             binary=binary[::-1]
             binary2=format(int(byte2[i,j]),'#010b').replace("0b","")
             binary2=binary[::-1]
             # print(binary)
             if int(binary[0])==1: # & int(binary[-4])==dayf:  # if mask is determined and is day/night
                 CldMsk[i,j]=int(binary[1:3],2)  
                 glint[i,j]=int(binary[4],2)
                 sfcice[i,j]=int(binary[5],2)
                 land[i,j]=int(binary[6:],2)
                 bit10=int(binary2[1])
                 # bit12=int(binary2[3])
                 thinCirr[i,j]=bit10
                 NonCld[i,j]=int(binary2[0])
     
     BWmsk=np.copy(CldMsk)
     if cldmsktyp=='relaxed':  # take only 95% confident as cloud          
        BWmsk[BWmsk>0]=1
        BWmsk=BWmsk.astype(bool)
        BWmsk=np.invert(BWmsk)   
        if consider_thinCirr:
            BWmsk[thinCirr==1]=1
     if cldmsktyp=='strict':    #cloud even if only 33% confident      
        BWmsk[BWmsk==0]=1
        BWmsk[BWmsk>1]=0
        BWmsk=BWmsk.astype(bool)
        if consider_thinCirr:
            BWmsk[thinCirr==1]=1
                 
     # --------------------- Reading clouds --------------------------------
     var="Cloud_Phase_Infrared_1km" 
     dataset = nc.Dataset(path+'MOD06/'+MOD06, 'r')
     # print(dataset.variables.keys()) # 
     mtrx = dataset.variables[var]
     # print(mtrx) # 
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     # mtrx=np.ma.masked_array(mtrx, mtrx==fillV) # mtrx[mtrx==fillV]=np.nan
     mtrx=np.where(mtrx==fillV,np.nan,mtrx) # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldPhs=(mtrx-offset)*scales
     # CldPhs=scipy.ndimage.zoom(CldPhs, 5, order=0)
     
     
     var="cloud_top_height_1km"
     mtrx = dataset.variables[var]
     # print(mtrx) # 
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldTop=(mtrx-offset)*scales
     

     # ------------------ GEOLOCATION ---------------------------------------------------------
     # dataset = nc.Dataset(MODpath+MOD03, 'r')
     # print(dataset.variables.keys()) # 
     # mtrx = dataset.variables[var]
     # print(mtrx) # 
     
     temp="Latitude" #"EV_1KM_RefSB"
     sclsNoffs=0
     Lat=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     temp="Longitude" #"EV_1KM_RefSB"
     Lon=readMODIS(path+'/MOD03/'+MOD03,temp,sclsNoffs)
     
     temp="SensorZenith" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarZenith" #"EV_1KM_RefSB"
     SolZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SolZen=SenZen*0.01
     
     
     temp="SensorAzimuth" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarAzimuth" #"EV_1KM_RefSB"
     SolAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     
     RAA=SolAz-SenAz-180
     RAA=np.where(RAA<0,RAA+360,RAA)
     RAA=np.abs(RAA-180)
     
     del SolAz, SenAz    
      
     
     # #################################### Calc distances ##################################################        
     DFNC=scipy.ndimage.distance_transform_edt(1-BWmsk)
     xx,yy=np.meshgrid(np.array(range(1,MODsz[1]+1,1)),np.array(range(1,MODsz[0]+1,1)))
     xx=np.minimum(xx,np.fliplr(xx))
     yy=np.minimum(yy,np.flipud(yy))
     tmp=np.minimum(yy,xx)   # smallest distance to image edge
     DFNC[DFNC>tmp]=np.nan
     
     tmp=np.copy(land)
     tmp[tmp>0]=1
     tmp=tmp.astype(bool)
     tmp=np.invert(tmp)
     DFL=scipy.ndimage.distance_transform_edt(tmp)
     
     tmp=np.copy(CldPhs)
     tmp[tmp==1]=0
     tmp[tmp==6]=0
     tmp[(tmp==2) | (tmp==3)]=1
     tmp=tmp.astype(bool)
     DFIC=scipy.ndimage.distance_transform_edt(1-tmp)
     
     tmp=np.copy(CldPhs)
     tmp[tmp<6]=0
     tmp[tmp==6]=1
     tmp=tmp.astype(bool)
     DFuncertPhs=scipy.ndimage.distance_transform_edt(1-tmp)
     
     if NoColdCld==1:
         tmp=CldTop>5000
         DFHC=scipy.ndimage.distance_transform_edt(1-tmp)
     elif NoColdCld==2:
         tmp=CldTop<6000
         DFHC=scipy.ndimage.distance_transform_edt(1-tmp)
     elif NoColdCld==0:
         DFHC=[]
     
     glint=1-glint
     glint=glint.astype(bool)
     
     unwanted=np.zeros((MODsz[0],MODsz[1]), dtype=bool)
     unwanted[DFL<10]=True
     unwanted[glint]=True
     unwanted[SenZen>20]=True
     
     if NoColdCld==1:
         unwanted[DFuncertPhs<10]=True   # restrictions!
         unwanted[DFHC<20]=True
         unwanted[DFIC<20]=True
     elif NoColdCld==2:
         unwanted[DFHC<10]=True  # these are low clouds in this case
     
     tmp2=np.stack([unwanted]*np.shape(nir)[0],axis=0)
     nir[tmp2]=np.nan
     tmp2=np.stack([unwanted]*np.shape(swir)[0],axis=0) 
     swir[tmp2]=np.nan 
     tmp2=np.stack([unwanted]*np.shape(vis)[0],axis=0) 
     vis[tmp2]=np.nan
     tmp2=np.stack([unwanted]*np.shape(thermal)[0],axis=0)
     thermal[tmp2]=np.nan
     DFNC[unwanted]=np.nan
     del MOD35D, glint, sfcice, land, thinCirr, NonCld, CldPhs, DFL, DFIC, DFuncertPhs, DFHC
     # -------------------------- Finding pure clear sky --------------------------
     DFNCbins=np.array(range(0,maxbin,1))
     
     if Npices==80:
         DX=110; DY=101; ri=0-DY 
         splitganules=[457,568,679,790];
     elif Npices==20:  
         DX=220; DY=203; ri=0-DY
         splitganules=[457,678]; splitf=bool(1)
     else:
         print('Bad subdomain input')
         

     for piece in range(1,Npices+1):
         # print('sub domian-'+ str(piece) )
         
         
         if Npices==20:   # how to move the subdomain (depend on Npices)
             splitf=(1-splitf)
             ci=splitganules[splitf]
             if splitf==0:
                 ri=ri+DY
         elif Npices==80:
             splitf=np.fmod(piece,4)
             ci=splitganules[splitf-1]
             if splitf==1:
                 ri=ri+DY 
             
         # print(ci,ri) 
         subDFNC=DFNC[ri:ri+DY,ci:ci+DX].copy()
         BWmsksub=BWmsk[ri:ri+DY,ci:ci+DX].copy()
         
         tmp=np.concatenate((vis,nir,swir),axis=0)
         subSolar=tmp[:,ri:ri+DY,ci:ci+DX].copy()
         subThermal=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         
         subSolZen=SolZen[ri:ri+DY,ci:ci+DX].copy()
         subSenZen=SenZen[ri:ri+DY,ci:ci+DX].copy()
         subRAA=RAA[ri:ri+DY,ci:ci+DX].copy()
         
         # CF=np.nansum(BWmsksub)/np.size(BWmsksub)                   
         tmp=BWmsksub.copy()
         tmp[unwanted[ri:ri+DY,ci:ci+DX]]=0
         CF=np.nansum(tmp)/np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))
         # print('CF=',CF)
         if (CF>0.9) or (CF<0.05):
              # print('cloud fration > 80%')                # restriction!
             continue
         if np.nansum(unwanted[ri:ri+DY,ci:ci+DX])>DX*DY*0.5:
             # print('unwanted > 50%')
             continue
         
      # ********************************************************************
         nirclr=np.zeros((2,len(nirls)))
         visclr=np.zeros((2,len(visls)))
         swirclr=np.zeros((2,len(swirls)))
         thermalclr=np.zeros((2,np.shape(thermal)[0]))
         if MaxDFNC>0:
             if np.nansum(subDFNC>MaxDFNC)<50:
                 continue       # restriction!
             # print('shapes',np.shape(vis)[0],np.shape(nir)[0],np.shape(swir)[0],np.shape(thermal)[0])
             tmp2=np.where(subDFNC>MaxDFNC)
             for i in range(0,np.shape(nir)[0]):
                 tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()                    
                 nirclr[0,i]=np.nanmean(tmp[tmp2])
                 nirclr[1,i]=np.nanvar(tmp[tmp2])

             for i in range(0,np.shape(vis)[0]):
                 tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                 visclr[0,i]=np.nanmean(tmp[tmp2])
                 visclr[1,i]=np.nanvar(tmp[tmp2])
                     
             for i in range(0,np.shape(swir)[0]):       
                 tmp=swir[i,ri:ri+DY,ci:ci+DX].copy() 
                 swirclr[0,i]=np.nanmean(tmp[tmp2])
                 swirclr[1,i]=np.nanvar(tmp[tmp2])
                     
             for i in range(0,np.shape(thermal)[0]):      
                 tmp=thermal[i,ri:ri+DY,ci:ci+DX].copy()
                 thermalclr[0,i]=np.nanmean(tmp[tmp2])
                 thermalclr[1,i]=np.nanvar(tmp[tmp2])
                 # ------ start collecting subdomain (piece) data ----------------
                 
         else:
             
             #--------- Radiance vs DFNC for vis, nir,swir,thermal
                 vis_dfnc,visDstd,visDN=bindata(vis[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 
                 tmp=vis_dfnc[0,:].copy()
                 lastbin=np.array(np.where(1-np.isnan(tmp)))
                 if (lastbin.size==0) or (np.nansum(np.isnan(tmp[1:6]))>1):
                     # print('no data as DFNC');
                     continue
                 lastbin=lastbin[0,-1]+1
                 if lastbin<7: 
                     # print('DFNC<7km')
                     continue           # restriction!
                 nir_dfnc,nirDstd,nirDN=bindata(nir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 swir_dfnc,swirDstd,swirDN=bindata(swir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 thermal_dfnc,thermalDstd,thermalDN=bindata(thermal[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
             
     
                 noisfacsw=0.05 # When noise factor decreases the algorithm has less tolorance to noise in saturation so saturation point get farther away.
                 noisfaclw=0.1
                 vissat,vissatdis,vis_fndsat=findsat(vis_dfnc[:,1:],noisfacsw,0) #DFNCbins
                 nirsat,nirsatdis,nir_fndsat=findsat(nir_dfnc[:,1:],noisfacsw,0)
                 swirsat,swirsatdis,swir_fndsat=findsat(swir_dfnc[:,1:],noisfacsw,0)
                 thermalsat,thermalsatdis,thermal_fndsat=findsat(thermal_dfnc[:,1:],noisfaclw,0)
                 
                 
                 if channel_adjusting:
                     tmp=np.reshape(np.concatenate((vis_fndsat,nir_fndsat,swir_fndsat),axis=0),(1,len(vis)+len(nir)+len(swir)))
                    # if totpiece==1:
                     #    FNDSATQsw=tmp
                     #    FNDSATQlw=np.reshape(thermal_fndsat,(1,len(thermal)))
                    # else:
                    #     FNDSATQsw=np.concatenate((FNDSATQsw,tmp),axis=0)
                    #     FNDSATQlw=np.concatenate((FNDSATQlw,np.reshape(thermal_fndsat,(1,len(thermal)))),axis=0)

                     if BroadLW:
                        BadLWchan=5
                     else:
                        BadLWchan=1 
                     if (np.nansum(tmp==-1)>4) or (np.nansum(thermal_fndsat==-1)>BadLWchan):
                         continue
                     
                 
                 for i in range(0,np.shape(thermal)[0]):
                     if i<3:
                         tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>nirsatdis[i])
                         nirclr[0,i]=np.nanmean(tmp[tmp2])
                         nirclr[1,i]=np.nanvar(tmp[tmp2])
                         
                         tmp=swir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>swirsatdis[i])  # changed when removed 1.64 04/12/24
                         swirclr[0,i]=np.nanmean(tmp[tmp2])
                         swirclr[1,i]=np.nanvar(tmp[tmp2])
                     if i<4:
                         tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>vissatdis[i])
                         visclr[0,i]=np.nanmean(tmp[tmp2])
                         visclr[1,i]=np.nanvar(tmp[tmp2])
                         
                         
                     tmp=thermal[i,ri:ri+DY,ci:ci+DX].copy()
                     tmp2=np.where(subDFNC>thermalsatdis[i])
                     thermalclr[0,i]=np.nanmean(tmp[tmp2])
                     thermalclr[1,i]=np.nanvar(tmp[tmp2])
         
         
                     # extentvec=np.concatenate((vissatdis,nirsatdis,swirsatdis,thermalsatdis),axis=0)
        
         # ------------------------- corners ----------------- 
         sublon=Lon[ri:ri+DY,ci:ci+DX] # MODIS SUB-AREA LON AND (LAT)
         sublat=Lat[ri:ri+DY,ci:ci+DX]
          
         North=np.max(sublat)  # MODIS SUB-AREA MAX POSITIONS
         South=np.min(sublat)
         East=np.max(sublon)
         West=np.min(sublon)
          
         if (np.isnan(North)) or (np.isnan(South)) or (np.isnan(West)) or (np.isnan(East)):
              print('<><><><><><><><><><><><><> Max coordinate is Nan <><><><><><><><><><><><><>' )
              continue
          
         if (East>0) & (West<0) & (East-West>150): 
              tmp=East.copy()
              East=West.copy()
              West=tmp.copy()    
        
          # ------ start collecting subdomain (piece) data ----------------
         totpiece=totpiece+1
         CloudFraction[totpiece]=CF
         mean_SZA[totpiece]=np.nanmean(SolZen[ri:ri+DY,ci:ci+DX].ravel())
         mean_RAA[totpiece]=np.nanmean(RAA[ri:ri+DY,ci:ci+DX].ravel())
         # solarls=np.concatenate((visls,nirls,swirls),axis=0)
         clrswspec=np.concatenate((visclr,nirclr,swirclr),axis=1)
         # specclr=np.concatenate((visclr,nirclr,swirclr,thermalclr),axis=1)
    
         
         # ---------------- solar -----------------------------------  
         tmp2=np.concatenate((vis[:,ri:ri+DY,ci:ci+DX],nir[:,ri:ri+DY,ci:ci+DX],swir[:,ri:ri+DY,ci:ci+DX]),axis=0)
         a= clrswspec[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=clrswspec[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMsw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMsw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
             
    
         TSMsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # clrstdsum=np.sum(np.sqrt(specclr[1,:]))
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorsw=tmp3/tmp4            # SHORTWAVE RHO (sumInc/sumIclr)
         TWfactorsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         swDom_mean_eps[totpiece]=np.nanmean(TWfactorsw)
         
         SumIsw=tmp3.copy()           # SHORTWAVE RHO (sumInc/sumIclr)
         SumIsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         SumIsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer=(tmp3<=tmp4+clrstdsum) & (tmp3>=tmp4-clrstdsum)
         # TWfac_uncer[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfactor[(TWfac_uncer) & (1-np.isnan(TWfac_uncer))]=1
      # ---------------- thermal -----------------------------------  
         tmp2=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         a=thermalclr[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=thermalclr[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMlw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMlw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # LongWAVE TWILIGHT-SPECTRAL-MEASURE
             
             
         TSMlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
    
         # clrstdsum=np.sum(np.sqrt(specclr[1,:]))
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorlw=tmp3/tmp4             # LONGWAVE RHO (sumInc/sumIclr)
         TWfactorlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         lwDom_mean_eps[totpiece]=np.nanmean(TWfactorlw)

         SumIlw=tmp3.copy()           # LHORTWAVE RHO (sumInc/sumIclr)
         SumIlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         SumIlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer=(tmp3<=tmp4+clrstdsum) & (tmp3>=tmp4-clrstdsum)
         # TWfac_uncer[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfactor[(TWfac_uncer) & (1-np.isnan(TWfac_uncer))]=1
             
         
         
     # -------------- CERES FOVs ----------------------------
         CEREStoday=np.where(CERESjdate==MODISday)[0]
         
         # my_column_names = ['Ffov','meanfovTWf','TSM>1_count']  # DATA FRAMES
         # DFswnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFswclr = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwclr = pd.DataFrame(data=[], columns=my_column_names)
         
         fovisw=0
         fovilw=0
         for fov in CEREStoday:
             if (np.isnan(Flw[fov]) & np.isnan(Fsw[fov])) or (Fsw[fov]<1) or (CERESvza[fov]>25): continue  # don't analyze CERES fov if its Nan/nightime(SW=0)/Not NADIR VZA
             
             fovlat=CERESlat[fov] # CERES FOV LAT (AND LON)
             fovlon=CERESlon[fov]
             if ((fovlat<North) & (fovlat>South)) & ((fovlon>West) & (fovlon<East)):
                 
                 Cdiff=np.sqrt((sublat-fovlat)**2+(sublon-fovlon)**2)
                 subsubC=np.nonzero(Cdiff==np.min(Cdiff)) # CERES FOV LAT IN MODIS SUB-AREA COORDINATES
                 if subsubC[0].size==0:
                     continue
                 subsubCy=subsubC[0][0].copy(); subsubCx=subsubC[1][0].copy()
                 
                 # Finding the location of neighbor FOV                    # changed 9/7 S
                 fovlat2=CERESlat[fov+1] 
                 fovlon2=CERESlon[fov+1]
                 Cdiff2=np.sqrt((sublat-fovlat2)**2+(sublon-fovlon2)**2)
                 subsubC2=np.nonzero(Cdiff2==np.min(Cdiff2))
                 if subsubC2[0].size==0:
                     print('########## No min diff')
                     continue
                 subsubCx2=subsubC2[1][0]                 # changed 9/7 E
                 if PSFf:
                     if subsubCx2<subsubCx:   # changed 9/7 S
                         PSFc=PSFc2.copy()
                         FOVmsk=np.fliplr(PSF)
                     else:
                         PSFc=PSFc1.copy()
                         FOVmsk=PSF.copy()  # changed 9/7 E
                     subsubRi=int(subsubCy-fovsc[0]/2)
                     subsubRf=int(subsubCy+fovsc[0]/2)
                     subsubCi=int(subsubCx-(PSFc-1))
                     subsubCf=int(subsubCx+(fovsc[1]-PSFc))
                 else:
                     subsubRi=int(subsubCy-fovsc/2)
                     subsubRf=int(subsubCy+fovsc/2)
                     subsubCi=int(subsubCx-fovsc/2)
                     subsubCf=int(subsubCx+fovsc/2)
                 
                 if (subsubRi<0) | (subsubRf>DY-1) | (subsubCi<0) | (subsubCf>DX-1): # if FOV is out of the image continue
                     # print('out of bound!!!',subsubRi,subsubRf,subsubCi,subsubCf)
                     continue
     # ------------------------------- calculating for CERES FOV SW ---------------------------------------- 
                 FswNAN=True
                 if np.isnan(Fsw[fov])==False:
                     FswNAN=False
                     if subsubRf==DY-1:
                        subsubTSMsw=TSMsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWfsw=TWfactorsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                        subSumIsw=SumIsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                        solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                         subSumIsw=SumIsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subSumIsw=SumIsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMsw))
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         if np.sum(FOVmsk[tmpy,tmpx])>CERESnanprec:
                             continue
                         sw_meanTSM[totpiece,fovisw]=np.nansum(subsubTSMsw*FOVmsk)
                         sw_meanRHO[totpiece,fovisw]=np.nansum(subsubTWfsw*FOVmsk)
                         sw_meanSumI[totpiece,fovisw]=np.nansum(subSumIsw*FOVmsk)
                         sw_meanDFNC[totpiece,fovisw]=np.sum(subsubDFNC*FOVmsk)
                         tmp=np.nonzero(subsubTSMsw>1)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         sw_TSM1f[totpiece,fovisw]=np.nansum(subsubTSMsw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMsw*FOVmsk)
                         
                         tmp=np.nonzero(subsubTSMsw>3)  # find where TSM>3 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         sw_TSM2f[totpiece,fovisw]=np.nansum(subsubTSMsw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMsw*FOVmsk)
                         
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                         tmp2=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                         sw_meanSolarspec[totpiece,fovisw,:]=tmp2
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                         sw_meanThermalspec[totpiece,fovisw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                         
                     else:
                         if np.sum(np.isnan(subsubTSMsw[FOVmsk]))>0:
                             continue
                         subsubTSMsw[(~FOVmsk)]=np.nan 
                         subsubTWfsw[(~FOVmsk)]=np.nan
                         subSumIsw[(~FOVmsk)]=np.nan
                         subsubDFNC[(~FOVmsk)]=np.nan
                         sw_TSM1f[totpiece,fovisw]=np.nansum(subsubTSMsw>1)/NopxFOVmsk
                         sw_meanRHO[totpiece,fovisw]=np.nanmean(subsubTWfsw)
                         sw_meanTSM[totpiece,fovisw]=np.nanmean(subsubTSMsw)
                         sw_meanDFNC[totpiece,fovisw]=np.nanmean(subsubDFNC)
                         sw_meanSumI[totpiece,fovisw]=np.nanmean(subSumIsw)
                         
                     CERESSZAsw[totpiece,fovisw]=CERESsza[fov]     
                     CERESRAAsw[totpiece,fovisw]=CERESraa[fov]    
                     CERESVZAsw[totpiece,fovisw]=CERESvza[fov]
                     # cotime[totpiece,fovisw]=CERESjdate[fov]-MODISday
                     # coloc[totpiece,fovisw]=np.min(Cdiff)
                     
                     # swCERES_MOD_name[totpiece,fovisw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                     sw_F[totpiece,fovisw]=Fsw[fov] 
                     sw_I[totpiece,fovisw]=Isw[fov]
                     sw_AODfov[totpiece,fovisw]=AODfov[fov]
                     sw_Clrfov[totpiece,fovisw]=Clrfov[fov]
                     
                     # sw_Fclrspec[totpiece,fovisw,:]=specclr[0,:]
                  #  sw_FclrspecSTD[totpiece,fovisw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
                     
                     # if sw_meanDFNC[totpiece,fovisw]<12:
                     #    fig2 = plt.figure(figsize=(21,11))
                     #    ax1 = fig2.add_subplot(2, 2, 2)
                     #    imgplot1 = plt.imshow(subDFNC,cmap=mpl.colormaps['turbo'])
                     #    imgplot1.set_clim(0, 50)
                     #    plt.plot(subsubCx,subsubCy,'.m')
                     #    plt.plot([subsubCx-22,subsubCx-22,subsubCx+22,subsubCx+22],[subsubCy-15,subsubCy+15,subsubCy-15,subsubCy+15],'*m')
                     #    ax1 = fig2.add_subplot(2, 2, 1)
                     #    imgplot1 = plt.imshow(TSMsw,cmap=mpl.colormaps['turbo'])
                     #    imgplot1.set_clim(1, 5)
                     #    plt.plot(subsubCx,subsubCy,'.m')
                     #    plt.plot([subsubCx-22,subsubCx-22,subsubCx+22,subsubCx+22],[subsubCy-15,subsubCy+15,subsubCy-15,subsubCy+15],'*m')
                     #    ax2 = fig2.add_subplot(2, 2, 3)
                     #    imgplot2 = plt.imshow(subsubTSMsw,cmap=mpl.colormaps['turbo'])
                     #    imgplot2.set_clim(1, 5)
                     #    plt.title('TSMf='+str(np.nansum(subsubTSMsw>1)/NopxFOVmsk)+' F='+str(Fsw[fov]))
                     #    plt.colorbar( orientation='horizontal')
                     #    ax3 = fig2.add_subplot(2, 2, 4)
                     #    imgplot3 = plt.imshow(subsubDFNC,cmap=mpl.colormaps['turbo'])
                     #    imgplot3.set_clim(0, 50)
                     #    plt.title('mean DFNC= '+str(np.nansum(subsubDFNC*FOVmsk)))
                     #    plt.colorbar( orientation='horizontal')
                      
                     #    plt.savefig('/export/home/eeytan/projects/Twilight_MODIS_CERES/figs' +'/'+ str(fov) + '.png')
                     #    plt.close(fig2)    
                  
                     fovisw=fovisw+1
                     
    
    # --------------------------------- Analyzing LW fov---------------------------------------------                 
                 if np.isnan(Flw[fov])==False:
                        
                     if subsubRf==DY-1:
                        subsubTSMlw=TSMlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWflw=TWfactorlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subSumIlw=SumIlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        if FswNAN:
                            subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                            thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                            solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:].copy() 
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subSumIlw=SumIlw[subsubRi:subsubRf+1,subsubCi:].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subSumIlw=SumIlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMlw))
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         if np.sum(FOVmsk[tmpy,tmpx])>CERESnanprec:  
                             continue
                         lw_meanTSM[totpiece,fovilw]=np.nansum(subsubTSMlw*FOVmsk)
                         lw_meanRHO[totpiece,fovilw]=np.nansum(subsubTWflw*FOVmsk)
                         lw_meanSumI[totpiece,fovilw]=np.nansum(subSumIlw*FOVmsk)
                         lw_meanDFNC[totpiece,fovilw]=np.sum(subsubDFNC*FOVmsk) 
                         tmp=np.nonzero(subsubTSMlw>1)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         lw_TSM1f[totpiece,fovilw]=np.nansum(subsubTSMlw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMlw*FOVmsk)
                         tmp=np.nonzero(subsubTSMlw>3)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         lw_TSM2f[totpiece,fovilw]=np.nansum(subsubTSMlw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMlw*FOVmsk)
                         
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                         lw_meanSolarspec[totpiece,fovilw,:]=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                         lw_meanThermalspec[totpiece,fovilw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                     else:
                         if np.sum(np.isnan(subsubTSMlw[FOVmsk]))>0:
                             continue                              
                         subsubTSMlw[(~FOVmsk)]=np.nan 
                         subsubTWflw[(~FOVmsk)]=np.nan 
                         subSumIlw[(~FOVmsk)]=np.nan
                         lw_meanRHO[totpiece,fovilw]=np.nanmean(subsubTWflw)
                         lw_meanTSM[totpiece,fovilw]=np.nanmean(subsubTSMlw)
                         lw_meanSumI[totpiece,fovilw]=np.nanmean(subSumIlw)
                         if FswNAN:
                             subsubDFNC[(~FOVmsk)]=np.nan
                         lw_meanDFNC[totpiece,fovilw]=np.nanmean(subsubDFNC[FOVmsk])
                         TSM1sum=np.nansum(subsubTSMlw>1)
                         lw_TSM1f[totpiece,fovilw]=TSM1sum/ NopxFOVmsk
                         
                     CERESSZAlw[totpiece,fovilw]=CERESsza[fov]
                     CERESRAAlw[totpiece,fovilw]=CERESraa[fov]    
                     CERESVZAlw[totpiece,fovilw]=CERESvza[fov]

                     # lwCERES_MOD_name[totpiece,fovilw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                     lw_F[totpiece,fovilw]=Flw[fov]  
                     lw_I[totpiece,fovilw]=Ilw[fov] 
                                                   
                     # lw_Fclrspec[totpiece,fovilw,:]=specclr[0,:]
                  #  lw_FclrspecSTD[totpiece,fovilw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
    
                     fovilw=fovilw+1  
         
 ##################################### Taking MODIS clear sky pixels for ML flux retrievals   ###################################                    
         # if fovisw>0:
         CloudDomain_i=CloudDomain_i+1
         # print('<><><><><><><><><><><><><> ',CloudDomain_i,totpiece,' <><><><><><><><><><><><><>' )
         MODclri=np.nonzero(TSMsw<TSMth)
         if np.size(MODclri)>100:   # analyze only cloud fields with more than 100 ckear-sky MODIS pixels
             tmp=np.concatenate((subSolar,subThermal),axis=0)
             esh=np.tile(subSolZen[np.newaxis,:,:],(3,1,1))
             esh[1,:,:,]=np.reshape(subSenZen,(1,np.shape(subSenZen)[0],np.shape(subSenZen)[1]))
             esh[2,:,:,]=np.reshape(subRAA,(1,np.shape(subRAA)[0],np.shape(subRAA)[1]))
             # print('# MODIS pixels',len(MODclri[0]))
             if len(MODclri[0])<=np.shape(MODclrsky_sw)[1]:
                for modpx in range(0,len(MODclri[0])):
                     MODclrsky_sw[CloudDomain_i,modpx,:]=tmp[:,MODclri[0][modpx],MODclri[1][modpx]] 
                     MODangles_sw[CloudDomain_i,modpx,:]=esh[:,MODclri[0][modpx],MODclri[1][modpx]] 
             else:
                #print('##### more MODsw pixels than array size',len(MODclri[0]) , '########')
                tmp0=MODclri[0].T
                tmp1=MODclri[1].T
                tmp2=np.tile(tmp0[:,np.newaxis],(1,2))
                tmp2[:,1]=tmp1
                np.random.shuffle(tmp2)
                np.random.shuffle(tmp2)
                for modpx in range(0,np.shape(MODclrsky_sw)[1]):
                    MODclrsky_sw[CloudDomain_i,modpx,:]=tmp[:,tmp2[modpx,0],tmp2[modpx,1]] 
                    MODangles_sw[CloudDomain_i,modpx,:]=esh[:,tmp2[modpx,0],tmp2[modpx,1]] 
                        # print('Too much pixels SW',modpx)
             if fovisw>3:
                CERES_CldDomain_isw[CloudDomain_i]=totpiece
         # if fovilw>0:
         MODclri=np.nonzero(TSMlw<TSMth)
         if np.size(MODclri)>100:
            tmp=np.concatenate((subSolar,subThermal),axis=0)
            esh=np.tile(subSolZen[np.newaxis,:,:],(3,1,1))
            esh[1,:,:,]=np.reshape(subSenZen,(1,np.shape(subSenZen)[0],np.shape(subSenZen)[1]))
            esh[2,:,:,]=np.reshape(subRAA,(1,np.shape(subRAA)[0],np.shape(subRAA)[1]))
            if len(MODclri[0])<=np.shape(MODclrsky_lw)[1]:
                for modpx in range(0,len(MODclri[0])):
                    MODclrsky_lw[CloudDomain_i,modpx,:]=tmp[:,MODclri[0][modpx],MODclri[1][modpx]] 
                    MODangles_lw[CloudDomain_i,modpx,:]=esh[:,MODclri[0][modpx],MODclri[1][modpx]] 
            else:
                # print('##### more MODlw pixels than array size',len(MODclri[0]) , '########')
                tmp0=MODclri[0].T
                tmp1=MODclri[1].T
                tmp2=np.tile(tmp0[:,np.newaxis],(1,2))
                tmp2[:,1]=tmp1
                np.random.shuffle(tmp2)
                np.random.shuffle(tmp2)
                for modpx in range(0,np.shape(MODclrsky_lw)[1]):
                    MODclrsky_lw[CloudDomain_i,modpx,:]=tmp[:,tmp2[modpx,0],tmp2[modpx,1]]
                    MODangles_lw[CloudDomain_i,modpx,:]=esh[:,tmp2[modpx,0],tmp2[modpx,1]]
            if fovilw>3:
                CERES_CldDomain_ilw[CloudDomain_i]=totpiece
         
         foviswmax=np.maximum(foviswmax,fovisw)
         fovilwmax=np.maximum(fovilwmax,fovilw)

     # print('%%%%%%%%%%%%%%%%%%%%%%%%',fovisw,fovilw)
     

     # swallnanrawMOD=np.nonzero(np.nansum(np.isnan(MODclrsky_sw[:,:,1]),axis=1)<np.shape(MODclrsky_sw[:,:,1])[1])[0]                  
     # lwallnanrawMOD=np.nonzero(np.nansum(np.isnan(MODclrsky_lw[:,:,8]),axis=1)<np.shape(MODclrsky_lw[:,:,8])[1])[0] 
     
     # swallnanraw=np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1])[0]                  
     # lwallnanraw=np.nonzero(np.nansum(np.isnan(lw_F),axis=1)<np.shape(lw_F)[1])[0] 
     # for ii in range(0,len(CERES_CldDomain_isw)):
     #     if ~np.isnan(CERES_CldDomain_isw[ii]):
     #         if ~(swallnanraw==CERES_CldDomain_isw[ii]).any():
     #             CERES_CldDomain_isw[ii]=np.nan
                 
     # for ii in range(0,len(CERES_CldDomain_ilw)):
     #     if ~np.isnan(CERES_CldDomain_ilw[ii]):
     #         if ~(lwallnanraw==CERES_CldDomain_ilw[ii]).any():
     #             CERES_CldDomain_ilw[ii]=np.nan
     
     swallnanraw=np.nonzero(np.nansum(np.isnan(MODclrsky_sw[:,:,1]),axis=1)<np.shape(MODclrsky_sw[:,:,1])[1])[0]         
     lwallnanraw=np.nonzero(np.nansum(np.isnan(MODclrsky_lw[:,:,8]),axis=1)<np.shape(MODclrsky_lw[:,:,8])[1])[0]
     # print('<><><><><><><><><><><>  swNi cloud fields',swallnanraw,'<><><><><><><><><><><>')
     # print('<><><><><><><><><><><>  lwNi cloud fields',lwallnanraw,'<><><><><><><><><><><>')
     
    # print('###### SW arrays ', len(swallnanraw),len(np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1])[0]))
    # print('###### LW arrays ', len(lwallnanraw),len(np.nonzero(np.nansum(np.isnan(lw_F),axis=1)<np.shape(lw_F)[1])[0]))
     # swallnanraw=np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1]                
     # lwallnanraw=np.nansum(np.isnan(lw_F),axis=1)<np.shape(lw_F)[1]

     # swallnanraw=np.nonzero((swallnanrawMOD) | (swallnanraw))[0]      
     # lwallnanraw=np.nonzero((lwallnanrawMOD) | (lwallnanraw))[0] 
     
     # swallnanspec=np.nonzero(np.nansum(np.nansum(np.isnan(sw_meanSolarspec),axis=2),axis=1)<(np.shape(sw_meanSolarspec)[1]*np.shape(sw_meanSolarspec)[2]))[0] 
     # if np.sum(swallnanraw-swallnanspec)!=0:
         # input()
     
     # if np.nansum(swallnanraw==lwallnanraw)!=len(swallnanraw):
         # print('DOMAIN DONT OVERLAP IN SW AND LW')
         # input()
         
     if np.size(swallnanraw)+np.size(lwallnanraw)==0:
         # print('nothing_found')
         return('nothing_found')
     else:
         SWdic = {"F": sw_F[swallnanraw,:], "I": sw_I[swallnanraw,:], "TSM1f": sw_TSM1f[swallnanraw,:], "TSM2f": sw_TSM2f[swallnanraw,:], "mRHO":sw_meanRHO[swallnanraw,:],
                  "mDFNC": sw_meanDFNC[swallnanraw,:], "mTSM": sw_meanTSM[swallnanraw,:], 
                  "mSumI":sw_meanSumI[swallnanraw,:], "VZA":CERESVZAsw[swallnanraw,:], "SZA":CERESSZAsw[swallnanraw,:], "RAA":CERESRAAsw[swallnanraw,:],
                  "CF": CloudFraction[swallnanraw], "SZApiece": mean_SZA[swallnanraw], "RAApiece": mean_RAA[swallnanraw], "swDomEPS": swDom_mean_eps[swallnanraw],
                  "AOD": sw_AODfov[swallnanraw,:], "Clrf": sw_Clrfov[swallnanraw,:],"SolarCLRspec": sw_meanSolarspec[swallnanraw,:,:],"ThermalCLRspec": sw_meanThermalspec[swallnanraw,:,:],
                  "MODCLRspec": MODclrsky_sw[swallnanraw,:,:], "MODangles":MODangles_sw[swallnanraw,:,:], "MODclrfovi":CERES_CldDomain_isw[swallnanraw]  } 
          
         LWdic = {"F": lw_F[lwallnanraw,:], "I": lw_I[lwallnanraw,:], "TSM1f": lw_TSM1f[lwallnanraw,:], "TSM2f": lw_TSM2f[lwallnanraw,:], "mRHO": lw_meanRHO[lwallnanraw,:],
                  "mDFNC": lw_meanDFNC[lwallnanraw,:], "mTSM": lw_meanTSM[lwallnanraw,:], "mSumI":lw_meanSumI[lwallnanraw,:], "SZA":CERESSZAlw[lwallnanraw,:], 
                  "VZA":CERESVZAlw[lwallnanraw,:], "RAA":CERESRAAlw[lwallnanraw,:],"CF": CloudFraction[lwallnanraw],"lwDomEPS": lwDom_mean_eps[lwallnanraw],
                  "SolarCLRspec": lw_meanSolarspec[lwallnanraw,:,:],"ThermalCLRspec": lw_meanThermalspec[lwallnanraw,:,:],"MODCLRspec": MODclrsky_lw[lwallnanraw,:,:], 
                  "MODangles":MODangles_lw[lwallnanraw,:,:],"MODclrfovi":CERES_CldDomain_ilw[lwallnanraw]  }
                  
         return(SWdic,LWdic) #,Pieceprop)

     
# %% $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
def MOD_CERES_clrsky(path,MOD02,MOD03,MOD06,MOD35,ceres_data,cldmsktyp,NoColdCld,MaxDFNC,channel_adjusting,weightTSM,Npices,FOVmsk_t,PSFf,TSMgt1_ratio):
    
    
     # PSFf=True  
     CERESnanprec=0.001 # precent of CERES fov (energy wise) to allow for nan values and use this FOV
     # TSMgt1_ratio=1  # TSM>1 fraction thershold for defining a pure clear sky FOV. Tollarence to how much contribution of TSM>1 to mean TSM
     # print('CERESnanprec delos', CERESnanprec,TSMgt1_ratio)
     
     # print(MOD02)
     CERESjdate=ceres_data[0,:]
     CERESlat=ceres_data[1,:]
     CERESlon=ceres_data[2,:]
     CERESsza=ceres_data[3,:]
     CERESvza=ceres_data[4,:]
     CERESraa=ceres_data[5,:]
     Fsw=ceres_data[6,:]
     Isw=ceres_data[7,:]
     Flw=ceres_data[8,:]
     Ilw=ceres_data[9,:]
     AODfov=ceres_data[10,:]
     Clrfov=ceres_data[11,:]
     
     foviswmax=0; fovilwmax=0
     maxbin=30
     minN=50
     
     if PSFf: 
         # if np.shape(FOVmsk)[0]%2==0:
         PSF=FOVmsk_t.copy()
         tmp=np.shape(PSF)[0]-1
         # if np.shape(FOVmsk)[1]%2!=0:
         tmp1=np.shape(PSF)[1]
         fovsc=[tmp,tmp1]
         tmp=np.sum(PSF,axis=0)
         PSFc1=np.nonzero(tmp==np.max(tmp))[0][0] 
         PSFc2=tmp1-PSFc1-1
         # tmp=np.fliplr(PSF)
         # tmp=np.sum(tmp,axis=0)
         # PSFc2=np.nonzero(tmp==np.max(tmp))[0][0]  
     else:
         fovsc=np.shape(FOVmsk_t)[0]-1
         
     
     totpiece=-1
     NopxFOVmsk=np.size(PSF)    # number of modis pixels in CERES FOV


     visls=[0.47,0.55,0.66,0.85]
     nirls=[0.905,0.935,0.940]
     swirls=[1.24,1.64,2.13,1.375]
     thermalls=np.array([3.75,3.96,3.96,4.05,4.46,4.5,6.7,7.32,8.55,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
     
     estimated_Npieces=Npices
     mfovp=100
     
     sw_F=np.ones((estimated_Npieces,mfovp),dtype=np.float32)*np.nan
     sw_TSM1f=sw_F.copy()
     sw_meanRHO=sw_F.copy()
     # sw_Fclr=sw_F.copy()
     sw_meanDFNC=sw_F.copy()
     sw_meanTSM=sw_F.copy()
     # swCERES_MOD_name=['nan']*Nsamples
     sw_meanSumI=sw_F.copy()
     sw_I=sw_F.copy()
     sw_AODfov=sw_F.copy()
     sw_Clrfov=sw_F.copy()
     sw_meanSZA=sw_F.copy()
     sw_meanVZA=sw_F.copy()
     sw_meanRAA=sw_F.copy()
       
     
     
     lw_F=sw_F.copy()
     lw_TSM1f=sw_F.copy()
     lw_meanRHO=sw_F.copy()
     # lw_Fclr=sw_F.copy()
     lw_meanDFNC=sw_F.copy()
     lw_meanTSM=sw_F.copy()
     # lwCERES_MOD_name=['nan']*Nsamples
     lw_meanSumI=sw_F.copy()
     lw_I=sw_F.copy()
     lw_meanSZA=sw_F.copy()
     lw_meanVZA=sw_F.copy()
     lw_meanRAA=sw_F.copy()
     
     CERESVZAsw=sw_F.copy() #np.ones((estimated_Npieces*mfovp),dtype=np.float32)*np.nan
     CERESRAAsw=CERESVZAsw.copy()
     CERESSZAsw=CERESVZAsw.copy()
     CERESlonsw=CERESVZAsw.copy()
     CERESlatsw=CERESVZAsw.copy()
     CERESSZAlw=CERESVZAsw.copy()
     CERESRAAlw=CERESRAAsw.copy()
     CERESVZAlw=CERESVZAsw.copy()
     CERESlonlw=CERESVZAsw.copy()
     CERESlatlw=CERESVZAsw.copy()
     # coloc=CERESVZAsw.copy()
     # cotime=CERESVZAsw.copy()
     
     
     
     
     # EPSsw=np.ones((estimated_Npieces,1))*np.nan
     # EPSlw=np.ones((estimated_Npieces,1))*np.nan
     # meanKsw=np.ones((estimated_Npieces,1))*np.nan
     # meanKlw=np.ones((estimated_Npieces,1))*np.nan
     # meanFswclr=np.ones((estimated_Npieces,1))*np.nan
     # meanFlwclr=np.ones((estimated_Npieces,1))*np.nan
     CloudFraction=np.ones((estimated_Npieces,1))*np.nan
     mean_SZA=CloudFraction.copy()
     mean_RAA=CloudFraction.copy()
     swDom_mean_eps=CloudFraction.copy()
     lwDom_mean_eps=CloudFraction.copy()
     
     dtNtm=MOD02[9:22]
     
     
     MODISday=float(dtNtm[1:11])
# ------------------------ Reading MODIS L2 images ------------------------------------
     temp="EV_500_Aggr1km_RefSB" #"EV_1KM_RefSB"
     sclsNoffs="radiance" # "radiance"
     Rsw500=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_250_Aggr1km_RefSB" #"EV_1KM_RefSB"
     Rsw250=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_1KM_RefSB" #"EV_1KM_RefSB"
     Rswkm=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     vis=np.append(Rsw500[0:2],Rsw250,axis=0)
     nir=Rswkm[-4:-1]
     tmp=np.expand_dims(Rswkm[-1], axis=0).copy()
     swir=np.append(Rsw500[-3:],tmp,axis=0)
     
     temp="EV_1KM_Emissive" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     thermal=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     # thermalfull= Planckfunc(thermalls*1e-6,thermal,True) #  thermal #  
     # thermal=thermalfull
     del Rswkm,Rsw250,Rsw500
     
     sw_meanSolarspec=np.ones((estimated_Npieces,mfovp,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]),dtype=np.float32)*np.nan
     lw_meanSolarspec=sw_meanSolarspec.copy()
     sw_meanThermalspec=np.ones((estimated_Npieces,mfovp,np.shape(thermal)[0]),dtype=np.float32)*np.nan
     lw_meanThermalspec=sw_meanThermalspec.copy()   
     
 #---------------------------------------Reading cloud mask ------------------------------
     temp = nc.Dataset(path+'/MOD35/'+MOD35, 'r')
     MOD35D = temp.variables["Cloud_Mask"]
     
     # fillV=temp.variables[MOD35D].getncattr('_FillValue')
     # CldMsk[CldMsk==fillV]=np.nan
     
     MODsz=np.shape(thermal)
     MODsz=MODsz[1:3]
     CldMsk=np.zeros(MODsz)
     glint=np.zeros(MODsz)
     sfcice=np.zeros(MODsz)
     land=np.zeros(MODsz)
     thinCirr=np.zeros(MODsz)
     NonCld=np.zeros(MODsz)
     byte1=np.array(MOD35D[0,:,:])
     byte1[byte1<0]=256+byte1[byte1<0]
     byte2=np.array(MOD35D[1,:,:])
     byte2[byte2<0]=256+byte2[byte2<0]  # correct for negative values
     for i in range(0,MODsz[0]):
         for j in range(0,MODsz[1]):  
             binary=format(int(byte1[i,j]),'#010b').replace("0b","")
             binary=binary[::-1]
             binary2=format(int(byte2[i,j]),'#010b').replace("0b","")
             binary2=binary[::-1]
             if int(binary[0])==1: # & int(binary[-4])==dayf:  # if mask is determined and is day/night
                 CldMsk[i,j]=int(binary[1:3],2)  
                 glint[i,j]=int(binary[4],2)
                 sfcice[i,j]=int(binary[5],2)
                 land[i,j]=int(binary[6:],2)
                 bit10=int(binary2[1])
                 # bit12=int(binary2[3])
                 thinCirr[i,j]=bit10
                 NonCld[i,j]=int(binary2[0])
     
     BWmsk=np.copy(CldMsk)
     if cldmsktyp=='relaxed':  # take only 95% confident as cloud          
        BWmsk[BWmsk>0]=1
        BWmsk=BWmsk.astype(bool)
        BWmsk=np.invert(BWmsk)    
     if cldmsktyp=='strict':    #cloud even if only 33% confident      
        BWmsk[BWmsk==0]=1
        BWmsk[BWmsk>1]=0
        BWmsk=BWmsk.astype(bool)
         
     # --------------------- Reading clouds --------------------------------
     var="Cloud_Phase_Infrared_1km" 
     dataset = nc.Dataset(path+'MOD06/'+MOD06, 'r')
     mtrx = dataset.variables[var]
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     # mtrx=np.ma.masked_array(mtrx, mtrx==fillV) # mtrx[mtrx==fillV]=np.nan
     mtrx=np.where(mtrx==fillV,np.nan,mtrx) # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldPhs=(mtrx-offset)*scales
     # CldPhs=scipy.ndimage.zoom(CldPhs, 5, order=0)
     
     
     var="cloud_top_height_1km"
     mtrx = dataset.variables[var]
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldTop=(mtrx-offset)*scales
     
     # ------------------ GEOLOCATION ---------------------------------------------------------    
     temp="Latitude" #"EV_1KM_RefSB"
     sclsNoffs=0
     Lat=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     temp="Longitude" #"EV_1KM_RefSB"
     Lon=readMODIS(path+'/MOD03/'+MOD03,temp,sclsNoffs)
     
     temp="SensorZenith" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarZenith" #"EV_1KM_RefSB"
     SolZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SolZen=SenZen*0.01
     
     
     temp="SensorAzimuth" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarAzimuth" #"EV_1KM_RefSB"
     SolAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     
     RAA=SolAz-SenAz-180
     RAA=np.where(RAA<0,RAA+360,RAA)
     RAA=np.abs(RAA-180)
     
     del SolAz, SenAz    
    
     
     # #################################### Calc distances ##################################################        
     DFNC=scipy.ndimage.distance_transform_edt(1-BWmsk)
     xx,yy=np.meshgrid(np.array(range(1,MODsz[1]+1,1)),np.array(range(1,MODsz[0]+1,1)))
     xx=np.minimum(xx,np.fliplr(xx))
     yy=np.minimum(yy,np.flipud(yy))
     tmp=np.minimum(yy,xx)   # smallest distance to image edge
     DFNC[DFNC>tmp]=np.nan
     
     tmp=np.copy(land)
     tmp[tmp>0]=1
     tmp=tmp.astype(bool)
     tmp=np.invert(tmp)
     DFL=scipy.ndimage.distance_transform_edt(tmp)
     
     # tmp=np.copy(CldPhs)
     # tmp[tmp==1]=0
     # tmp[tmp==6]=0
     # tmp[(tmp==2) | (tmp==3)]=1
     # tmp=tmp.astype(bool)
     # DFIC=scipy.ndimage.distance_transform_edt(1-tmp)
     
     # tmp=np.copy(CldPhs)
     # tmp[tmp<6]=0
     # tmp[tmp==6]=1
     # tmp=tmp.astype(bool)
     # DFuncertPhs=scipy.ndimage.distance_transform_edt(1-tmp)
     
     # tmp=CldTop>5000
     # DFHC=scipy.ndimage.distance_transform_edt(1-tmp)
     
     glint=1-glint
     glint=glint.astype(bool)
     
     unwanted=np.zeros((MODsz[0],MODsz[1]), dtype=bool)
     unwanted[DFL<10]=True
     unwanted[glint]=True
     unwanted[SenZen>20]=True
     
     # if NoColdCld:
     #     unwanted[DFuncertPhs<10]=True   # restrictions!
     #     unwanted[DFHC<20]=True
     #     unwanted[DFIC<20]=True
     
     
     tmp2=np.stack([unwanted]*3,axis=0)
     nir[tmp2]=np.nan
     tmp2=np.stack([unwanted]*4,axis=0)
     swir[tmp2]=np.nan
     vis[tmp2]=np.nan
     tmp2=np.stack([unwanted]*np.shape(thermal)[0],axis=0)
     thermal[tmp2]=np.nan
     DFNC[unwanted]=np.nan
     # -------------------------- Finding pure clear sky --------------------------
     DFNCbins=np.array(range(0,maxbin,1))
     
     if Npices==80:
         DX=110; DY=101; ri=0-DY 
         splitganules=[457,568,679,790];
     elif Npices==20:  
         DX=220; DY=203; ri=0-DY
         splitganules=[457,678]; splitf=bool(1)
     else:
         print('Bad subdomain input')
         

     for piece in range(1,Npices+1):		 # Looping through subdomains of the MODIS granule       
         
         if Npices==20:   # how to move the subdomain (depend on Npices)
             splitf=(1-splitf)
             ci=splitganules[splitf]
             if splitf==0:
                 ri=ri+DY
         elif Npices==80:
             splitf=np.fmod(piece,4)
             ci=splitganules[splitf-1]
             if splitf==1:
                 ri=ri+DY 
         
         subSolZen=SolZen[ri:ri+DY,ci:ci+DX].copy()
         subSenZen=SenZen[ri:ri+DY,ci:ci+DX].copy()
         subRAA=RAA[ri:ri+DY,ci:ci+DX].copy()
         subDFNC=DFNC[ri:ri+DY,ci:ci+DX].copy()
         BWmsksub=BWmsk[ri:ri+DY,ci:ci+DX].copy()
         
         tmp=np.concatenate((vis,nir,swir),axis=0)
         subSolar=tmp[:,ri:ri+DY,ci:ci+DX].copy()
         subThermal=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         
         # CF=np.nansum(BWmsksub)/np.size(BWmsksub)                   
         tmp=BWmsksub.copy()
         tmp[unwanted[ri:ri+DY,ci:ci+DX]]=0
         CF=np.nansum(tmp)/np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))
         if (CF>0.9):
              # print('cloud fration > 80%')                # restriction!
             continue
         if np.nansum(unwanted[ri:ri+DY,ci:ci+DX])>DX*DY*0.5:
             # print('unwanted > 50%')
             continue
         
      # ********************************************************************
         
         nirclr=np.zeros((2,np.shape(nirls)[0]))
         visclr=np.zeros((2,np.shape(visls)[0]))
         swirclr=np.zeros((2,np.shape(swirls)[0]))
         thermalclr=np.zeros((2,np.shape(thermal)[0]))
         clrswspec=np.zeros((2,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]))
         if MaxDFNC>0:
             # print('take FDNC threshold for clear sky')
             if np.nansum(subDFNC>MaxDFNC)<50:
                 continue
             
             for i in range(0,np.shape(thermal)[0]):
                 # if i<3:
                 #     tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()
                 #     tmp2=np.where(DFNC[ri:ri+DY,ci:ci+DX]>MaxDFNC)
                 #     nirclr[0,i]=np.nanmean(tmp[tmp2])
                 #     nirclr[1,i]=np.nanvar(tmp[tmp2])
                 # if i<4:
                 #     tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                 #     tmp2=np.where(DFNC[ri:ri+DY,ci:ci+DX]>MaxDFNC)
                 #     visclr[0,i]=np.nanmean(tmp[tmp2])
                 #     visclr[1,i]=np.nanvar(tmp[tmp2])
                 #     tmp=swir[i,ri:ri+DY,ci:ci+DX].copy()
                 #     tmp2=np.where(DFNC[ri:ri+DY,ci:ci+DX]>MaxDFNC)
                 #     swirclr[0,i]=np.nanmean(tmp[tmp2])
                 #     swirclr[1,i]=np.nanvar(tmp[tmp2])
                 tmp=subThermal[i,:,:].copy()
                 tmp2=np.where(subDFNC>MaxDFNC)
                 thermalclr[0,i]=np.nanmean(tmp[tmp2])
                 thermalclr[1,i]=np.nanvar(tmp[tmp2])
                 if i<np.shape(subSolar)[0]:
                     tmp=subSolar[i,:,:].copy()
                     clrswspec[0,i]=np.nanmean(tmp[tmp2])
                     clrswspec[1,i]=np.nanvar(tmp[tmp2])
        
                 
                 # ------ start collecting subdomain (piece) data ----------------
             totpiece=totpiece+1
             CloudFraction[totpiece]=CF
             mean_SZA[totpiece]=np.nanmean(SolZen[ri:ri+DY,ci:ci+DX].ravel())
             mean_RAA[totpiece]=np.nanmean(RAA[ri:ri+DY,ci:ci+DX].ravel())
                 
         else:
             
             #--------- Radiance vs DFNC for vis, nir,swir,thermal
                 vis_dfnc,visDstd,visDN=bindata(vis[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 
                 tmp=vis_dfnc[0,:].copy()
                 lastbin=np.array(np.where(1-np.isnan(tmp)))
                 if (lastbin.size==0) or (np.nansum(np.isnan(tmp[1:6]))>1):
                     # print('no data as DFNC');
                     continue
                 lastbin=lastbin[0,-1]+1
                 if lastbin<7: 
                     # print('DFNC<7km')
                     continue           # restriction!
                 nir_dfnc,nirDstd,nirDN=bindata(nir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 swir_dfnc,swirDstd,swirDN=bindata(swir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 thermal_dfnc,thermalDstd,thermalDN=bindata(thermal[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
             
     
                 noisfacsw=0.05 # When noise factor decreases the algorithm has less tolorance to noise in saturation so saturation point get farther away.
                 noisfaclw=0.1 
                 vissat,vissatdis,vis_fndsat=findsat(vis_dfnc[:,1:],noisfacsw,0) #DFNCbins
                 nirsat,nirsatdis,nir_fndsat=findsat(nir_dfnc[:,1:],noisfacsw,0)
                 swirsat,swirsatdis,swir_fndsat=findsat(swir_dfnc[:,1:],noisfacsw,0)
                 thermalsat,thermalsatdis,thermal_fndsat=findsat(thermal_dfnc[:,1:],noisfaclw,0)
                 
                 
                 # ------ start collecting subdomain (piece) data ----------------
                 totpiece=totpiece+1
                 CloudFraction[totpiece]=CF
                 
                 
                 if channel_adjusting:
                     tmp=np.reshape(np.concatenate((vis_fndsat,nir_fndsat,swir_fndsat),axis=0),(1,len(vis)+len(nir)+len(swir)))
                    # if totpiece==1:
                     #    FNDSATQsw=tmp
                     #    FNDSATQlw=np.reshape(thermal_fndsat,(1,len(thermal)))
                    # else:
                    #     FNDSATQsw=np.concatenate((FNDSATQsw,tmp),axis=0)
                    #     FNDSATQlw=np.concatenate((FNDSATQlw,np.reshape(thermal_fndsat,(1,len(thermal)))),axis=0)

                     BadLWchan=5
                     if (np.nansum(tmp==-1)>4) or (np.nansum(thermal_fndsat==-1)>BadLWchan):
                         continue
                     
                 
                 for i in range(0,np.shape(thermal)[0]):
                     if i<3:
                         tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>nirsatdis[i])
                         nirclr[0,i]=np.nanmean(tmp[tmp2])
                         nirclr[1,i]=np.nanvar(tmp[tmp2])
                     if i<4:
                         tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>vissatdis[i])
                         visclr[0,i]=np.nanmean(tmp[tmp2])
                         visclr[1,i]=np.nanvar(tmp[tmp2])
                         tmp=swir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>swirsatdis[i])
                         swirclr[0,i]=np.nanmean(tmp[tmp2])
                         swirclr[1,i]=np.nanvar(tmp[tmp2])
                     tmp=thermal[i,ri:ri+DY,ci:ci+DX].copy()
                     tmp2=np.where(subDFNC>thermalsatdis[i])
                     thermalclr[0,i]=np.nanmean(tmp[tmp2])
                     thermalclr[1,i]=np.nanvar(tmp[tmp2])
                 clrswspec=np.concatenate((visclr,nirclr,swirclr),axis=1)
         
                     # extentvec=np.concatenate((vissatdis,nirsatdis,swirsatdis,thermalsatdis),axis=0)
    
         # solarls=np.concatenate((visls,nirls,swirls),axis=0)
          
         # specclr=np.concatenate((visclr,nirclr,swirclr,thermalclr),axis=1)
    
         
         # ---------------- solar -----------------------------------  
         tmp2=np.concatenate((vis[:,ri:ri+DY,ci:ci+DX],nir[:,ri:ri+DY,ci:ci+DX],swir[:,ri:ri+DY,ci:ci+DX]),axis=0)
         a= clrswspec[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=clrswspec[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMsw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMsw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
    
         TSMsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # clrstdsum=np.sum(np.sqrt(specclr[1,:]))
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorsw=tmp3/tmp4            # SHORTWAVE TWILIGHT FACTOR (Fnc/Fclr)
         TWfactorsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         swDom_mean_eps[totpiece]=np.nanmean(TWfactorsw)
         
         # TWfac_uncer=(tmp3<=tmp4+clrstdsum) & (tmp3>=tmp4-clrstdsum)
         # TWfac_uncer[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfactor[(TWfac_uncer) & (1-np.isnan(TWfac_uncer))]=1
      # ---------------- thermal -----------------------------------  
         tmp2=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         a=thermalclr[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=thermalclr[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMlw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMlw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
             
         TSMlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
    
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorlw=tmp3/tmp4             # LONGWAVE TWILIGHT FACTOR (Fnc/Fclr)
         TWfactorlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         lwDom_mean_eps[totpiece]=np.nanmean(TWfactorlw)

        # ------------------------- SST corners ----------------- 
         sublon=Lon[ri:ri+DY,ci:ci+DX] # MODIS SUB-AREA LON AND (LAT)
         sublat=Lat[ri:ri+DY,ci:ci+DX]
         
         North=np.max(sublat)  # MODIS SUB-AREA MAX POSITIONS
         South=np.min(sublat)
         East=np.max(sublon)
         West=np.min(sublon)
         
         if (np.isnan(North)) or (np.isnan(South)) or (np.isnan(West)) or (np.isnan(East)):
             continue
         
         if (East>0) & (West<0) & (East-West>150): 
             tmp=East.copy()
             East=West.copy()
             West=tmp.copy()                   
     # -------------- CERES FOVs ----------------------------
         CEREStoday=np.where(CERESjdate==MODISday)[0]
         
         # my_column_names = ['Ffov','meanfovTWf','TSM>1_count']  # DATA FRAMES
         # DFswnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFswclr = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwclr = pd.DataFrame(data=[], columns=my_column_names)
         
         fovisw=0
         fovilw=0
         

         for fov in CEREStoday:		# Looping through CERES FOVS in same space and time as the subdomian image
             
             if (np.isnan(Flw[fov]) & np.isnan(Fsw[fov])) or (Fsw[fov]<1) or (CERESvza[fov]>25): continue  # don't analyze CERES fov if its Nan/nightime(SW=0)/Not NADIR VZA
             
             fovlat=CERESlat[fov] # CERES FOV LAT (AND LON)
             fovlon=CERESlon[fov]
             if ((fovlat<North) & (fovlat>South)) & ((fovlon>West) & (fovlon<East)):
                 
                 Cdiff=np.sqrt((sublat-fovlat)**2+(sublon-fovlon)**2)
                 
                 subsubC=np.nonzero(Cdiff==np.min(Cdiff)) # CERES FOV LAT IN MODIS SUB-AREA COORDINATES
                 if subsubC[0].size==0:
                     continue
                 
                 # Cdiffv[esh]=np.min(Cdiff)
                 # esh=esh+1
                 subsubCy=subsubC[0][0].copy(); subsubCx=subsubC[1][0].copy()
                 
                 # Finding the location of neighbor FOV                    # changed 9/7 S
                 fovlat2=CERESlat[fov+1] 
                 fovlon2=CERESlon[fov+1]
                 Cdiff2=np.sqrt((sublat-fovlat2)**2+(sublon-fovlon2)**2)
                 subsubC2=np.nonzero(Cdiff2==np.min(Cdiff2))
                 if subsubC2[0].size==0:
                     print('No mindiff')
                     continue
                 subsubCx2=subsubC2[1][0]                
                 if PSFf:
                     if subsubCx2<subsubCx:   
                         PSFc=PSFc2.copy()
                         FOVmsk=np.fliplr(PSF)
                         fovorin=False
                     else:
                         PSFc=PSFc1.copy()
                         FOVmsk=PSF.copy()                 # changed 9/7 E
                 
                 # input('CERES part')
                 if PSFf:
                     subsubRi=int(subsubCy-fovsc[0]/2)
                     subsubRf=int(subsubCy+fovsc[0]/2)
                     subsubCi=int(subsubCx-(PSFc-1))
                     subsubCf=int(subsubCx+(fovsc[1]-PSFc))
                 else:
                     subsubRi=int(subsubCy-fovsc/2)
                     subsubRf=int(subsubCy+fovsc/2)
                     subsubCi=int(subsubCx-fovsc/2)
                     subsubCf=int(subsubCx+fovsc/2)
                 
                 if (subsubRi<0) | (subsubRf>DY-1) | (subsubCi<0) | (subsubCf>DX-1): # if FOV is out of the image continue
                     continue
     # ------------------------------- calculating for CERES FOV SW ---------------------------------------- 
                 FswNAN=True
                 if np.isnan(Fsw[fov])==False:
                     FswNAN=False
                     if subsubRf==DY-1:
                        subsubTSMsw=TSMsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWfsw=TWfactorsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubSZA=subSolZen[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubVZA=subSenZen[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubRAA=subRAA[subsubRi:,subsubCi:subsubCf+1].copy()
                        thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                        solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubSZA=subSolZen[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubVZA=subSenZen[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubRAA=subRAA[subsubRi:subsubRf+1,subsubCi:].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubSZA=subSolZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubVZA=subSenZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubRAA=subRAA[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                        
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMsw))
                         #tmpx=tmp[1] X,Y corrdinates of NaN values in FOV
                         #tmpy=tmp[0]
                         nanfrac=np.sum(FOVmsk[tmp])
                         if nanfrac<CERESnanprec:  # if there are Nan in unimportnat regions of the PSF (sum smaller than CERESnanprec), still use the FOV
                             tmp=np.nonzero(subsubTSMsw>1.0)
                             #tmpx=tmp[1]
                             #tmpy=tmp[0]
                             tmp5=np.nansum(subsubTSMsw[tmp]*FOVmsk[tmp])
                             tmp1=np.nansum(subsubTSMsw*FOVmsk)
                             if tmp5/tmp1<TSMgt1_ratio:  # If the contribution of pixels with TSM>1 is negligible to the weighted average TSM of FOV (sum smaller than CERESnanprec), still use the FOV   
                                 sw_meanTSM[totpiece,fovisw]=np.nansum(subsubTSMsw*FOVmsk)
                                 sw_meanRHO[totpiece,fovisw]=np.nansum(subsubTWfsw*FOVmsk)
                                 sw_meanDFNC[totpiece,fovisw]=np.sum(subsubDFNC*FOVmsk)
                                 sw_meanSZA[totpiece,fovisw]=np.nansum(subsubSZA*FOVmsk)
                                 sw_meanVZA[totpiece,fovisw]=np.nansum(subsubVZA*FOVmsk)
                                 sw_meanRAA[totpiece,fovisw]=np.nansum(subsubRAA*FOVmsk)
                                 tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                                 sw_meanSolarspec[totpiece,fovisw,:]=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                                 tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                                 sw_meanThermalspec[totpiece,fovisw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                                 
                                 CERESSZAsw[totpiece,fovisw]=CERESsza[fov]     
                                 CERESRAAsw[totpiece,fovisw]=CERESraa[fov]    
                                 CERESVZAsw[totpiece,fovisw]=CERESvza[fov]
                                 CERESlonsw[totpiece,fovisw]=fovlon
                                 CERESlatsw[totpiece,fovisw]=fovlat
                                 # cotime[totpiece,fovisw]=CERESjdate[fov]-MODISday
                                 # coloc[totpiece,fovisw]=np.min(Cdiff)
                                 
                                 # swCERES_MOD_name[totpiece,fovisw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                                 sw_F[totpiece,fovisw]=Fsw[fov] 
                                 sw_I[totpiece,fovisw]=Isw[fov]
                                 sw_AODfov[totpiece,fovisw]=AODfov[fov]
                                 sw_Clrfov[totpiece,fovisw]=Clrfov[fov]
                                 
                                 # sw_Fclrspec[totpiece,fovisw,:]=specclr[0,:]
                              #  sw_FclrspecSTD[totpiece,fovisw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
                                 
                                 fovisw=fovisw+1
                         
                     else:
                         if np.sum(np.isnan(subsubTSMsw[FOVmsk]))>0:
                             continue
                         subsubTSMsw[(~FOVmsk)]=np.nan 
                         subsubTWfsw[(~FOVmsk)]=np.nan
                         subsubDFNC[(~FOVmsk)]=np.nan
                         sw_TSM1f[totpiece,fovisw]=np.nansum(subsubTSMsw>1)/NopxFOVmsk
                         sw_meanRHO[totpiece,fovisw]=np.nanmean(subsubTWfsw)
                         sw_meanTSM[totpiece,fovisw]=np.nanmean(subsubTSMsw)
                         sw_meanDFNC[totpiece,fovisw]=np.nanmean(subsubDFNC)
                         
                         CERESSZAsw[totpiece,fovisw]=CERESsza[fov]     
                         CERESRAAsw[totpiece,fovisw]=CERESraa[fov]    
                         CERESVZAsw[totpiece,fovisw]=CERESvza[fov]
                         CERESlonsw[totpiece,fovisw]=fovlon
                         CERESlatsw[totpiece,fovisw]=fovlat
                         # cotime[totpiece,fovisw]=CERESjdate[fov]-MODISday
                         # coloc[totpiece,fovisw]=np.min(Cdiff)
                         
                         # swCERES_MOD_name[totpiece,fovisw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                         sw_F[totpiece,fovisw]=Fsw[fov] 
                         sw_I[totpiece,fovisw]=Isw[fov]
                         sw_AODfov[totpiece,fovisw]=AODfov[fov]
                         sw_Clrfov[totpiece,fovisw]=Clrfov[fov]
                         
                         # sw_Fclrspec[totpiece,fovisw,:]=specclr[0,:]
                      #  sw_FclrspecSTD[totpiece,fovisw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
                        
                         fovisw=fovisw+1
                     
    
    # --------------------------------- Analyzing LW fov---------------------------------------------                 
                 if np.isnan(Flw[fov])==False:
                        
                     if subsubRf==DY-1:
                        subsubTSMlw=TSMlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWflw=TWfactorlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        if FswNAN:
                            subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                            subsubSZA=subSolZen[subsubRi:,subsubCi:subsubCf+1].copy()
                            subsubVZA=subSenZen[subsubRi:,subsubCi:subsubCf+1].copy()
                            subsubRAA=subRAA[subsubRi:,subsubCi:subsubCf+1].copy()
                            thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                            solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:].copy() 
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                             subsubSZA=subSolZen[subsubRi:subsubRf+1,subsubCi:].copy()
                             subsubVZA=subSenZen[subsubRi:subsubRf+1,subsubCi:].copy()
                             subsubRAA=subRAA[subsubRi:subsubRf+1,subsubCi:].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             subsubSZA=subSolZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             subsubVZA=subSenZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             subsubRAA=subRAA[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMlw))
                         #tmpx=tmp[1]
                         #tmpy=tmp[0]
                         nanfrac=np.sum(FOVmsk[tmp])
                         if nanfrac<CERESnanprec:  # if there are Nan in unimportnat regions of the PSF (sum smaller than CERESnanprec), still use the FOV                            
                            tmp=np.nonzero(subsubTSMlw>1.0)
                            #tmpx=tmp[1]
                            #tmpy=tmp[0]
                            tmp5=np.nansum(subsubTSMlw[tmp]*FOVmsk[tmp])
                            tmp1=np.nansum(subsubTSMlw*FOVmsk)
                            if tmp5/tmp1<TSMgt1_ratio:  # If the contribution of pixels with TSM>1 is negligible to the weighted average TSM of FOV (sum smaller than CERESnanprec), still use the FOV
                                # print('FOV lw TSM: ', tmp5/tmp1)
                            
                                lw_meanTSM[totpiece,fovilw]=np.nansum(subsubTSMlw*FOVmsk)
                                lw_meanRHO[totpiece,fovilw]=np.nansum(subsubTWflw*FOVmsk)
                                lw_meanDFNC[totpiece,fovilw]=np.sum(subsubDFNC*FOVmsk) 
                                lw_meanSZA[totpiece,fovilw]=np.nansum(subsubSZA*FOVmsk)
                                lw_meanVZA[totpiece,fovilw]=np.nansum(subsubVZA*FOVmsk)
                                lw_meanRAA[totpiece,fovilw]=np.nansum(subsubRAA*FOVmsk)
                                
                                tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                                lw_meanSolarspec[totpiece,fovilw,:]=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                                tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                                lw_meanThermalspec[totpiece,fovilw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                                
                                CERESSZAlw[totpiece,fovilw]=CERESsza[fov] 
                                CERESRAAlw[totpiece,fovilw]=CERESraa[fov]    
                                CERESVZAlw[totpiece,fovilw]=CERESvza[fov]
                                CERESlonlw[totpiece,fovilw]=fovlon
                                CERESlatlw[totpiece,fovilw]=fovlat
                                # lwCERES_MOD_name[totpiece,fovilw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                                lw_F[totpiece,fovilw]=Flw[fov]  
                                lw_I[totpiece,fovilw]=Ilw[fov] 
               
                                fovilw=fovilw+1 
                        
                     else:
                         if np.sum(np.isnan(subsubTSMlw[FOVmsk]))>0:
                             continue                              
                         subsubTSMlw[(~FOVmsk)]=np.nan 
                         subsubTWflw[(~FOVmsk)]=np.nan 
                         lw_meanRHO[totpiece,fovilw]=np.nanmean(subsubTWflw)
                         lw_meanTSM[totpiece,fovilw]=np.nanmean(subsubTSMlw)
                         if FswNAN:
                             subsubDFNC[(~FOVmsk)]=np.nan
                         lw_meanDFNC[totpiece,fovilw]=np.nanmean(subsubDFNC[FOVmsk])
                         TSM1sum=np.nansum(subsubTSMlw>1)
                         lw_TSM1f[totpiece,fovilw]=TSM1sum/ NopxFOVmsk
                         
                         CERESSZAlw[totpiece,fovilw]=CERESsza[fov]
                         CERESRAAlw[totpiece,fovilw]=CERESraa[fov]    
                         CERESVZAlw[totpiece,fovilw]=CERESvza[fov]
                         CERESlonlw[totpiece,fovilw]=fovlon
                         CERESlatlw[totpiece,fovilw]=fovlat
                         # lwCERES_MOD_name[totpiece,fovilw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                         lw_F[totpiece,fovilw]=Flw[fov]  
                         lw_I[totpiece,fovilw]=Ilw[fov] 
                                                       
                         # lw_Fclrspec[totpiece,fovilw,:]=specclr[0,:]
                      #  lw_FclrspecSTD[totpiece,fovilw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
        
                         fovilw=fovilw+1    
                        

                   
         foviswmax=np.maximum(foviswmax,fovisw)
         fovilwmax=np.maximum(fovilwmax,fovilw)

     
     swallnanraw=np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1])[0]                  
     lwallnanraw=np.nonzero(np.nansum(np.isnan(lw_F),axis=1)<np.shape(lw_F)[1])[0] 
     # print('******',np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1]))
     # if np.nansum(swallnanraw==lwallnanraw)!=len(swallnanraw):
     #     print('DOMAIN DONT OVERLAP IN SW AND LW')
     #     input()
     # print('N Fsw FOVs- ',np.sum(~np.isnan(sw_F[swallnanraw,:]),axis=1))
     # print(np.nanmean(sw_F),print(np.sum(~np.isnan(sw_F[swallnanraw,:]))))
     # print(np.nanmean(lw_F),print(np.sum(~np.isnan(lw_F[lwallnanraw,:]))))
     
     swallnanspec=np.nonzero(np.nansum(np.nansum(np.isnan(sw_meanSolarspec),axis=2),axis=1)<(np.shape(sw_meanSolarspec)[1]*np.shape(sw_meanSolarspec)[2]))[0] 
     if np.sum(swallnanraw-swallnanspec)!=0:
         input()


     if np.size(swallnanraw)+np.size(lwallnanraw)==0:
         return('nothing_found')
     else:
         SWdic = {"F": sw_F[swallnanraw,:], "I": sw_I[swallnanraw,:], "mRHO":sw_meanRHO[swallnanraw,:],
                  "mDFNC": sw_meanDFNC[swallnanraw,:], "mTSM": sw_meanTSM[swallnanraw,:], 
                  "VZA":CERESVZAsw[swallnanraw,:], "SZA":CERESSZAsw[swallnanraw,:], "RAA":CERESRAAsw[swallnanraw,:], "Lat":CERESlatsw[swallnanraw,:], "Lon":CERESlonsw[swallnanraw,:],
                  "modSZA": sw_meanSZA[swallnanraw,:],"modVZA": sw_meanVZA[swallnanraw,:],"modRAA": sw_meanRAA[swallnanraw,:], "SZApiece": mean_SZA[swallnanraw], "RAApiece": mean_RAA[swallnanraw],
                  "AOD": sw_AODfov[swallnanraw,:], "Clrf": sw_Clrfov[swallnanraw,:],"SolarCLRspec": sw_meanSolarspec[swallnanraw,:,:],"ThermalCLRspec": sw_meanThermalspec[swallnanraw,:,:],"CF": CloudFraction[swallnanraw], "swDomEPS": swDom_mean_eps[swallnanraw] } 
          
         LWdic = {"F": lw_F[lwallnanraw,:], "I": lw_I[lwallnanraw,:], "mRHO": lw_meanRHO[lwallnanraw,:],
                  "mDFNC": lw_meanDFNC[lwallnanraw,:], "mTSM": lw_meanTSM[lwallnanraw,:], "Lat":CERESlatlw[lwallnanraw,:], "Lon":CERESlonlw[lwallnanraw,:], 
                  "modVZA": lw_meanVZA[lwallnanraw,:],"modRAA": lw_meanRAA[lwallnanraw,:], "modSZA": lw_meanSZA[lwallnanraw,:], "SZA":CERESSZAlw[lwallnanraw,:],
                  "VZA":CERESVZAlw[lwallnanraw,:], "RAA":CERESRAAlw[lwallnanraw,:],"SolarCLRspec": lw_meanSolarspec[lwallnanraw,:,:],"ThermalCLRspec": lw_meanThermalspec[lwallnanraw,:,:],"CF": CloudFraction[lwallnanraw], "lwDomEPS": lwDom_mean_eps[lwallnanraw]  }
         
         # Pieceprop = { "CF": CloudFraction[swallnanraw], "SZApiece": mean_SZA[swallnanraw], "RAApiece": mean_RAA[swallnanraw], "swDomEPS": swDom_mean_eps[swallnanraw], "lwDomEPS": lwDom_mean_eps[swallnanraw]  }
         
         return(SWdic,LWdic) #,Pieceprop) 
    
#%%
def MOD_CERES_allsky(path,MOD02,MOD03,MOD06,MOD35,ceres_data,cldmsktyp,FOVmsk_t,spath,GenIm):
    
     print('started MOD_CERES_allsky') 
     starttime=time.time()
     
     CERESnanprec=0.001 # precent of CERES fov (energy wise) to allow for nan values and use this FOV
     # TSMgt1_ratio=1  # TSM>1 fraction thershold for defining a pure clear sky FOV. Tollarence to how much contribution of TSM>1 to mean TSM
     # print('CERESnanprec delos', CERESnanprec,TSMgt1_ratio)
     
     # print(MOD02)
     CERESjdate=ceres_data[0,:]
     CERESlat=ceres_data[1,:]
     CERESlon=ceres_data[2,:]
     CERESsza=ceres_data[3,:]
     CERESvza=ceres_data[4,:]
     CERESraa=ceres_data[5,:]
     Fsw=ceres_data[6,:]
     Isw=ceres_data[7,:]
     Flw=ceres_data[8,:]
     Ilw=ceres_data[9,:]
     AODfov=ceres_data[10,:]
     Clrfov=ceres_data[11,:]
     
     # if np.shape(FOVmsk)[0]%2==0:
     PSF=FOVmsk_t.copy()
     tmp=np.shape(PSF)[0]-1
     # if np.shape(FOVmsk)[1]%2!=0:
     tmp1=np.shape(PSF)[1]
     fovsc=[tmp,tmp1]
     tmp=np.sum(PSF,axis=0)
     PSFc1=np.nonzero(tmp==np.max(tmp))[0][0] 
     PSFc2=tmp1-PSFc1-1

     PSFsz=np.shape(PSF)        
     NopxFOVmsk=np.size(PSF)    # number of modis pixels in CERES FOV


     solarls=np.array([0.47,0.55,0.66,0.85,0.905,0.935,0.940,1.24,1.64,2.13,1.375])
     thermalls=np.array([3.75,3.96,3.96,4.05,4.46,4.5,6.7,7.32,8.55,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
     
     estimated_fovs=10000  #Npices
     Nscalars=20
     Nwavlngs=20
     
     # FOV_scalar=np.ones((estimated_fovs,PSFsz[0],PSFsz[1],Nwavlngs+3),dtype=np.float32)*np.nan
     # FOV_im=np.ones((estimated_fovs,Nscalars),dtype=np.float32)*np.nan
     
     
     swF=np.ones((estimated_fovs),dtype=np.float32)*np.nan     
     # # swCERES_MOD_name=['nan']*Nsamples
     meanSumIsw=swF.copy()
     stdSumIsw=swF.copy()
     swI=swF.copy()
     AOD=swF.copy()
     ClrSkyf=swF.copy()
     
     CERESSZA=swF.copy()   
     CERESRAA=swF.copy()    
     CERESVZA=swF.copy() 
     CERESLON=swF.copy() 
     CERESLAT=swF.copy() 
     FOVorientation=swF.copy() 
     
     lwF=swF.copy()
     meanSumIlw=swF.copy()
     stdSumIlw=swF.copy()
     lwI=swF.copy()
     
     meanDFNC=swF.copy() 
     stdDFNC=swF.copy() 
     CF=swF.copy()
     glintf=swF.copy()
     landf=swF.copy()
     LiqPhsf=swF.copy()
     meanCTH=swF.copy()
     stdCTH=swF.copy()


     dtNtm=MOD02[9:22] 
     MODISday=float(dtNtm[1:11])
     print(dtNtm)
# ------------------------ Reading MODIS L2 images ------------------------------------
     temp="EV_500_Aggr1km_RefSB" #"EV_1KM_RefSB"
     sclsNoffs="radiance" # "radiance"
     Rsw500=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_250_Aggr1km_RefSB" #"EV_1KM_RefSB"
     Rsw250=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_1KM_RefSB" #"EV_1KM_RefSB"
     Rswkm=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     vis=np.append(Rsw500[0:2],Rsw250,axis=0)
     nir=Rswkm[-4:-1]
     tmp=np.expand_dims(Rswkm[-1], axis=0).copy()
     swir=np.append(Rsw500[-3:],tmp,axis=0)
     
     temp="EV_1KM_Emissive" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     thermal=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     # thermalfull= Planckfunc(thermalls*1e-6,thermal,True) #  thermal #  
     # thermal=thermalfull
     
     solar=np.concatenate((vis,nir,swir),axis=0)
     del Rswkm,Rsw250,Rsw500, vis,nir,swir
     Granulrsz=[np.shape(solar)[1],np.shape(solar)[2]] 
     Nsws=np.shape(solar)[0]
     Nlws=np.shape(thermal)[0]
     
     meanSolarspec=np.ones((estimated_fovs,Nsws),dtype=np.float32)*np.nan 
     stdSolarspec=meanSolarspec.copy()
     skewSolarspec=meanSolarspec.copy()
     kurtSolarspec=meanSolarspec.copy()
     meanThermalspec=np.ones((estimated_fovs,Nlws),dtype=np.float32)*np.nan 
     stdThermalspec=meanThermalspec.copy()
     skewThermalspec=meanThermalspec.copy()
     kurtThermalspec=meanThermalspec.copy()
     
     if GenIm:
         Imsw=np.ones((estimated_fovs,Nsws,PSFsz[0],PSFsz[1]),dtype=np.float32)*np.nan
         Imlw=np.ones((estimated_fovs,Nlws,PSFsz[0],PSFsz[1]),dtype=np.float32)*np.nan
         Imview=np.ones((estimated_fovs,3,PSFsz[0],PSFsz[1]),dtype=np.float32)*np.nan
 #---------------------------------------Reading cloud mask ------------------------------
     temp = nc.Dataset(path+'/MOD35/'+MOD35, 'r')
     MOD35D = temp.variables["Cloud_Mask"]
     
     # fillV=temp.variables[MOD35D].getncattr('_FillValue')
     # CldMsk[CldMsk==fillV]=np.nan
     
     MODsz=np.shape(thermal)
     MODsz=MODsz[1:3]
     CldMsk=np.zeros(MODsz)
     glint=np.zeros(MODsz)
     sfcice=np.zeros(MODsz)
     land=np.zeros(MODsz)
     thinCirr=np.zeros(MODsz)
     NonCld=np.zeros(MODsz)
     byte1=np.array(MOD35D[0,:,:])
     byte1[byte1<0]=256+byte1[byte1<0]
     byte2=np.array(MOD35D[1,:,:])
     byte2[byte2<0]=256+byte2[byte2<0]  # correct for negative values
     for i in range(0,MODsz[0]):
         for j in range(0,MODsz[1]):  
             binary=format(int(byte1[i,j]),'#010b').replace("0b","")
             binary=binary[::-1]
             binary2=format(int(byte2[i,j]),'#010b').replace("0b","")
             binary2=binary[::-1]
             if int(binary[0])==1: # & int(binary[-4])==dayf:  # if mask is determined and is day/night
                 CldMsk[i,j]=int(binary[1:3],2)  
                 glint[i,j]=int(binary[4],2)
                 sfcice[i,j]=int(binary[5],2)
                 land[i,j]=int(binary[6:],2)
                 bit10=int(binary2[1])
                 # bit12=int(binary2[3])
                 thinCirr[i,j]=bit10
                 NonCld[i,j]=int(binary2[0])
     
     BWmsk=np.copy(CldMsk)
     if cldmsktyp=='relaxed':  # take only 95% confident as cloud          
        BWmsk[BWmsk>0]=1
        BWmsk=BWmsk.astype(bool)
        BWmsk=np.invert(BWmsk)    
     if cldmsktyp=='strict':    #cloud even if only 33% confident      
        BWmsk[BWmsk==0]=1
        BWmsk[BWmsk>1]=0
        BWmsk=BWmsk.astype(bool)
        
     glint=1-glint
     glint=glint.astype(bool)
         
     # --------------------- Reading clouds --------------------------------
     var="Cloud_Phase_Infrared_1km" 
     dataset = nc.Dataset(path+'MOD06/'+MOD06, 'r')
     mtrx = dataset.variables[var]
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     # mtrx=np.ma.masked_array(mtrx, mtrx==fillV) # mtrx[mtrx==fillV]=np.nan
     mtrx=np.where(mtrx==fillV,np.nan,mtrx) # mtrx[mtrx==fillV]=np.nan
     # scalesn='scale_factor'
     # offsn='add_offset'
     # scales=dataset.variables[var].getncattr(scalesn)
     # offset=dataset.variables[var].getncattr(offsn)
     CldPhs=mtrx.copy() #(mtrx-offset)*scales
     
     
     # CldPhs=scipy.ndimage.zoom(CldPhs, 5, order=0)
     
     
     var="cloud_top_height_1km"
     mtrx = dataset.variables[var]
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
     CldTop=mtrx.copy()
     
     # ------------------ GEOLOCATION ---------------------------------------------------------    
     temp="Latitude" #"EV_1KM_RefSB"
     sclsNoffs=0
     Lat=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     temp="Longitude" #"EV_1KM_RefSB"
     Lon=readMODIS(path+'/MOD03/'+MOD03,temp,sclsNoffs)
     
     temp="SensorZenith" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarZenith" #"EV_1KM_RefSB"
     SolZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SolZen=SenZen*0.01
     
     
     temp="SensorAzimuth" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarAzimuth" #"EV_1KM_RefSB"
     SolAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     
     RAA=SolAz-SenAz-180
     RAA=np.where(RAA<0,RAA+360,RAA)
     RAA=np.abs(RAA-180)
     
     del SolAz, SenAz    
    
     
     # #################################### Calc distances ##################################################        
     DFNC=scipy.ndimage.distance_transform_edt(1-BWmsk)
     xx,yy=np.meshgrid(np.array(range(1,MODsz[1]+1,1)),np.array(range(1,MODsz[0]+1,1)))
     xx=np.minimum(xx,np.fliplr(xx))
     yy=np.minimum(yy,np.flipud(yy))
     tmp=np.minimum(yy,xx)   # smallest distance to image edge
     DFNC[DFNC>tmp]=np.nan
     

     # tmp=np.copy(CldPhs)
     # tmp[tmp==1]=0
     # tmp[tmp==6]=0
     # tmp[(tmp==2) | (tmp==3)]=1
     # tmp=tmp.astype(bool)
     # DFIC=scipy.ndimage.distance_transform_edt(1-tmp)
     
     # tmp=np.copy(CldPhs)
     # tmp[tmp<6]=0
     # tmp[tmp==6]=1
     # tmp=tmp.astype(bool)
     # DFuncertPhs=scipy.ndimage.distance_transform_edt(1-tmp)
     
     
     unwanted=np.zeros((MODsz[0],MODsz[1]), dtype=bool)
     unwanted[np.nonzero(np.isnan(SenZen))]=True
     unwanted[np.nonzero(np.isnan(SolZen))]=True
     unwanted[np.nonzero(np.isnan(RAA))]=True
     unwanted[np.nonzero(np.sum(np.isnan(solar),axis=0)>0)]=True
     unwanted[np.nonzero(np.sum(np.isnan(thermal),axis=0)>0)]=True
            
         
     North=np.max(Lat)  # MODIS SUB-AREA MAX POSITIONS
     South=np.min(Lat)
     East=np.max(Lon)
     West=np.min(Lon)
        
     if (np.isnan(North)) or (np.isnan(South)) or (np.isnan(West)) or (np.isnan(East)):
           print('!!!!!!!!!!!!!!!!!!! norh/south/west/east is NaN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
           return ('NaN in corners')
        
     if (East>0) & (West<0) & (East-West>150): 
            tmp=East.copy()
            East=West.copy()
            West=tmp.copy()                   
    # -------------- CERES FOVs ----------------------------
     CEREStoday=np.where(CERESjdate==MODISday)[0]
        
     fovisw=0
     fovilw=0
         

     for fov in CEREStoday:		# Looping through CERES FOVS in same space and time as the subdomian image
             
             if (np.isnan(Flw[fov]) & np.isnan(Fsw[fov])) or (Fsw[fov]<1): continue  # don't analyze CERES fov if its Nan/nightime(SW=0)
             
             fovlat=CERESlat[fov] # CERES FOV LAT (AND LON)
             fovlon=CERESlon[fov]
             if ((fovlat<North) & (fovlat>South)) & ((fovlon>West) & (fovlon<East)):
                 
                 Cdiff=np.sqrt((Lat-fovlat)**2+(Lon-fovlon)**2)
                 
                 subsubC=np.nonzero(Cdiff==np.min(Cdiff)) # CERES FOV LAT IN MODIS SUB-AREA COORDINATES
                 if subsubC[0].size==0:
                     continue
                 
                 # Cdiffv[esh]=np.min(Cdiff)
                 # esh=esh+1
                 subsubCy=subsubC[0][0].copy(); subsubCx=subsubC[1][0].copy()
                 
                 # Finding the location of neighbor FOV                    # changed 9/7 S
                 fovlat2=CERESlat[fov+1] 
                 fovlon2=CERESlon[fov+1]
                 Cdiff2=np.sqrt((Lat-fovlat2)**2+(Lon-fovlon2)**2)
                 subsubC2=np.nonzero(Cdiff2==np.min(Cdiff2))
                 if subsubC2[0].size==0:
                     print('No mindiff')
                     continue
                 subsubCx2=subsubC2[1][0]                

                 if subsubCx2<subsubCx:   
                     PSFc=PSFc2.copy()
                     FOVmsk=np.fliplr(PSF)
                     fovorin=False
                 else:
                     PSFc=PSFc1.copy()
                     FOVmsk=PSF.copy()
                     fovorin=True
                 
                 # input('CERES part')

                 subsubRi=int(subsubCy-fovsc[0]/2)
                 subsubRf=int(subsubCy+fovsc[0]/2)
                 subsubCi=int(subsubCx-(PSFc-1))
                 subsubCf=int(subsubCx+(fovsc[1]-PSFc))

                 
                 if (subsubRi<0) | (subsubRf>Granulrsz[0]-1) | (subsubCi<0) | (subsubCf>Granulrsz[1]-1): # if FOV is out of the image continue
                     continue
     # ------------------------------- calculating for CERES FOV SW ---------------------------------------- 
                 if (np.isnan(Fsw[fov]) & np.isnan(Flw[fov]))==True:
                     print('NaN Fsw & Flw')
                     continue
                 
                 if subsubRf==Granulrsz[0]-1:
                     unwantfov=unwanted[subsubRi:,subsubCi:subsubCf+1]
                 elif subsubCf==Granulrsz[1]-1:
                     unwantfov=unwanted[subsubRi:subsubRf+1,subsubCi:]
                 else:    
                     unwantfov=unwanted[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                 
                 tmp=np.nonzero(unwantfov)
                 nanfrac=np.sum(FOVmsk[tmp])
                 if nanfrac>CERESnanprec:  # if there are Nan in unimportnat regions of the PSF (sum smaller than CERESnanprec), still use the FOV   
                    print('NaN frac>threshold')
                    continue
                 if subsubRf==Granulrsz[0]-1:
                        subsubSZA=SolZen[subsubRi:,subsubCi:subsubCf+1]
                        subsubVZA=SenZen[subsubRi:,subsubCi:subsubCf+1]
                        subsubRAA=RAA[subsubRi:,subsubCi:subsubCf+1]
                        thermalfov=thermal[:,subsubRi:,subsubCi:subsubCf+1]
                        solarfov=solar[:,subsubRi:,subsubCi:subsubCf+1]
                        subsubDFNC=DFNC[subsubRi:,subsubCi:subsubCf+1]
                        cthfov=CldTop[subsubRi:,subsubCi:subsubCf+1]
                        cphsfov=CldPhs[subsubRi:,subsubCi:subsubCf+1]
                        glintfov=glint[subsubRi:,subsubCi:subsubCf+1]
                        landfov=land[subsubRi:,subsubCi:subsubCf+1]
                        cldmskfov=BWmsk[:subsubRi:,subsubCi:subsubCf+1]
                 elif subsubCf==Granulrsz[1]-1:
                         subsubSZA=SolZen[subsubRi:subsubRf+1,subsubCi:]
                         subsubVZA=SenZen[subsubRi:subsubRf+1,subsubCi:]
                         subsubRAA=RAA[subsubRi:subsubRf+1,subsubCi:]
                         thermalfov=thermal[:,subsubRi:subsubRf+1,subsubCi:]
                         solarfov=solar[:,subsubRi:subsubRf+1,subsubCi:]
                         subsubDFNC=DFNC[subsubRi:subsubRf+1,subsubCi:]
                         cthfov=CldTop[subsubRi:subsubRf+1,subsubCi:]
                         cphsfov=CldPhs[subsubRi:subsubRf+1,subsubCi:]
                         glintfov=glint[subsubRi:subsubRf+1,subsubCi:]
                         landfov=land[subsubRi:subsubRf+1,subsubCi:]
                         cldmskfov=BWmsk[subsubRi:subsubRf+1,subsubCi:]
                 else:    
                         subsubSZA=SolZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         subsubVZA=SenZen[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         subsubRAA=RAA[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         thermalfov=thermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         solarfov=solar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         subsubDFNC=DFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         cthfov=CldTop[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         cphsfov=CldPhs[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         glintfov=glint[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         landfov=land[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                         cldmskfov=BWmsk[subsubRi:subsubRf+1,subsubCi:subsubCf+1]
                        
                 tmp=np.where(subsubDFNC<0.1,np.nan,subsubDFNC)
                 tmp2=np.nansum(np.where(subsubDFNC<0.1,np.nan,FOVmsk))
                 meanDFNC[fovisw]=np.nansum(tmp*FOVmsk/tmp2)
                 stdDFNC[fovisw]=np.sqrt(np.nansum(FOVmsk/tmp2*(tmp-meanDFNC[fovisw])**2)) # supposbly could divide by (n-1)/n where n in non-zero weights but this goes to 1: (np.size(FOVmsk)-1)/np.size(FOVmsk)
                 tmp=np.where(cthfov<0.1,np.nan,cthfov)
                 tmp2=np.nansum(np.where(cthfov<0.1,np.nan,FOVmsk))
                 meanCTH[fovisw]=np.nansum(tmp*FOVmsk/tmp2)            
                 stdCTH[fovisw]=np.sqrt(np.nansum(FOVmsk/tmp2*(tmp-meanCTH[fovisw])**2))
                 
                 CERESSZA[fovisw]=CERESsza[fov]     
                 CERESRAA[fovisw]=CERESraa[fov]    
                 CERESVZA[fovisw]=CERESvza[fov]
                 CERESLON[fovisw]=fovlon
                 CERESLAT[fovisw]=fovlat
                 
                 FOVorientation[fovisw]=fovorin
                 
                 AOD[fovisw]=AODfov[fov]
                 ClrSkyf[fovisw]=Clrfov[fov]
                 
                 CF[fovisw]=np.nansum(cldmskfov)/np.nansum(np.invert(unwantfov))
                 LiqPhsf[fovisw]=np.nansum(CldPhs==1)/np.nansum(cldmskfov)#np.nansum(CldPhs>0)
                 glintf[fovisw]=np.nansum(glintfov)/np.nansum(np.invert(unwantfov))
                 landf[fovisw]=np.nansum(landfov)/np.nansum(np.invert(unwantfov))
                # swCERES_MOD_name[totpiece,fovisw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                 if np.isnan(Fsw[fov])==False:
                     swF[fovisw]=Fsw[fov] 
                     swI[fovisw]=Isw[fov]
                     tmp=np.sum(solarfov,axis=0)
                     meanSumIsw[fovisw]=np.nansum(tmp*FOVmsk)
                     stdSumIsw[fovisw]=np.sqrt(np.nansum(FOVmsk*(tmp-meanSumIsw[fovisw])**2))
                 if np.isnan(Flw[fov])==False:
                     lwF[fovisw]=Flw[fov] 
                     lwI[fovisw]=Ilw[fov]
                     tmp=np.sum(thermalfov,axis=0)
                     meanSumIlw[fovisw]=np.nansum(tmp*FOVmsk)
                     stdSumIlw[fovisw]=np.sqrt(np.nansum(FOVmsk*(tmp-meanSumIlw[fovisw])**2))
                 # tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(solarfov)[0],1,1))
                 for i in range(0,np.shape(solar)[0]):
                     meanSolarspec[fovisw,i]=np.nansum(np.squeeze(solarfov[i,:,:])*FOVmsk)
                     stdSolarspec[fovisw,i]=np.sqrt(np.nansum(FOVmsk*(np.squeeze(solarfov[i,:,:])-meanSolarspec[fovisw,i])**2)) # supposbly could divide by (n-1)/n where n in non-zero weights but this goes to 1: (np.size(FOVmsk)-1)/np.size(FOVmsk)
                     skewSolarspec[fovisw,i]=np.nansum(FOVmsk*((np.squeeze(solarfov[i,:,:])-meanSolarspec[fovisw,i])/stdSolarspec[fovisw,i])**3)
                     kurtSolarspec[fovisw,i]=np.nansum(FOVmsk*((np.squeeze(solarfov[i,:,:])-meanSolarspec[fovisw,i])/stdSolarspec[fovisw,i])**4)-3
                     if GenIm:
                         Imsw[fovisw,:,:,:]=solarfov #np.reshape(solarfov,(1,PSFsz[0],PSFsz[1],Nsws))
                 # tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(thermalfov)[0],1,1))
                 for i in range(0,np.shape(thermal)[0]):
                     meanThermalspec[fovisw,i]=np.nansum(np.squeeze(thermalfov[i,:,:])*FOVmsk)
                     stdThermalspec[fovisw,i]=np.sqrt(np.nansum(FOVmsk*(np.squeeze(thermalfov[i,:,:])-meanThermalspec[fovisw,i])**2))
                     skewThermalspec[fovisw,i]=np.nansum(FOVmsk*((np.squeeze(thermalfov[i,:,:])-meanThermalspec[fovisw,i])/stdThermalspec[fovisw,i])**3)
                     kurtThermalspec[fovisw,i]=np.nansum(FOVmsk*((np.squeeze(thermalfov[i,:,:])-meanThermalspec[fovisw,i])/stdThermalspec[fovisw,i])**4)-3
                     if GenIm:
                         Imlw[fovisw,:,:,:]=thermalfov
                 if GenIm:
                     Imview[fovisw,0,:,:]=subsubSZA
                     Imview[fovisw,1,:,:]=subsubVZA
                     Imview[fovisw,2,:,:]=subsubRAA                        
                                 
                 fovisw=fovisw+1                   

     
     swallnanraw=~np.nonzero(np.isnan(swF))[0] 
     print('Fsw',len(swF[swallnanraw]),np.nanmean(swF)) 
     tttt=(time.time()-starttime)/60
     print('Finished calc elapsed time ' + str(tttt))              
     # print('******',np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1]))
     # if np.nansum(swallnanraw==lwallnanraw)!=len(swallnanraw):
     #     print('DOMAIN DONT OVERLAP IN SW AND LW')
     #     input()
     # print('N Fsw FOVs- ',np.sum(~np.isnan(sw_F[swallnanraw,:]),axis=1))
     # print(np.nanmean(sw_F),print(np.sum(~np.isnan(sw_F[swallnanraw,:]))))
     # print(np.nanmean(lw_F),print(np.sum(~np.isnan(lw_F[lwallnanraw,:]))))
     


     if np.size(swallnanraw)==0:
         return('nothing_found')
     else:
         # Scalardic = {"Fsw": swF[swallnanraw], "Isw": swI[swallnanraw], "Flw": lwF[swallnanraw], "Ilw": lwI[swallnanraw],
         #          "mSumIsw": meanSumIsw[swallnanraw], "stdSumIsw": stdSumIsw[swallnanraw], "mSumIlw": meanSumIlw[swallnanraw], "stdSumIlw": stdSumIlw[swallnanraw],
         #          "mDFNC": meanDFNC[swallnanraw], "stdDFNC": stdDFNC[swallnanraw], "CF": CF[swallnanraw], "meanCTH": meanCTH[swallnanraw], "stdCTH": stdCTH[swallnanraw],"LiqPhsf": LiqPhsf[swallnanraw],
         #          "VZA":CERESVZA[swallnanraw], "SZA":CERESSZA[swallnanraw], "RAA":CERESRAA[swallnanraw], "Lat":CERESLAT[swallnanraw], "Lon":CERESLON[swallnanraw],                 
         #          "AOD": AOD[swallnanraw], "Clrf": ClrSkyf[swallnanraw], "glintf": glintf[swallnanraw], "landf": landf[swallnanraw]}                   
         # Immoms = {"mSolarspec": meanSolarspec[swallnanraw,:],"mThermalspec": meanThermalspec[swallnanraw,:], "stdSolarspec": stdSolarspec[swallnanraw,:],"stdThermalspec": stdThermalspec[swallnanraw,:],
         #          "skewSolarspec": skewSolarspec[swallnanraw,:],"skewThermalspec": skewThermalspec[swallnanraw,:], "kurtSolarspec": kurtSolarspec[swallnanraw,:],"kurtThermalspec": kurtThermalspec[swallnanraw,:],
         #          "SWwavlngs": solarls,"LWwavlngs": thermalls} 
         # Imdic = {"Imviews": Imview[swallnanraw,:,:,:],"ImSW": Imsw[swallnanraw,:,:,:],"ImLW": Imlw[swallnanraw,:,:,:],"FOVorientation": FOVorientation[swallnanraw]} 
         
         # np.savez(spath+r'/Allsky_'+dtNtm+'.npz',Scalars=Scalardic,SpecMoments=Immoms,Images=Imdic)
         
         np.savez(spath+r'/scalars/'+dtNtm+'.npz',Fsw=swF[swallnanraw], Isw=swI[swallnanraw], Flw=lwF[swallnanraw], Ilw=lwI[swallnanraw],
                  mSumIsw= meanSumIsw[swallnanraw], stdSumIsw= stdSumIsw[swallnanraw], mSumIlw= meanSumIlw[swallnanraw], stdSumIlw= stdSumIlw[swallnanraw],
                  mDFNC= meanDFNC[swallnanraw], stdDFNC= stdDFNC[swallnanraw], CF= CF[swallnanraw], meanCTH= meanCTH[swallnanraw], stdCTH= stdCTH[swallnanraw],LiqPhsf= LiqPhsf[swallnanraw],
                  VZA=CERESVZA[swallnanraw], SZA=CERESSZA[swallnanraw], RAA=CERESRAA[swallnanraw], Lat=CERESLAT[swallnanraw], Lon=CERESLON[swallnanraw],                 
                  AOD= AOD[swallnanraw], Clrf= ClrSkyf[swallnanraw], glintf= glintf[swallnanraw], landf= landf[swallnanraw])
         np.savez(spath+r'/moments/'+dtNtm+'.npz', mSWspec= meanSolarspec[swallnanraw,:],mLWspec= meanThermalspec[swallnanraw,:], stdSWspec= stdSolarspec[swallnanraw,:],stdLWspec= stdThermalspec[swallnanraw,:],
                  skewSWspec= skewSolarspec[swallnanraw,:],skewLWspec= skewThermalspec[swallnanraw,:], kurtSWspec= kurtSolarspec[swallnanraw,:],kurtLWspec= kurtThermalspec[swallnanraw,:],
                  SWwavlngs= solarls,LWwavlngs= thermalls)
         if GenIm:
             np.savez(spath+r'/images/'+dtNtm+'.npz',Imviews= Imview[swallnanraw,:,:,:],ImSW= Imsw[swallnanraw,:,:,:],ImLW= Imlw[swallnanraw,:,:,:], FOVorientation=FOVorientation[swallnanraw])
         print('Finished saving elapsed time ' + str(time.time()/60-tttt))
         return()#Scalardic,Immoms,Imdic)
     
        
#%%
#!!! start org
def MOD_CERES_org(path,MOD02,MOD03,MOD06,MOD35,ceres_data,cldmsktyp,NoColdCld,BroadLW,MaxDFNC,channel_adjusting,weightTSM,Npices,thermal_channels,FOVmsk_t,PSFf,TSMth,consider_thinCirr):
     
     # PSFf=True  
     CERESnanprec=0.001 # precent of CERES fov (energy wise) to allow for nan values and use this FOV
     # print(MOD02)
     CERESjdate=ceres_data[0,:]
     CERESlat=ceres_data[1,:]
     CERESlon=ceres_data[2,:]
     CERESsza=ceres_data[3,:]
     CERESvza=ceres_data[4,:]
     CERESraa=ceres_data[5,:]
     Fsw=ceres_data[6,:]
     Isw=ceres_data[7,:]
     Flw=ceres_data[8,:]
     Ilw=ceres_data[9,:]
     AODfov=ceres_data[10,:]
     Clrfov=ceres_data[11,:]
     SSTfov=ceres_data[12,:]
     PWfov=ceres_data[13,:]
     DTzfov=ceres_data[14,:]
     # EISfov=ceres_data[15,:]
     
     del ceres_data
     
     foviswmax=0; fovilwmax=0
     maxbin=30
     minN=50
     ZColdClds=5000  # height to define cold clouds
     
     if PSFf: 
         # if np.shape(FOVmsk)[0]%2==0:
         PSF=FOVmsk_t.copy()
         tmp=np.shape(PSF)[0]-1
         # if np.shape(FOVmsk)[1]%2!=0:
         tmp1=np.shape(PSF)[1]
         fovsc=[tmp,tmp1]
         tmp=np.sum(PSF,axis=0)
         PSFc1=np.nonzero(tmp==np.max(tmp))[0][0]   
         PSFc2=tmp1-PSFc1-1
         # tmp=np.fliplr(PSF)
         # tmp=np.sum(tmp,axis=0)
         # PSFc2=np.nonzero(tmp==np.max(tmp))[0][0]  
     else:
         fovsc=np.shape(FOVmsk_t)[0]-1
         
     
     totpiece=-1  # cloud domains with CERES  FOVs
     Imkeepi=0  # index of images that are kept
     CloudDomain_i=-1 # all cloud domains with MODIS data
     NopxFOVmsk=np.size(PSF)    # number of modis pixels in CERES FOV
     Sampling_prob=100 # precentage of image sampling 

     visls=[0.47,0.66,0.85] # ,0.55
     nirls=[0.905,0.935] #,0.940
     swirls=[1.24,2.13,1.375]  # 1.24 out ,1.64 is out because its bad in AQUA
     thermalls=np.array([3.75,3.96,3.96,4.05,4.46,4.5,6.7,7.32,8.55,9.7,11.0,12.0,13.33,13.63,13.93,14.23])
    
     #---- added for ORG --------
     solarls=np.append(np.append(visls,nirls),swirls)
     lessthermalls=thermalls[thermal_channels]
     
     
     estimated_Npieces=Npices
     mfovp=100  # max number of CERES FOVs to analize in an image
     mpixp=5000 # max number of MODIS pixels for clear sky (usually around ~10,000 but can get as high as 20,000). The smaplying is random mpixp pixels
     
     
     sw_F=np.ones((estimated_Npieces,mfovp),dtype=np.float32)*np.nan
     sw_TSM1f=sw_F.copy()
     sw_TSM2f=sw_F.copy()
     sw_meanRHO=sw_F.copy()
     # sw_Fclr=sw_F.copy()
     sw_meanDFNC=sw_F.copy()
     sw_meanTSM=sw_F.copy()
     # swCERES_MOD_name=['nan']*Nsamples
     sw_meanSumI=sw_F.copy()
     sw_I=sw_F.copy()
     sw_AODfov=sw_F.copy()
     sw_Clrfov=sw_F.copy()
     # sw_Fclrspec=np.ones((estimated_Npieces,estimated_Npieces*mfovp,len(lessthermalls)+len(solarls)),dtype=np.float32)*np.nan
     sw_SSTfov=sw_F.copy()
     sw_PWfov=sw_F.copy()
     sw_DTzfov=sw_F.copy()
     # sw_EISfov=sw_F.copy()
     
     
     lw_F=sw_F.copy()
     lw_TSM1f=sw_F.copy()
     lw_TSM2f=sw_F.copy()
     lw_meanRHO=sw_F.copy()
     # lw_Fclr=sw_F.copy()
     lw_meanDFNC=sw_F.copy()
     lw_meanTSM=sw_F.copy()
     # lwCERES_MOD_name=['nan']*Nsamples
     lw_meanSumI=sw_F.copy()
     lw_I=sw_F.copy()
     # lw_Fclrspec=np.ones((estimated_Npieces,estimated_Npieces*mfovp,len(lessthermalls)+len(solarls)),dtype=np.float32)*np.nan

     CERESVZAsw=sw_F.copy() #np.ones((estimated_Npieces*mfovp),dtype=np.float32)*np.nan
     CERESRAAsw=CERESVZAsw.copy()
     CERESSZAsw=CERESVZAsw.copy()
     CERESSZAlw=CERESRAAsw.copy()
     CERESRAAlw=CERESRAAsw.copy()
     CERESVZAlw=CERESVZAsw.copy()
     # coloc=CERESVZAsw.copy()
     # cotime=CERESVZAsw.copy()
     
     
     
     
     # EPSsw=np.ones((estimated_Npieces,1))*np.nan
     # EPSlw=np.ones((estimated_Npieces,1))*np.nan
     # meanKsw=np.ones((estimated_Npieces,1))*np.nan
     # meanKlw=np.ones((estimated_Npieces,1))*np.nan
     # meanFswclr=np.ones((estimated_Npieces,1))*np.nan
     # meanFlwclr=np.ones((estimated_Npieces,1))*np.nan
     CloudFraction=np.ones((estimated_Npieces,1))*np.nan
     # mean_SZA=CloudFraction.copy()
     # mean_RAA=CloudFraction.copy()
     swDom_mean_eps=CloudFraction.copy()
     lwDom_mean_eps=CloudFraction.copy()
     
     #---- added for ORG --------
          
     PCSf_sw=CloudFraction.copy()  # sw pure-clear-sky fraction of domain such that CF+PSCf+TWf=1 
     PCSf_lw=CloudFraction.copy()  # lw pure-clear-sky fraction of domain such that CF+PSCf+TWf=1 
     Fall_sw=CloudFraction.copy()   # sw integrated scene flux
     Fall_lw=CloudFraction.copy()   # lw integrated scene flux
     
     Iorg=CloudFraction.copy()
     KScld=CloudFraction.copy()
     KSclr=CloudFraction.copy()
     maxDFNC=CloudFraction.copy()  # 99 percentile DFNC distribution to represent clear-sky void
     meanDFNC=CloudFraction.copy() 
     maxCLDa=CloudFraction.copy()  # 99 percentile of object (cloud) area distribution
     meanCLDa=CloudFraction.copy()  # mean object (cloud) area distribution
     Perimeter=CloudFraction.copy()# total perimeter of clouds
     # medCLDa=CloudFraction.copy()  # median object (cloud) area distribution
     # Fpcs_sw=CloudFraction.copy()   # PCS TOA sw flux. integrated CERES FOVs for entire domian  Can Calc this offline
     # Fpcs_lw=CloudFraction.copy()   # PCS TOA lw flux. integrated CERES FOVs for entire domian    Can Calc this offline
     COD95=np.ones((estimated_Npieces,1),dtype=np.float32)*np.nan  # 95 percentile of cloud optical thickness 
     COD25=COD95.copy()
     COD50=COD95.copy()
     ThinCldf=COD95.copy()
     meanCOD=COD95.copy()
     stdCTH=CloudFraction.copy()   # cloud top height STD
     CTH95=CloudFraction.copy()  # 95 percentile of CTH
     meanCTH=CloudFraction.copy()  # 95 percentile of CTH
     
     
     ImLon=COD95.copy()
     ImLat=COD95.copy()
     ImDate=COD95.copy()
     
     BT_waves=np.array([6.7,7.32,8.55,11,12])
     sw_waves=np.array([0.47,0.66,0.85,1.24,2.13])
     
     # window_wavs_i=[np.nonzero((value == lessthermalls)&((value>=8)&(value<=12)))[0][0] for value in BT_waves if np.size(np.nonzero((value == lessthermalls)&((value>=8)&(value<=12))))>0]
     window_wavs_i=np.nonzero(((BT_waves>=8)&(BT_waves<=12)))[0]
     BT_wavs_i=[np.nonzero(value == lessthermalls)[0][0] for value in BT_waves]
     sw_waves_i=[np.nonzero(value == solarls)[0][0] for value in sw_waves]
     BT_11_i=np.nonzero(11 == BT_waves)[0][0]
     BT_12_i=np.nonzero(12 == BT_waves)[0][0]
     
     stdBTlw_cld=np.ones((estimated_Npieces,len(window_wavs_i)),dtype=np.float32)*np.nan  # STD of cloudy pixel brightness temperature (to represent cloud top height std) for a few wavelengths
     minBTlw_cld=np.ones((estimated_Npieces,len(window_wavs_i)),dtype=np.float32)*np.nan  # 5 percentile of cloudy pixel brightness temperature to represent max cloud top (coldest clouds) and boundary layer  for a few wavelengths
     maxRefsw_cld=np.ones((estimated_Npieces,len(sw_waves)),dtype=np.float32)*np.nan  # 95 percentile of cloudy pixel reflectance to represent most optical thick clouds for a few wavelengths
     meanRefsw_cld=np.ones((estimated_Npieces,len(sw_waves)),dtype=np.float32)*np.nan  # mean cloudy pixel reflectance to represent most optical thick clouds for a few wavelengths
     meanBTlw_clr=np.ones((estimated_Npieces,len(BT_waves)),dtype=np.float32)*np.nan  # mean radiance of pcs pixel to represent SST and humidity  for a few wavelengths (6.7,7.2,8.55,11,12)
     stdBT1112_clr=CloudFraction.copy()   # std BT11-BT12 to represent humidity variability
     meanBT1112_clr=CloudFraction.copy()  # mean BT11-BT12 to represent humidity
     #---------------------------------------------------------------
     
     dtNtm=MOD02[9:22]
     
     MODISday=float(dtNtm[1:11])  
     
# ------------------------ Reading MODIS L2 images ------------------------------------
     temp="EV_500_Aggr1km_RefSB" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     Rsw500=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_250_Aggr1km_RefSB" #"EV_1KM_RefSB"
     Rsw250=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     
     temp="EV_1KM_RefSB" #"EV_1KM_RefSB"
     Rswkm=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     # vis=np.append(Rsw500[0:2],Rsw250,axis=0) # ORIG
     vis=np.stack((Rsw500[0],Rsw250[0],Rsw250[1]),axis=0)
     
     nir=np.stack((Rswkm[-4],Rswkm[-3]),axis=0)
     # nir=Rswkm[-4:-1]  # ORIG
     # nir=np.expand_dims(Rswkm[-4],axis=0).copy()
     
     # tmp=np.expand_dims(Rswkm[-1], axis=0).copy()  # readded 164
     # swir=np.append(Rsw500[-3:],tmp,axis=0)
     swir=np.stack((Rsw500[-3],Rsw500[-1],Rswkm[-1]),axis=0) # ORIG,  
     # swir=np.expand_dims(Rsw500[-3],axis=0).copy()
     # swir=np.reshape(swir,(1,np.shape(swir)[0],np.shape(swir)[1]))
     # swir=np.transpose(swir, (2, 0, 1)) # ORIG
     
     temp="EV_1KM_Emissive" #"EV_1KM_RefSB"
     sclsNoffs="radiance"
     thermal=readMODIS(path+'/MOD02/'+MOD02,temp,sclsNoffs)
     thermalfull=thermal #PlanckTemp(thermalls*1e-6,thermal)
     thermal=thermalfull[thermal_channels]
     del Rswkm,Rsw250,Rsw500
     
     sw_meanSolarspec=np.ones((estimated_Npieces,mfovp,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]),dtype=np.float32)*np.nan
     lw_meanSolarspec=sw_meanSolarspec.copy()
     sw_meanThermalspec=np.ones((estimated_Npieces,mfovp,np.shape(thermal)[0]),dtype=np.float32)*np.nan
     lw_meanThermalspec=sw_meanThermalspec.copy() 
     
     MODclrsky_sw=np.ones((estimated_Npieces,mpixp,np.shape(vis)[0]+np.shape(nir)[0]+np.shape(swir)[0]+np.shape(thermal)[0]),dtype=np.float32)*np.nan
     MODclrsky_lw=MODclrsky_sw.copy()
     
     MODangles_sw=np.ones((estimated_Npieces*3,mpixp,3),dtype=np.float32)*np.nan
     MODangles_lw=MODangles_sw.copy()
     
     CERES_CldDomain_isw=np.ones((estimated_Npieces*3),dtype=np.float32)*np.nan
     CERES_CldDomain_ilw=CERES_CldDomain_isw.copy()
 #---------------------------------------Reading cloud mask ------------------------------
     temp = nc.Dataset(path+'/MOD35/'+MOD35, 'r')
     # print(temp.variables.keys()) # 
     MOD35D = temp.variables["Cloud_Mask"]
     # print(MOD35D) # 
     
     # fillV=temp.variables[MOD35D].getncattr('_FillValue')
     # CldMsk[CldMsk==fillV]=np.nan
     
     MODsz=np.shape(thermal)
     MODsz=MODsz[1:3]
     CldMsk=np.zeros(MODsz)
     glint=np.zeros(MODsz)
     sfcice=np.zeros(MODsz)
     land=np.zeros(MODsz)
     thinCirr=np.zeros(MODsz)
     NonCld=np.zeros(MODsz)
     byte1=np.array(MOD35D[0,:,:])
     byte1[byte1<0]=256+byte1[byte1<0]
     byte2=np.array(MOD35D[1,:,:])
     byte2[byte2<0]=256+byte2[byte2<0]  # correct for negative values
     for i in range(0,MODsz[0]):
         for j in range(0,MODsz[1]):  
             # print(bin(int(byte1[i,j])))
             binary=format(int(byte1[i,j]),'#010b').replace("0b","")
             binary=binary[::-1]
             binary2=format(int(byte2[i,j]),'#010b').replace("0b","")
             binary2=binary[::-1]
             # print(binary)
             if int(binary[0])==1: # & int(binary[-4])==dayf:  # if mask is determined and is day/night
                 CldMsk[i,j]=int(binary[1:3],2)  
                 glint[i,j]=int(binary[4],2)
                 sfcice[i,j]=int(binary[5],2)
                 land[i,j]=int(binary[6:],2)
                 bit10=int(binary2[1])
                 # bit12=int(binary2[3])
                 thinCirr[i,j]=bit10
                 NonCld[i,j]=int(binary2[0])
     
     BWmsk=np.copy(CldMsk)
     if cldmsktyp=='relaxed':  # take only 95% confident as cloud          
        BWmsk[BWmsk>0]=1
        BWmsk=BWmsk.astype(bool)
        BWmsk=np.invert(BWmsk)   
        if consider_thinCirr:
            BWmsk[thinCirr==1]=1
     if cldmsktyp=='strict':    #cloud even if only 33% confident      
        BWmsk[BWmsk==0]=1
        BWmsk[BWmsk>1]=0
        BWmsk=BWmsk.astype(bool)
        if consider_thinCirr:
            BWmsk[thinCirr==1]=1
                 
     # --------------------- Reading clouds --------------------------------
     var="Cloud_Phase_Infrared_1km" 
     dataset = nc.Dataset(path+'MOD06/'+MOD06, 'r')
     # print(dataset.variables.keys()) # 
     mtrx = dataset.variables[var]
     # print(mtrx) # 
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     # mtrx=np.ma.masked_array(mtrx, mtrx==fillV) # mtrx[mtrx==fillV]=np.nan
     mtrx=np.where(mtrx==fillV,np.nan,mtrx) # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldPhs=mtrx.copy() #(mtrx-offset)*scales
     # CldPhs=scipy.ndimage.zoom(CldPhs, 5, order=0)
     
     
     var="cloud_top_height_1km"
     mtrx = dataset.variables[var]
     # print(mtrx) # 
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     CldTop=mtrx.copy() #(mtrx-offset)*scales
     
     
     # ---------- aded for ORG ---------------------------    
     var="Cloud_Optical_Thickness_1621"
     mtrx = dataset.variables[var]
     mtrx=np.array(mtrx) 
     mtrx=mtrx.astype('float')
     fillV=dataset.variables[var].getncattr('_FillValue')
     mtrx=np.where(mtrx==fillV,np.nan,mtrx)  # mtrx[mtrx==fillV]=np.nan
     scalesn='scale_factor'
     offsn='add_offset'
     scales=dataset.variables[var].getncattr(scalesn)
     offset=dataset.variables[var].getncattr(offsn)
     COD=mtrx.copy() #(mtrx-offset)*scales
     
     
     # ------------------ GEOLOCATION ---------------------------------------------------------
     # dataset = nc.Dataset(MODpath+MOD03, 'r')
     # print(dataset.variables.keys()) # 
     # mtrx = dataset.variables[var]
     # print(mtrx) # 
     
     temp="Latitude" #"EV_1KM_RefSB"
     sclsNoffs=0
     Lat=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     temp="Longitude" #"EV_1KM_RefSB"
     Lon=readMODIS(path+'/MOD03/'+MOD03,temp,sclsNoffs)
     
     temp="SensorZenith" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarZenith" #"EV_1KM_RefSB"
     SolZen=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SolZen=SenZen*0.01
     
     
     temp="SensorAzimuth" #"EV_1KM_RefSB"
     sclsNoffs=0
     SenAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     # SenZen=SenZen*0.01
     temp="SolarAzimuth" #"EV_1KM_RefSB"
     SolAz=readMODIS(path+'MOD03/'+MOD03,temp,sclsNoffs)
     
     RAA=SolAz-SenAz-180
     RAA=np.where(RAA<0,RAA+360,RAA)
     RAA=np.abs(RAA-180)
     
     del SolAz, SenAz    
      
     
     # #################################### Calc distances ##################################################        
     DFNC=scipy.ndimage.distance_transform_edt(1-BWmsk)
     xx,yy=np.meshgrid(np.array(range(1,MODsz[1]+1,1)),np.array(range(1,MODsz[0]+1,1)))
     xx=np.minimum(xx,np.fliplr(xx))
     yy=np.minimum(yy,np.flipud(yy))
     tmp=np.minimum(yy,xx)   # smallest distance to image edge
     DFNC[DFNC>tmp]=np.nan
     
     tmp=np.copy(land)
     tmp[tmp>0]=1
     tmp=tmp.astype(bool)
     tmp=np.invert(tmp)
     DFL=scipy.ndimage.distance_transform_edt(tmp)
     
     tmp=np.copy(CldPhs)
     tmp[tmp==1]=0
     tmp[tmp==6]=0
     tmp[(tmp==2) | (tmp==3)]=1
     tmp=tmp.astype(bool)
     DFIC=scipy.ndimage.distance_transform_edt(1-tmp)
     
     tmp=np.copy(CldPhs)
     tmp[tmp<6]=0
     tmp[tmp==6]=1
     tmp=tmp.astype(bool)
     DFuncertPhs=scipy.ndimage.distance_transform_edt(1-tmp)
     
     if NoColdCld==1:
         tmp=CldTop>ZColdClds
         DFHC=scipy.ndimage.distance_transform_edt(1-tmp)
     elif NoColdCld==2:
         tmp=CldTop<ZColdClds
         DFHC=scipy.ndimage.distance_transform_edt(1-tmp)
     elif NoColdCld==0:
         DFHC=[]
     
     glint=1-glint
     glint=glint.astype(bool)
     
     unwanted=np.zeros((MODsz[0],MODsz[1]), dtype=bool)
     unwanted[DFL<10]=True
     unwanted[glint]=True
     unwanted[SenZen>20]=True
     
     if NoColdCld==1:
         unwanted[DFuncertPhs<10]=True   # restrictions!
         unwanted[DFHC<15]=True
         unwanted[DFIC<15]=True
     elif NoColdCld==2:
         unwanted[DFHC<10]=True  # these are low clouds in this case
     
     tmp2=np.stack([unwanted]*np.shape(nir)[0],axis=0)
     nir[tmp2]=np.nan
     tmp2=np.stack([unwanted]*np.shape(swir)[0],axis=0) 
     swir[tmp2]=np.nan 
     tmp2=np.stack([unwanted]*np.shape(vis)[0],axis=0) 
     vis[tmp2]=np.nan
     tmp2=np.stack([unwanted]*np.shape(thermal)[0],axis=0)
     thermal[tmp2]=np.nan
     DFNC[unwanted]=np.nan
     del MOD35D, glint, sfcice, land, thinCirr, NonCld, CldPhs, DFL, DFIC, DFuncertPhs, DFHC
     # -------------------------- Finding pure clear sky --------------------------
     DFNCbins=np.array(range(0,maxbin,1))
     
     if Npices==80:
         DX=110; DY=101; ri=0-DY 
         splitganules=[457,568,679,790];
     elif Npices==20:  
         DX=220; DY=203; ri=0-DY
         splitganules=[457,678]; splitf=bool(1)
     else:
         print('Bad subdomain input')
     
     estimated_image_samp=estimated_Npieces*Sampling_prob/100    
     ImSamp=np.ones((DY,DX,estimated_Npieces,2))*np.nan # keep two images and other scence variables
     ImSamp_metrics=np.ones((estimated_Npieces,20))*np.nan  # keep scalar metrics of the cloud fields. CHANGE 4th DIMENTION WHEN ADDING A METRIC!!!
     
     # xx,yy=np.meshgrid(np.array(range(1,DX+1,1)),np.array(range(1,DY+1,1)))
     # xx=np.minimum(xx,np.fliplr(xx))
     # yy=np.minimum(yy,np.flipud(yy))
     # edge_dist=np.minimum(yy,xx)   # smallest distance to image edge
     
     for piece in range(1,Npices+1):
         # print('sub domian-'+ str(piece) )
         
         
         if Npices==20:   # how to move the subdomain (depend on Npices)
             splitf=(1-splitf)
             ci=splitganules[splitf]
             if splitf==0:
                 ri=ri+DY
         elif Npices==80:
             splitf=np.fmod(piece,4)
             ci=splitganules[splitf-1]
             if splitf==1:
                 ri=ri+DY 
             
         # print(ci,ri) 
         subDFNC=DFNC[ri:ri+DY,ci:ci+DX].copy()
         # subDFNC[subDFNC>edge_dist]=np.nan
         
         BWmsksub=BWmsk[ri:ri+DY,ci:ci+DX].copy()
         
         if np.nansum(unwanted[ri:ri+DY,ci:ci+DX])>DX*DY*0.2:  # restriction!
             # print('unwanted > 50%')
             continue
         
         # ---------- added ORG ---------------
         
         subCTH=CldTop[ri:ri+DY,ci:ci+DX].copy()
         if NoColdCld==1:
             if np.nansum(subCTH>ZColdClds)>0.25*DX*DY:     # restriction!   
             #!!!    dont analyze fields with deep clouds cover of more than 5%
                 continue
         subCOD=COD[ri:ri+DY,ci:ci+DX].copy()
         
         tmp=np.concatenate((vis,nir,swir),axis=0)
         subSolar=tmp[:,ri:ri+DY,ci:ci+DX].copy()
         subThermal=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         # ---------- added ORG ---------------
         subBT=Planckfunc(lessthermalls*1e-6,subThermal,True)
         
         subSolZen=SolZen[ri:ri+DY,ci:ci+DX].copy()
         subSenZen=SenZen[ri:ri+DY,ci:ci+DX].copy()
         subRAA=RAA[ri:ri+DY,ci:ci+DX].copy()
         
         # CF=np.nansum(BWmsksub)/np.size(BWmsksub)                   
         tmp=BWmsksub.copy()
         tmp[unwanted[ri:ri+DY,ci:ci+DX]]=0
         #!!! should I consider NaNs in CF calc.  DONT FORGET TO CHANGE ALSO FOR PCS FRACTION!!!!
         # CF=np.nansum(tmp)/(DX*DY) #np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))   
         CF=np.nansum(tmp)/np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))
         # print('CF=',CF)
         if (CF>0.95) or (CF<0.05):
              # print('cloud fration > 80%')                # restriction!
             continue
         
         tmp=BWmsksub.copy()
         
         
      # ********************************************************************
         nirclr=np.zeros((2,len(nirls)))
         visclr=np.zeros((2,len(visls)))
         swirclr=np.zeros((2,len(swirls)))
         thermalclr=np.zeros((2,np.shape(thermal)[0]))
         if MaxDFNC>0:
             if np.nansum(subDFNC>MaxDFNC)<10:
                 continue       # restriction!
             # print('shapes',np.shape(vis)[0],np.shape(nir)[0],np.shape(swir)[0],np.shape(thermal)[0])
             tmp2=np.where(subDFNC>MaxDFNC)
             for i in range(0,np.shape(nir)[0]):
                 tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()                    
                 nirclr[0,i]=np.nanmean(tmp[tmp2])
                 nirclr[1,i]=np.nanvar(tmp[tmp2])

             for i in range(0,np.shape(vis)[0]):
                 tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                 visclr[0,i]=np.nanmean(tmp[tmp2])
                 visclr[1,i]=np.nanvar(tmp[tmp2])
                     
             for i in range(0,np.shape(swir)[0]):       
                 tmp=swir[i,ri:ri+DY,ci:ci+DX].copy() 
                 swirclr[0,i]=np.nanmean(tmp[tmp2])
                 swirclr[1,i]=np.nanvar(tmp[tmp2])
                     
             for i in range(0,np.shape(thermal)[0]):      
                 tmp=thermal[i,ri:ri+DY,ci:ci+DX].copy()
                 thermalclr[0,i]=np.nanmean(tmp[tmp2])
                 thermalclr[1,i]=np.nanvar(tmp[tmp2])
                 # ------ start collecting subdomain (piece) data ----------------
                 
         else:
             
             #--------- Radiance vs DFNC for vis, nir,swir,thermal
                 vis_dfnc,visDstd,visDN=bindata(vis[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 
                 tmp=vis_dfnc[0,:].copy()
                 lastbin=np.array(np.where(1-np.isnan(tmp)))
                 if (lastbin.size==0) or (np.nansum(np.isnan(tmp[1:6]))>1):
                     # print('no data as DFNC');
                     continue
                 lastbin=lastbin[0,-1]+1
                 if lastbin<7: 
                     # print('DFNC<7km')
                     continue           # restriction!
                 nir_dfnc,nirDstd,nirDN=bindata(nir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 swir_dfnc,swirDstd,swirDN=bindata(swir[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
                 thermal_dfnc,thermalDstd,thermalDN=bindata(thermal[:,ri:ri+DY,ci:ci+DX],subDFNC,DFNCbins,minN)
             
     
                 noisfacsw=0.05 # When noise factor decreases the algorithm has less tolorance to noise in saturation so saturation point get farther away.
                 noisfaclw=0.1
                 vissat,vissatdis,vis_fndsat=findsat(vis_dfnc[:,1:],noisfacsw,0) #DFNCbins
                 nirsat,nirsatdis,nir_fndsat=findsat(nir_dfnc[:,1:],noisfacsw,0)
                 swirsat,swirsatdis,swir_fndsat=findsat(swir_dfnc[:,1:],noisfacsw,0)
                 thermalsat,thermalsatdis,thermal_fndsat=findsat(thermal_dfnc[:,1:],noisfaclw,0)
                 
                 
                 if channel_adjusting:
                     tmp=np.reshape(np.concatenate((vis_fndsat,nir_fndsat,swir_fndsat),axis=0),(1,len(vis)+len(nir)+len(swir)))
                    # if totpiece==1:
                     #    FNDSATQsw=tmp
                     #    FNDSATQlw=np.reshape(thermal_fndsat,(1,len(thermal)))
                    # else:
                    #     FNDSATQsw=np.concatenate((FNDSATQsw,tmp),axis=0)
                    #     FNDSATQlw=np.concatenate((FNDSATQlw,np.reshape(thermal_fndsat,(1,len(thermal)))),axis=0)

                     if BroadLW:
                        BadLWchan=5
                     else:
                        BadLWchan=1 
                     if (np.nansum(tmp==-1)>4) or (np.nansum(thermal_fndsat==-1)>BadLWchan):
                         continue
                     
                 
                 for i in range(0,np.shape(thermal)[0]):
                     if i<3:
                         tmp=nir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>nirsatdis[i])
                         nirclr[0,i]=np.nanmean(tmp[tmp2])
                         nirclr[1,i]=np.nanvar(tmp[tmp2])
                         
                         tmp=swir[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>swirsatdis[i])  # changed when removed 1.64 04/12/24
                         swirclr[0,i]=np.nanmean(tmp[tmp2])
                         swirclr[1,i]=np.nanvar(tmp[tmp2])
                     if i<4:
                         tmp=vis[i,ri:ri+DY,ci:ci+DX].copy()
                         tmp2=np.where(subDFNC>vissatdis[i])
                         visclr[0,i]=np.nanmean(tmp[tmp2])
                         visclr[1,i]=np.nanvar(tmp[tmp2])
                         
                         
                     tmp=thermal[i,ri:ri+DY,ci:ci+DX].copy()
                     tmp2=np.where(subDFNC>thermalsatdis[i])
                     thermalclr[0,i]=np.nanmean(tmp[tmp2])
                     thermalclr[1,i]=np.nanvar(tmp[tmp2])
         
         
                     # extentvec=np.concatenate((vissatdis,nirsatdis,swirsatdis,thermalsatdis),axis=0)
        
         # ------------------------- corners ----------------- 
         sublon=Lon[ri:ri+DY,ci:ci+DX] # MODIS SUB-AREA LON AND (LAT)
         sublat=Lat[ri:ri+DY,ci:ci+DX]
          
         North=np.max(sublat)  # MODIS SUB-AREA MAX POSITIONS
         South=np.min(sublat)
         East=np.max(sublon)
         West=np.min(sublon)
          
         if (np.isnan(North)) or (np.isnan(South)) or (np.isnan(West)) or (np.isnan(East)):
              print('<><><><><><><><><><><><><> Max coordinate is Nan <><><><><><><><><><><><><>' )
              continue
          
         if (East>0) & (West<0) & (East-West>150): 
              tmp=East.copy()
              East=West.copy()
              West=tmp.copy()    
        
          # ------ start collecting subdomain (piece) data ----------------
         totpiece=totpiece+1
         CloudFraction[totpiece]=CF
         
         # ------------ added ORG -------------------
         tmpMASK=BWmsksub.copy()
         
         tmp=subBT[BT_wavs_i,:,:]
         tmp=np.where(tmpMASK,tmp,np.nan)
         stdBTlw_cld[totpiece,:]=np.nanstd(tmp[window_wavs_i,:,:],axis=(1,2))  # STD of cloudy pixel brightness temperature (to represent cloud top height std) for a few wavelengths
         minBTlw_cld[totpiece,:]=np.nanpercentile(tmp[window_wavs_i,:,:],5,axis=(1,2))  # 5 percentile of cloudy pixel brightness temperature to represent max cloud top (coldest clouds) and boundary layer  for a few wavelengths
         
         tmp=subSolar[sw_waves_i,:,:]
         maxRefsw_cld[totpiece,:]=np.nanpercentile(tmp,95,axis=(1,2))   # 95 percentile of cloudy pixel reflectance to represent most optical thick clouds for a few wavelengths
         meanRefsw_cld[totpiece,:]=np.nanmean(tmp,axis=(1,2))  
         
         tmp=subBT[BT_wavs_i,:,:]
         tmp=np.where(subDFNC<15,np.nan,tmp)        
         meanBTlw_clr[totpiece,:]=np.nanmean(tmp,axis=(1,2))  # mean radiance of pcs pixel to represent SST and humidity  for a few wavelengths 
         # print(np.shape(tmp[BT_11_i,:,:]-tmp[BT_12_i,:,:]))
         stdBT1112_clr[totpiece]=np.nanstd(tmp[BT_11_i,:,:]-tmp[BT_12_i,:,:],axis=(0,1))  # std BT11-BT12 to represent humidity variability
         meanBT1112_clr[totpiece]=np.nanmean(tmp[BT_11_i,:,:]-tmp[BT_12_i,:,:],axis=(0,1))  # mean BT11-BT12 to represent humidity
         
         
         # mean_SZA[totpiece]=np.nanmean(SolZen[ri:ri+DY,ci:ci+DX].ravel())
         # mean_RAA[totpiece]=np.nanmean(RAA[ri:ri+DY,ci:ci+DX].ravel())
         # solarls=np.concatenate((visls,nirls,swirls),axis=0)
         clrswspec=np.concatenate((visclr,nirclr,swirclr),axis=1)
         # specclr=np.concatenate((visclr,nirclr,swirclr,thermalclr),axis=1)
    
         
         # ---------------- solar -----------------------------------  
         tmp2=np.concatenate((vis[:,ri:ri+DY,ci:ci+DX],nir[:,ri:ri+DY,ci:ci+DX],swir[:,ri:ri+DY,ci:ci+DX]),axis=0)
         a= clrswspec[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=clrswspec[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMsw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMsw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
             
    
         TSMsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # clrstdsum=np.sum(np.sqrt(specclr[1,:]))
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorsw=tmp3/tmp4            # SHORTWAVE RHO (sumInc/sumIclr)
         TWfactorsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         swDom_mean_eps[totpiece]=np.nanmean(TWfactorsw)
         
         SumIsw=tmp3.copy()           # SHORTWAVE RHO (sumInc/sumIclr)
         SumIsw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         SumIsw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer=(tmp3<=tmp4+clrstdsum) & (tmp3>=tmp4-clrstdsum)
         # TWfac_uncer[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfactor[(TWfac_uncer) & (1-np.isnan(TWfac_uncer))]=1
      # ---------------- thermal -----------------------------------  
         tmp2=thermal[:,ri:ri+DY,ci:ci+DX].copy()
         a=thermalclr[0,:].copy()
         a=np.tile(a[:, np.newaxis, np.newaxis], (1,DY,DX)) # (1,MODsz[0],MODsz[1]))
         b=thermalclr[1,:].copy()
         b=np.tile(b[:, np.newaxis, np.newaxis], (1,DY,DX))
         
         if weightTSM:
             TSMlw=np.nansum(np.sqrt(((tmp2-a)**2/b))*a,axis=0)/np.nansum(a,axis=0) # SHORTWAVE TWILIGHT-SPECTRAL-MEASURE
         else:
             TSMlw=np.nanmean(np.sqrt(((tmp2-a)**2/b)),axis=0) # LongWAVE TWILIGHT-SPECTRAL-MEASURE
             
             
         TSMlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TSMlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
    
         # clrstdsum=np.sum(np.sqrt(specclr[1,:]))
         tmp3=np.sum(tmp2,axis=0)
         tmp4=np.sum(a,axis=0)
         TWfactorlw=tmp3/tmp4             # LONGWAVE RHO (sumInc/sumIclr)
         TWfactorlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         TWfactorlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         
         lwDom_mean_eps[totpiece]=np.nanmean(TWfactorlw)

         SumIlw=tmp3.copy()           # LHORTWAVE RHO (sumInc/sumIclr)
         SumIlw[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         SumIlw[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer=(tmp3<=tmp4+clrstdsum) & (tmp3>=tmp4-clrstdsum)
         # TWfac_uncer[unwanted[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfac_uncer[BWmsk[ri:ri+DY,ci:ci+DX]]=np.nan
         # TWfactor[(TWfac_uncer) & (1-np.isnan(TWfac_uncer))]=1
         
         # ------------------- added ORG ----------------------
         tmp=BWmsksub.copy()
         tmp[unwanted[ri:ri+DY,ci:ci+DX]]=0
         # L, N = label_skim(tmp, connectivity=2, return_num=True)
         # # Get region properties (Centroid and Area)
         # props = regionprops(L)
         # Areas=np.ones(N)*np.nan
         # surface_length=Areas.copy()
         # for rr in range(N):
         #     region=props[rr]
         #     Areas[rr]=region.area  # *PxL**2 pixel size is 1km 
         #     surface_length[rr]=region.perimeter_crofton # *PxL pixel size is 1km 
         # labeled_image, num_features = label(BWmsksub)
         # size=np.ones((num_features))*np.nan
         # for kk in range(0,num_features):           
         #     size[kk]=np.nansum(labeled_image==kk+1)
         # meanCLDsz=np.nanmean(size)
         # medCLDsz=np.nanmedian(size)
         # maxCLDsz=np.nanpercentile(size,99)
         
         tmpIorg,tmpCLDf,tmpPerimeter,Areas=Iorgcalc(tmp,1,2)
         tmpKScld,tmpKSclr=LvL(tmp)
         
         if np.abs(tmpCLDf-CloudFraction[totpiece])>0.05:
             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!! CF and CF of Iorg func dont agree !!!!!!!!!')
         
         Iorg[totpiece]=tmpIorg
         KScld[totpiece]=tmpKScld
         KSclr[totpiece]=tmpKSclr
         
         maxDFNC[totpiece]=np.nanpercentile(subDFNC, 99) # 99 percentile DFNC distribution to represent clear-sky void
         meanDFNC[totpiece]=np.nanmean(subDFNC) 
         maxCLDa[totpiece]=np.nanpercentile(Areas, 99) # 99 percentile of object (cloud) area distribution
         meanCLDa[totpiece]=np.nanmean(Areas)
         Perimeter[totpiece]=tmpPerimeter.copy() #np.nansum(surface_length)
         
         stdCTH[totpiece] = np.nanstd(subCTH)
         CTH95[totpiece]= np.nanpercentile(subCTH,95)
         meanCTH[totpiece]= np.nanmean(subCTH)
         COD95[totpiece]= np.nanpercentile(subCOD,95)
         meanCOD[totpiece]=np.nanmean(subCOD)
         COD50[totpiece]= np.nanpercentile(subCOD,50)
         COD25[totpiece]= np.nanpercentile(subCOD,25)
         
         tmp=np.where(subCOD<3,1,0)
         tmp[unwanted[ri:ri+DY,ci:ci+DX]]=0
         ThinCldf[totpiece]=np.nansum(tmp)/np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))
         
         ImLon[totpiece]=sublon[round(DY/2),round(DX/2)]
         ImLat[totpiece]=sublat[round(DY/2),round(DX/2)]
         ImDate[totpiece]=MODISday
         
             
             
             
     # -------------- CERES FOVs ----------------------------
         CEREStoday=np.where(CERESjdate==MODISday)[0]
         
         # my_column_names = ['Ffov','meanfovTWf','TSM>1_count']  # DATA FRAMES
         # DFswnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFswclr = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwnc = pd.DataFrame(data=[], columns=my_column_names)
         # DFlwclr = pd.DataFrame(data=[], columns=my_column_names)
         
         fovisw=0
         fovilw=0
         #--------- added ORG------------
         intFsw=0 # accumulating FOVs fluxes to get the mean for the domain (Fall)
         intFlw=0
         Nfovssw=0
         Nfovslw=0
         
         for fov in CEREStoday:
             if (np.isnan(Flw[fov]) & np.isnan(Fsw[fov])) or (Fsw[fov]<1) or (CERESvza[fov]>25): continue  # don't analyze CERES fov if its Nan/nightime(SW=0)/Not NADIR VZA
             
             fovlat=CERESlat[fov] # CERES FOV LAT (AND LON)
             fovlon=CERESlon[fov]
             if ((fovlat<North) & (fovlat>South)) & ((fovlon>West) & (fovlon<East)):
                 
                 Cdiff=np.sqrt((sublat-fovlat)**2+(sublon-fovlon)**2)
                 subsubC=np.nonzero(Cdiff==np.min(Cdiff)) # CERES FOV LAT IN MODIS SUB-AREA COORDINATES
                 if subsubC[0].size==0:
                     continue
                 subsubCy=subsubC[0][0].copy(); subsubCx=subsubC[1][0].copy()
                 
                 # Finding the location of neighbor FOV                    # changed 9/7 S
                 fovlat2=CERESlat[fov+1] 
                 fovlon2=CERESlon[fov+1]
                 Cdiff2=np.sqrt((sublat-fovlat2)**2+(sublon-fovlon2)**2)
                 subsubC2=np.nonzero(Cdiff2==np.min(Cdiff2))
                 if subsubC2[0].size==0:
                     print('########## No min diff')
                     continue
                 subsubCx2=subsubC2[1][0]                 # changed 9/7 E
                 if PSFf:
                     if subsubCx2<subsubCx:   # changed 9/7 S
                         PSFc=PSFc2.copy()
                         FOVmsk=np.fliplr(PSF)
                     else:
                         PSFc=PSFc1.copy()
                         FOVmsk=PSF.copy()  # changed 9/7 E
                     subsubRi=int(subsubCy-fovsc[0]/2)
                     subsubRf=int(subsubCy+fovsc[0]/2)
                     subsubCi=int(subsubCx-(PSFc-1))
                     subsubCf=int(subsubCx+(fovsc[1]-PSFc))
                 else:
                     subsubRi=int(subsubCy-fovsc/2)
                     subsubRf=int(subsubCy+fovsc/2)
                     subsubCi=int(subsubCx-fovsc/2)
                     subsubCf=int(subsubCx+fovsc/2)
                 
                 if (subsubRi<0) | (subsubRf>DY-1) | (subsubCi<0) | (subsubCf>DX-1): # if FOV is out of the image continue
                     # print('out of bound!!!',subsubRi,subsubRf,subsubCi,subsubCf)
                     continue
                 
                 
     # ------------------------------- calculating for CERES FOV SW ---------------------------------------- 
                 FswNAN=True
                 if np.isnan(Fsw[fov])==False:
                     FswNAN=False
                     if subsubRf==DY-1:
                        subsubTSMsw=TSMsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWfsw=TWfactorsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                        subSumIsw=SumIsw[subsubRi:,subsubCi:subsubCf+1].copy()
                        thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                        solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                         subSumIsw=SumIsw[subsubRi:subsubRf+1,subsubCi:].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMsw=TSMsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWfsw=TWfactorsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subSumIsw=SumIsw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                     
                     # ------------- added ORG -----------------
                     intFsw=intFsw+Fsw[fov]
                     Nfovssw=Nfovssw+1
                     if ~np.isnan(Flw[fov]):
                         intFlw=intFlw+Flw[fov]
                         Nfovslw=Nfovslw+1
                     #----------------------------------------  
                    
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMsw))
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         if np.sum(FOVmsk[tmpy,tmpx])>CERESnanprec:
                             continue
                         
                          
                         
                         sw_meanTSM[totpiece,fovisw]=np.nansum(subsubTSMsw*FOVmsk)
                         sw_meanRHO[totpiece,fovisw]=np.nansum(subsubTWfsw*FOVmsk)
                         sw_meanSumI[totpiece,fovisw]=np.nansum(subSumIsw*FOVmsk)
                         sw_meanDFNC[totpiece,fovisw]=np.sum(subsubDFNC*FOVmsk)
                         tmp=np.nonzero(subsubTSMsw>1)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         sw_TSM1f[totpiece,fovisw]=np.nansum(subsubTSMsw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMsw*FOVmsk)
                         
                         tmp=np.nonzero(subsubTSMsw>3)  # find where TSM>3 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         sw_TSM2f[totpiece,fovisw]=np.nansum(subsubTSMsw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMsw*FOVmsk)
                         
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                         tmp2=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                         sw_meanSolarspec[totpiece,fovisw,:]=tmp2
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                         sw_meanThermalspec[totpiece,fovisw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                         
                     else:
                         if np.sum(np.isnan(subsubTSMsw[FOVmsk]))>0:
                             continue
                         subsubTSMsw[(~FOVmsk)]=np.nan 
                         subsubTWfsw[(~FOVmsk)]=np.nan
                         subSumIsw[(~FOVmsk)]=np.nan
                         subsubDFNC[(~FOVmsk)]=np.nan
                         sw_TSM1f[totpiece,fovisw]=np.nansum(subsubTSMsw>1)/NopxFOVmsk
                         sw_meanRHO[totpiece,fovisw]=np.nanmean(subsubTWfsw)
                         sw_meanTSM[totpiece,fovisw]=np.nanmean(subsubTSMsw)
                         sw_meanDFNC[totpiece,fovisw]=np.nanmean(subsubDFNC)
                         sw_meanSumI[totpiece,fovisw]=np.nanmean(subSumIsw)
                         
                     CERESSZAsw[totpiece,fovisw]=CERESsza[fov]     
                     CERESRAAsw[totpiece,fovisw]=CERESraa[fov]    
                     CERESVZAsw[totpiece,fovisw]=CERESvza[fov]
                     # cotime[totpiece,fovisw]=CERESjdate[fov]-MODISday
                     # coloc[totpiece,fovisw]=np.min(Cdiff)
                     
                     # swCERES_MOD_name[totpiece,fovisw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                     sw_F[totpiece,fovisw]=Fsw[fov] 
                     sw_I[totpiece,fovisw]=Isw[fov]
                     sw_AODfov[totpiece,fovisw]=AODfov[fov]
                     sw_Clrfov[totpiece,fovisw]=Clrfov[fov]
                     
                     sw_SSTfov[totpiece,fovisw]=SSTfov[fov]
                     sw_PWfov[totpiece,fovisw]=PWfov[fov]
                     sw_DTzfov[totpiece,fovisw]=DTzfov[fov]
                     # sw_EISfov[totpiece,fovisw]=EISfov[fov]
                     
                     # sw_Fclrspec[totpiece,fovisw,:]=specclr[0,:]
                  #  sw_FclrspecSTD[totpiece,fovisw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
                     
                     # if sw_meanDFNC[totpiece,fovisw]<12:
                     #    fig2 = plt.figure(figsize=(21,11))
                     #    ax1 = fig2.add_subplot(2, 2, 2)
                     #    imgplot1 = plt.imshow(subDFNC,cmap=mpl.colormaps['turbo'])
                     #    imgplot1.set_clim(0, 50)
                     #    plt.plot(subsubCx,subsubCy,'.m')
                     #    plt.plot([subsubCx-22,subsubCx-22,subsubCx+22,subsubCx+22],[subsubCy-15,subsubCy+15,subsubCy-15,subsubCy+15],'*m')
                     #    ax1 = fig2.add_subplot(2, 2, 1)
                     #    imgplot1 = plt.imshow(TSMsw,cmap=mpl.colormaps['turbo'])
                     #    imgplot1.set_clim(1, 5)
                     #    plt.plot(subsubCx,subsubCy,'.m')
                     #    plt.plot([subsubCx-22,subsubCx-22,subsubCx+22,subsubCx+22],[subsubCy-15,subsubCy+15,subsubCy-15,subsubCy+15],'*m')
                     #    ax2 = fig2.add_subplot(2, 2, 3)
                     #    imgplot2 = plt.imshow(subsubTSMsw,cmap=mpl.colormaps['turbo'])
                     #    imgplot2.set_clim(1, 5)
                     #    plt.title('TSMf='+str(np.nansum(subsubTSMsw>1)/NopxFOVmsk)+' F='+str(Fsw[fov]))
                     #    plt.colorbar( orientation='horizontal')
                     #    ax3 = fig2.add_subplot(2, 2, 4)
                     #    imgplot3 = plt.imshow(subsubDFNC,cmap=mpl.colormaps['turbo'])
                     #    imgplot3.set_clim(0, 50)
                     #    plt.title('mean DFNC= '+str(np.nansum(subsubDFNC*FOVmsk)))
                     #    plt.colorbar( orientation='horizontal')
                      
                     #    plt.savefig('/export/home/eeytan/projects/Twilight_MODIS_CERES/figs' +'/'+ str(fov) + '.png')
                     #    plt.close(fig2)    
                  
                     fovisw=fovisw+1
                     
    
    # --------------------------------- Analyzing LW fov---------------------------------------------                 
                 if np.isnan(Flw[fov])==False:
                        
                     if subsubRf==DY-1:
                        subsubTSMlw=TSMlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subsubTWflw=TWfactorlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        subSumIlw=SumIlw[subsubRi:,subsubCi:subsubCf+1].copy()
                        if FswNAN:
                            subsubDFNC=subDFNC[subsubRi:,subsubCi:subsubCf+1].copy()
                            thermalfov=subThermal[:,subsubRi:,subsubCi:subsubCf+1].copy()
                            solarfov=subSolar[:,subsubRi:,subsubCi:subsubCf+1].copy()
                     elif subsubCf==DX-1:
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:].copy() 
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:].copy()
                         subSumIlw=SumIlw[subsubRi:subsubRf+1,subsubCi:].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:].copy()
                     else:    
                         subsubTSMlw=TSMlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subsubTWflw=TWfactorlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         subSumIlw=SumIlw[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                         if FswNAN:
                             subsubDFNC=subDFNC[subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             thermalfov=subThermal[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                             solarfov=subSolar[:,subsubRi:subsubRf+1,subsubCi:subsubCf+1].copy()
                     
                     
                     
                     if PSFf:
                         tmp=np.nonzero(np.isnan(subsubTSMlw))
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         if np.sum(FOVmsk[tmpy,tmpx])>CERESnanprec:  
                             continue
                         lw_meanTSM[totpiece,fovilw]=np.nansum(subsubTSMlw*FOVmsk)
                         lw_meanRHO[totpiece,fovilw]=np.nansum(subsubTWflw*FOVmsk)
                         lw_meanSumI[totpiece,fovilw]=np.nansum(subSumIlw*FOVmsk)
                         lw_meanDFNC[totpiece,fovilw]=np.sum(subsubDFNC*FOVmsk) 
                         tmp=np.nonzero(subsubTSMlw>1)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         lw_TSM1f[totpiece,fovilw]=np.nansum(subsubTSMlw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMlw*FOVmsk)
                         tmp=np.nonzero(subsubTSMlw>3)  # find where TSM>1 and calc. its contribution to weighted average TSM
                         tmpx=tmp[1]
                         tmpy=tmp[0]
                         lw_TSM2f[totpiece,fovilw]=np.nansum(subsubTSMlw[tmpy,tmpx]*FOVmsk[tmpy,tmpx])/np.nansum(subsubTSMlw*FOVmsk)
                         
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subSolar)[0],1,1))
                         lw_meanSolarspec[totpiece,fovilw,:]=np.nansum(np.nansum(solarfov*tmp,axis=1),axis=1)
                         tmp=np.tile(FOVmsk[np.newaxis,: ,:], (np.shape(subThermal)[0],1,1))
                         lw_meanThermalspec[totpiece,fovilw,:]=np.nansum(np.nansum(thermalfov*tmp,axis=1),axis=1)
                     else:
                         if np.sum(np.isnan(subsubTSMlw[FOVmsk]))>0:
                             continue                              
                         subsubTSMlw[(~FOVmsk)]=np.nan 
                         subsubTWflw[(~FOVmsk)]=np.nan 
                         subSumIlw[(~FOVmsk)]=np.nan
                         lw_meanRHO[totpiece,fovilw]=np.nanmean(subsubTWflw)
                         lw_meanTSM[totpiece,fovilw]=np.nanmean(subsubTSMlw)
                         lw_meanSumI[totpiece,fovilw]=np.nanmean(subSumIlw)
                         if FswNAN:
                             subsubDFNC[(~FOVmsk)]=np.nan
                         lw_meanDFNC[totpiece,fovilw]=np.nanmean(subsubDFNC[FOVmsk])
                         TSM1sum=np.nansum(subsubTSMlw>1)
                         lw_TSM1f[totpiece,fovilw]=TSM1sum/ NopxFOVmsk
                         
                     CERESSZAlw[totpiece,fovilw]=CERESsza[fov]
                     CERESRAAlw[totpiece,fovilw]=CERESraa[fov]    
                     CERESVZAlw[totpiece,fovilw]=CERESvza[fov]

                     # lwCERES_MOD_name[totpiece,fovilw]=str(MODISday)+'_'+str(piece)+'_'+str(fovlat)+','+str(fovlon)                        
                     lw_F[totpiece,fovilw]=Flw[fov]  
                     lw_I[totpiece,fovilw]=Ilw[fov] 
                                                   
                     # lw_Fclrspec[totpiece,fovilw,:]=specclr[0,:]
                  #  lw_FclrspecSTD[totpiece,fovilw,:]=specclr[1,:] # IN CASE WE NEED STANDARD DEVIATION FOR ML
    
                     fovilw=fovilw+1  
         
 ##################################### Taking MODIS clear sky pixels for ML flux retrievals   ###################################                    
         # if fovisw>0:
         CloudDomain_i=CloudDomain_i+1
         # print('<><><><><><><><><><><><><> ',CloudDomain_i,totpiece,' <><><><><><><><><><><><><>' )
         MODclri=np.nonzero(TSMsw<TSMth)
         if np.size(MODclri)>100:   # analyze only cloud fields with more than 100 ckear-sky MODIS pixels
             tmp=np.concatenate((subSolar,subThermal),axis=0)
             esh=np.tile(subSolZen[np.newaxis,:,:],(3,1,1))
             esh[1,:,:,]=np.reshape(subSenZen,(1,np.shape(subSenZen)[0],np.shape(subSenZen)[1]))
             esh[2,:,:,]=np.reshape(subRAA,(1,np.shape(subRAA)[0],np.shape(subRAA)[1]))
             # print('# MODIS pixels',len(MODclri[0]))
             if len(MODclri[0])<=np.shape(MODclrsky_sw)[1]:
                for modpx in range(0,len(MODclri[0])):
                     MODclrsky_sw[CloudDomain_i,modpx,:]=tmp[:,MODclri[0][modpx],MODclri[1][modpx]] 
                     MODangles_sw[CloudDomain_i,modpx,:]=esh[:,MODclri[0][modpx],MODclri[1][modpx]] 
             else:
                #print('##### more MODsw pixels than array size',len(MODclri[0]) , '########')
                tmp0=MODclri[0].T
                tmp1=MODclri[1].T
                tmp2=np.tile(tmp0[:,np.newaxis],(1,2))
                tmp2[:,1]=tmp1
                np.random.shuffle(tmp2)
                np.random.shuffle(tmp2)
                for modpx in range(0,np.shape(MODclrsky_sw)[1]):
                    MODclrsky_sw[CloudDomain_i,modpx,:]=tmp[:,tmp2[modpx,0],tmp2[modpx,1]] 
                    MODangles_sw[CloudDomain_i,modpx,:]=esh[:,tmp2[modpx,0],tmp2[modpx,1]] 
                        # print('Too much pixels SW',modpx)
             if fovisw>3:
                CERES_CldDomain_isw[CloudDomain_i]=totpiece
         # if fovilw>0:
         MODclri=np.nonzero(TSMlw<TSMth)
         if np.size(MODclri)>100:
            tmp=np.concatenate((subSolar,subThermal),axis=0)
            esh=np.tile(subSolZen[np.newaxis,:,:],(3,1,1))
            esh[1,:,:,]=np.reshape(subSenZen,(1,np.shape(subSenZen)[0],np.shape(subSenZen)[1]))
            esh[2,:,:,]=np.reshape(subRAA,(1,np.shape(subRAA)[0],np.shape(subRAA)[1]))
            if len(MODclri[0])<=np.shape(MODclrsky_lw)[1]:
                for modpx in range(0,len(MODclri[0])):
                    MODclrsky_lw[CloudDomain_i,modpx,:]=tmp[:,MODclri[0][modpx],MODclri[1][modpx]] 
                    MODangles_lw[CloudDomain_i,modpx,:]=esh[:,MODclri[0][modpx],MODclri[1][modpx]] 
            else:
                # print('##### more MODlw pixels than array size',len(MODclri[0]) , '########')
                tmp0=MODclri[0].T
                tmp1=MODclri[1].T
                tmp2=np.tile(tmp0[:,np.newaxis],(1,2))
                tmp2[:,1]=tmp1
                np.random.shuffle(tmp2)
                np.random.shuffle(tmp2)
                for modpx in range(0,np.shape(MODclrsky_lw)[1]):
                    MODclrsky_lw[CloudDomain_i,modpx,:]=tmp[:,tmp2[modpx,0],tmp2[modpx,1]]
                    MODangles_lw[CloudDomain_i,modpx,:]=esh[:,tmp2[modpx,0],tmp2[modpx,1]]
            if fovilw>3:
                CERES_CldDomain_ilw[CloudDomain_i]=totpiece
         
         foviswmax=np.maximum(foviswmax,fovisw)
         fovilwmax=np.maximum(fovilwmax,fovilw)
         
         #------------ added ORG ------------
         if Nfovssw>0:
             Fall_sw[totpiece]=intFsw/Nfovssw
         if Nfovslw>0:
             Fall_lw[totpiece]=intFlw/Nfovslw
         
         # TotArea=DX*DY  #!!! should I consider NaNs in fraction calc.
         TotArea=np.nansum(np.invert(unwanted[ri:ri+DY,ci:ci+DX]))
         PCSf_sw[totpiece]=np.nansum(TSMsw<1)/TotArea # pure clear-sky fraction; such that CF+PCSf+TWf=1
         PCSf_lw[totpiece]=np.nansum(TSMlw<1)/TotArea
         CSfNC_sw=PCSf_sw[totpiece]/(1-CF)  # pure-clear-sky fraction of all non-cloudy area such that Nnc=Npcs+Ntw
         CSfNC_lw=PCSf_lw[totpiece]/(1-CF)
         
         
         #------------- sampeling images ----------------------
         
         if np.random.randint(0, 100)<=Sampling_prob:
             
             ImSamp[:,:,Imkeepi,0]=subSolar[1,:,:]
             ImSamp[:,:,Imkeepi,1]=subThermal[8,:,:]
             ImSamp_metrics[Imkeepi,:]=np.array([maxDFNC[totpiece],meanDFNC[totpiece], maxCLDa[totpiece],meanCLDa[totpiece], meanCTH[totpiece],CTH95[totpiece], meanCOD[totpiece],COD95[totpiece], Iorg[totpiece],KScld[totpiece],KSclr[totpiece], Fall_sw[totpiece],Fall_lw[totpiece],ImLat[totpiece],MODISday, CloudFraction[totpiece],stdCTH[totpiece],ThinCldf[totpiece],COD25[totpiece],COD50[totpiece]])
             Imkeepi=Imkeepi+1
         # maxDFNCths=[0,15,25,40]
         # maxCLDths=[0,8,13,20]
         
         # Imsamp_counts=np.zeros(4,4)
         # Imsamp_Nsamp=np.array([[5,5,5,5],[5,10,10,5],[5,20,20,5],[5,5,5,5]])
        
         # # SampIms=[0]*4*4*np.max(Imsamp_Nsamp-1)
         # SampIms= [[0 for _ in range(len(maxCLDths))] for _ in range(len(maxDFNCths))]
         
         # if np.random.randint(0, 100)>33:
             
         #     closeDFNC=findmindiff(maxDFNC[totpiece],maxDFNCths)[0][0]
         #     if maxDFNC[totpiece]>=maxDFNCths[closeDFNC]:
         #         DFNCi=closeDFNC #; DFNCiH=closeDFNC+1
         #     else:
         #         DFNCi=closeDFNC-1 #; DFNCiH=closeDFNC
                 
         #     closeCLD=findmindiff(maxCLDa[totpiece],maxCLDths)[0][0]
         #     if maxCLDa[totpiece]>=maxCLDths[closeCLD]:
         #         CLDi=closeCLD #; CLDiH=closeCLD+1
         #     else:
         #         CLDi=closeCLD-1 #; CLDiH=closeCLD
                 
         #     if Imsamp_Nsamp[DFNCi,CLDi]<Imsamp_counts[DFNCi,CLDi]:
                
         #          Imsamp_counts[DFNCi,CLDi]=Imsamp_counts[DFNCi,CLDi]+1  
             
         
     # print('%%%%%%%%%%%%%%%%%%%%%%%%',fovisw,fovilw)
     

     
     # swallnanraw=np.nonzero(np.nansum(np.isnan(sw_F),axis=1)<np.shape(sw_F)[1])[0]                  
     # lwallnanraw=np.nonzero(np.nansum(np.isnan(lw_F),axis=1)<np.shape(lw_F)[1])[0] 
     # for ii in range(0,len(CERES_CldDomain_isw)):
     #     if ~np.isnan(CERES_CldDomain_isw[ii]):
     #         if ~(swallnanraw==CERES_CldDomain_isw[ii]).any():
     #             CERES_CldDomain_isw[ii]=np.nan
                 
     # for ii in range(0,len(CERES_CldDomain_ilw)):
     #     if ~np.isnan(CERES_CldDomain_ilw[ii]):
     #         if ~(lwallnanraw==CERES_CldDomain_ilw[ii]).any():
     #             CERES_CldDomain_ilw[ii]=np.nan
     
     ImSamp=ImSamp[:,:,:Imkeepi,:]
     ImSamp_metrics=ImSamp_metrics[:Imkeepi,:]
     
     swallnanraw=np.nonzero(np.nansum(np.isnan(MODclrsky_sw[:,:,1]),axis=1)<np.shape(MODclrsky_sw[:,:,1])[1])[0]      # TAKE ONLY CLOUD FIELDS THAT HAS AT LEAST 1 MODIS CLRSKY PIXEL
     lwallnanraw=swallnanraw.copy()  #np.nonzero(np.nansum(np.isnan(MODclrsky_lw[:,:,8]),axis=1)<np.shape(MODclrsky_lw[:,:,8])[1])[0]
     print('<><><><><><><><><><><>  swNi cloud fields',swallnanraw,'<><><><><><><><><><><>')
     print('<><><><><><><><><><><>  lwNi cloud fields',lwallnanraw,'<><><><><><><><><><><>')
     
    
     if np.size(swallnanraw)+np.size(lwallnanraw)==0:
         # print('nothing_found')
         return('nothing_found')
     else:
         SWdic = {"F": sw_F[swallnanraw,:], "I": sw_I[swallnanraw,:], "TSM1f": sw_TSM1f[swallnanraw,:], "TSM2f": sw_TSM2f[swallnanraw,:], "mRHO":sw_meanRHO[swallnanraw,:],
                  "mDFNC": sw_meanDFNC[swallnanraw,:], "mTSM": sw_meanTSM[swallnanraw,:], 
                  "mSumI":sw_meanSumI[swallnanraw,:], "VZA":CERESVZAsw[swallnanraw,:], "SZA":CERESSZAsw[swallnanraw,:], "RAA":CERESRAAsw[swallnanraw,:],
                  "CF": CloudFraction[swallnanraw], "swDomEPS": swDom_mean_eps[swallnanraw],
                  "AOD": sw_AODfov[swallnanraw,:], "Clrf": sw_Clrfov[swallnanraw,:], "SST": sw_SSTfov[swallnanraw,:], "PW": sw_PWfov[swallnanraw,:],  "DTz": sw_DTzfov[swallnanraw,:], #"EIS": sw_EISfov[swallnanraw,:], 
                  "SolarCLRspec": sw_meanSolarspec[swallnanraw,:,:],"ThermalCLRspec": sw_meanThermalspec[swallnanraw,:,:],
                  "MODCLRspec": MODclrsky_sw[swallnanraw,:,:], "MODangles":MODangles_sw[swallnanraw,:,:], "MODclrfovi":CERES_CldDomain_isw[swallnanraw],  
                  "Fall":Fall_sw[swallnanraw],"date":ImDate[swallnanraw],"Lon":ImLon[swallnanraw],"Lat":ImLat[swallnanraw],
                  "PCSf":PCSf_sw[swallnanraw],"Iorg":Iorg[swallnanraw],"KScld":KScld[swallnanraw],"KSclr":KSclr[swallnanraw], "maxDFNC":maxDFNC[swallnanraw], "meanDFNC":meanDFNC[swallnanraw], "maxCLDa":maxCLDa[swallnanraw], "meanCLDa":meanCLDa[swallnanraw], "Perimeter":Perimeter[swallnanraw], "stdCTH":stdCTH[swallnanraw],"meanCTH":meanCTH[swallnanraw], "CTH95":CTH95[swallnanraw], 
                  "COD95":COD95[swallnanraw], "meanCOD":meanCOD[swallnanraw], "COD50":COD50[swallnanraw], "COD25":COD25[swallnanraw], "ThinCldf":ThinCldf[swallnanraw],
                  "CLD_BTstd":stdBTlw_cld[swallnanraw,:], "CLDminBT":minBTlw_cld[swallnanraw,:], "CLDmeanRef":meanRefsw_cld[swallnanraw,:],"CLDmaxRef":maxRefsw_cld[swallnanraw,:], "CLRmeanBT":meanBTlw_clr[swallnanraw,:], "CLR1112std":stdBT1112_clr[swallnanraw], "CLR1112mean":meanBT1112_clr[swallnanraw], 
                  "ImSamp":ImSamp,"ImSamp_metrics": ImSamp_metrics,
                     } 
          
         LWdic = {"F": lw_F[lwallnanraw,:], "I": lw_I[lwallnanraw,:], "TSM1f": lw_TSM1f[lwallnanraw,:], "TSM2f": lw_TSM2f[lwallnanraw,:], "mRHO": lw_meanRHO[lwallnanraw,:],
                  "mDFNC": lw_meanDFNC[lwallnanraw,:], "mTSM": lw_meanTSM[lwallnanraw,:], "mSumI":lw_meanSumI[lwallnanraw,:], "SZA":CERESSZAlw[lwallnanraw,:], 
                  "VZA":CERESVZAlw[lwallnanraw,:], "RAA":CERESRAAlw[lwallnanraw,:],"CF": CloudFraction[lwallnanraw],"lwDomEPS": lwDom_mean_eps[lwallnanraw],
                  "SolarCLRspec": lw_meanSolarspec[lwallnanraw,:,:],"ThermalCLRspec": lw_meanThermalspec[lwallnanraw,:,:],"MODCLRspec": MODclrsky_lw[lwallnanraw,:,:], 
                  "MODangles":MODangles_lw[lwallnanraw,:,:],"MODclrfovi":CERES_CldDomain_ilw[lwallnanraw],
                  "Fall":Fall_lw[lwallnanraw],"date":ImDate[lwallnanraw],"Lon":ImLon[lwallnanraw],"Lat":ImLat[lwallnanraw],
                  "PCSf":PCSf_lw[lwallnanraw],"Iorg":Iorg[lwallnanraw],"KScld":KScld[lwallnanraw],"KSclr":KSclr[lwallnanraw],"maxDFNC":maxDFNC[lwallnanraw], "meanDFNC":meanDFNC[lwallnanraw],"maxCLDa":maxCLDa[lwallnanraw], "meanCLDa":meanCLDa[lwallnanraw], "Perimeter":Perimeter[lwallnanraw], "stdCTH":stdCTH[lwallnanraw],"meanCTH":meanCTH[lwallnanraw], "CTH95":CTH95[lwallnanraw], 
                  "COD95":COD95[lwallnanraw], "meanCOD":meanCOD[lwallnanraw], "COD50":COD50[lwallnanraw], "COD25":COD25[lwallnanraw], "ThinCldf":ThinCldf[lwallnanraw],
                  "CLD_BTstd":stdBTlw_cld[lwallnanraw,:], "CLDminBT":minBTlw_cld[lwallnanraw,:], "CLDmeanRef":meanRefsw_cld[lwallnanraw,:],"CLDmaxRef":maxRefsw_cld[lwallnanraw,:], "CLRmeanBT":meanBTlw_clr[lwallnanraw,:], "CLR1112std":stdBT1112_clr[lwallnanraw], "CLR1112mean":meanBT1112_clr[lwallnanraw], #"ImSamp":tmpsamp[lwallnanraw],  
                    }
         return(SWdic,LWdic) #,Pieceprop)











