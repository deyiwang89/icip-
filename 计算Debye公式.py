# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:49:37 2020

@author: deyiwang

计算黄海海水介电系数及趋肤深度

数据来源:
@article{doi:10.1002/2013JC009716,
author = {Gentemann, Chelle L.},
title = {Three way validation of MODIS and AMSR-E sea surface temperatures},
journal = {Journal of Geophysical Research: Oceans},
volume = {119},
number = {4},
pages = {2583-2598},
keywords = {AMSR-E, MODIS, SST, remote sensing, microwave, infrared},
doi = {10.1002/2013JC009716},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2013JC009716},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2013JC009716},
abstract = {Abstract The estimation of retrieval uncertainty and stability are essential for the accurate interpretation of data in scientific research, use in analyses, or numerical models. The primary uncertainty sources of satellite SST retrievals are due to errors in spacecraft navigation, sensor calibration, sensor noise, retrieval algorithms, and incomplete identification of corrupted retrievals. In this study, comparisons to in situ data are utilized to investigate retrieval accuracies of microwave (MW) SSTs from the Advanced Microwave Scanning Radiometer—Earth Observing System (AMSR-E) and infrared (IR) SSTs from the Moderate Resolution Imaging Spectroradiometer (MODIS). The highest quality MODIS data were averaged to 25 km for comparison. The in situ SSTs are used to determine dependencies on environmental parameters, evaluate the identification of erroneous retrievals, and examine biases and standard deviations (STD) for each of the satellite SST data sets. Errors were identified in both the MW and IR SST data sets: (1) at low atmospheric water vapor a posthoc correction added to AMSR-E was incorrectly applied and (2) there is significant cloud contamination of nighttime MODIS retrievals at SST <10°C. A correction is suggested for AMSR-E SSTs that will remove the vapor dependency. For MODIS, once the cloud contaminated data were excluded, errors were reduced but not eliminated. Biases were found to be −0.05°C and −0.13°C and standard deviations to be 0.48°C and 0.58°C for AMSR-E and MODIS, respectively. Using a three-way error analysis, individual standard deviations were determined to be 0.20°C (in situ), 0.28°C (AMSR-E), and 0.38°C (MODIS).},
year = {2014}
}
"""
import math

def eypsilon_1(S,T):
	x = (87.134-0.1949*T-0.01276*T*T+0.002491*T*T*T)*(1+1.613*10**(-5)*T*S-0.003656*S+3.21*10**(-5)*S**2-4.232*10**(-7)*S*S*S)
	return x

def tao(S,T):
    x = (1.768 * 10**(-11)-6.086 * 10**(-13)*T+1.104 * 10**(-14)*T**(2)-8.111 * 10**(-17)*T**(3))*(1.0+2.282 * 10**(-5)*T*S-7.638 * 10**(-4)*S-7.760 * 10**(-6)*S**(2)+1.105 * 10**(-8)*S**(3))
    return x

def sigma(S,T):
    sigma = S*(0.182521-0.00146192*S+2.09324*10**(-5)*S**(2)-1.28205*10**(-7)*S**(3))*math.exp((T-25)*(0.02033+0.0001266*(25-T)+2.464 * 10**(-6)*(25-T)**(2)-S*(1.849 * 10**(-5)-2.551 * 10**(-7)*(25-T)+2.551 * 10**(-8)*(25-T)**(2))))
    return sigma

def varepsilon(S,T,f):
    eypsilon_0 = 8.854*10**(-12)
    omega = 2*math.pi*f*10**(9)
    kk = 0 + 1j
    xy = 4.9 + (eypsilon_1(S,T)-4.9)/(1-(omega*tao(S,T)*kk))-kk*(sigma(S,T))/(omega*eypsilon_0)    
    return xy

def delta(S,T,f):
    omega = 2*math.pi*f*10**(9)
    return (2*omega/299792458*(varepsilon(S,T,f)**(0.5)).imag)**(-1)

if __name__=="__main__":
    f = 5.4
    
    max_r = 70
    min_r = 70
    max_i = 12
    min_i = 12
    
    max_s = 0
    max_t = 0
    min_s = 0
    min_t = 0
    
    for S in range(30,33):
        for T in range(6,26):
            real = varepsilon(S,T,f).real
            imag = varepsilon(S,T,f).imag
            if imag>max_i:
                max_i = imag
                max_s = S
                max_t = T
            elif imag<min_i:
                min_i = imag
                min_s = S
                min_t = T
                
                
            
            
    print(max_s,"-",max_t,":",varepsilon(max_s,max_t,f))
    print(min_s,"-",min_t,":",varepsilon(min_s,min_t,f))
#    maxmax = varepsilon(30,25,f)
#    minmin = varepsilon(32,6,f)
    print(delta(max_s,max_t,f))
    print(delta(min_s,min_t,f))