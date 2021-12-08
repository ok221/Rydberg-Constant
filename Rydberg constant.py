# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:42:16 2021

@author: Olivia Keene
"""

import numpy as np
import matplotlib.pyplot as plt

thetad,m=(np.loadtxt('Data/Rydberg Constant.txt',unpack=True))
theta=thetad*np.pi/180
uncertainties=((np.loadtxt('Data/Theta uncertainties.txt'))/60)*np.pi/180
u_p=uncertainties[0:7]
u_c=uncertainties[7:25]
u_m=uncertainties[25:]
theta_p=theta[0:7]
theta_c=theta[7:25]
theta_m=theta[25:]
sinp_u=np.sqrt((np.cos(theta_p))**2*u_p**2)
sinc_u=np.sqrt((np.cos(theta_c))**2*u_c**2)
sinm_u=np.sqrt((np.cos(theta_m))**2*u_m**2)
m_p=m[0:7]
m_c=m[7:25]
m_m=m[25:]
sintheta_p=np.sin(theta_p)
sintheta_c=np.sin(theta_c)
sintheta_m=np.sin(theta_m)
plt.errorbar(m_p,sintheta_p,xerr=None,yerr=sinp_u,fmt='g.',mew=1.2, ms=3, capsize=3)
fit_p,cov_p = np.polyfit(m_p,sintheta_p,1,w=1/sinp_u,cov=True)
line_p=np.poly1d(fit_p)
x=np.linspace(-3.0,3.0,100)
plt.plot(x,line_p(x),'m-')
plt.grid()
plt.xlabel("m (no units)", fontsize=14) #labels x-axis
plt.ylabel("sin(\u03B8) (no units)", fontsize=14) #labels y-axis
plt.title("Plot to find wavelength of purple", fontsize=16)
plt.savefig("Purple")
plt.show()
#%%
plt.errorbar(m_c,sintheta_c,xerr=None,yerr=sinc_u,fmt='g.',mew=1.2, ms=3, capsize=3)
fit_c,cov_c = np.polyfit(m_c,sintheta_c,1,w=1/sinc_u,cov=True)
line_c=np.poly1d(fit_c)
y=np.linspace(-8,9,100)
plt.plot(y,line_c(y),'b-')
plt.grid()
plt.xlabel("m (no units)", fontsize=14) #labels x-axis
plt.ylabel("sin(\u03B8) (no units)", fontsize=14) #labels y-axis
plt.title("Plot to find wavelength of cyan", fontsize=16)
plt.savefig("Cyan")
plt.show()
#%%
plt.errorbar(m_m,sintheta_m,xerr=None,yerr=sinm_u,fmt='g.',mew=1.2, ms=3, capsize=3)
fit_m,cov_m = np.polyfit(m_m,sintheta_m,1,w=1/sinm_u,cov=True)
line_m=np.poly1d(fit_m)
z=np.linspace(-5,7,100)
plt.plot(z,line_m(z),'r-')
plt.grid()
plt.xlabel("m (no units)", fontsize=14) #labels x-axis
plt.ylabel("sin(\u03B8) (no units)", fontsize=14) #labels y-axis
plt.title("Plot to find wavelength of magenta", fontsize=16)
plt.savefig("Magenta")
plt.show()
#%%
w_p=0.001/80*fit_p[0]
w_c=0.001/80*fit_c[0]
w_m=0.001/80*fit_m[0]
wu_p=0.001/80*np.sqrt(cov_p[0,0])
wu_c=0.001/80*np.sqrt(cov_c[0,0])
wu_m=0.001/80*np.sqrt(cov_m[0,0])
print("Purple wavelength is ",w_p," +/- ",wu_p," m")
print("Cyan wavelength is ",w_c," +/- ",wu_c," m")
print("Magenta wavelength is ",w_m," +/- ",wu_m," m")
#%%
x_p=1/5**2-1/2**2
x_c=1/4**2-1/2**2
x_m=1/3**2-1/2**2
wu=np.array([wu_p,wu_c,wu_m])
w=np.array([w_p,w_c,w_m])
x=np.array([x_p,x_c,x_m])
inversewu=np.sqrt((1/w**2)**2*wu**2)
plt.errorbar(-np.array(x),1/np.array(w),xerr=None,yerr=inversewu,fmt='m.',mew=1.2, ms=3, capsize=3)
fit,cov=np.polyfit(-np.array(x),1/np.array(w),1,w=1/inversewu,cov=True)
line=np.poly1d(fit)
points=np.linspace(-x_m,-x_p,1000)
plt.plot(points,line(points),'g-')
plt.grid()
plt.xlabel("1/p$^2$-1/n$^2$ (no units)", fontsize=12) #labels x-axis
plt.ylabel("1/wavelength (m$^-$$^1$)", fontsize=12) #labels y-axis
plt.title("Plot to find the Rydberg constant", fontsize=14)
print("Rydberg constant is ",fit[0]," +/- ",np.sqrt(cov[1,1])," m^-1")
plt.savefig("Rydberg")
plt.show()
#%%
plt.errorbar(m_p,sintheta_p,xerr=None,yerr=sinp_u,fmt='g.',mew=1.2, ms=3, capsize=3)
x=np.linspace(-3.0,3.0,100)
plt.plot(x,line_p(x),'m-')
plt.grid()
plt.xlabel("m (no units)", fontsize=12) #labels x-axis
plt.ylabel("sin(\u03B8) (no units)", fontsize=12) #labels y-axis
plt.errorbar(m_c,sintheta_c,xerr=None,yerr=sinc_u,fmt='g.',mew=1.2, ms=3, capsize=3)
y=np.linspace(-8,9,100)
plt.plot(y,line_c(y),'b-')
plt.errorbar(m_m,sintheta_m,xerr=None,yerr=sinm_u,fmt='g.',mew=1.2, ms=3, capsize=3)
z=np.linspace(-5,7,100)
plt.plot(z,line_m(z),'r-')
plt.title("sin(\u03B8) - m plot for all colours", fontsize=14)
plt.savefig("All wavelength graphs")
plt.show()



