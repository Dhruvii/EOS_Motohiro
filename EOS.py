print("code started")



# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:02:41 2022

@author: Dhruv
"""

#purpose of program: Find the initial conditions that lead to physically valid solutions.

#1- using optimize.minimize and optimize.root

#2- initial conditons chosen randomly from a set-
#        sigma= (0,93) (since at Saturation, sigma=93MeV)
#        omega= (0,50)                
#          rho= (-50,50)
#          MuQ= (-100,0)
#
#3- if solution is in the physically valid region, it is saved in a file for further analysis.
#        sigma ~ 50
#        omega ~ 10               
#          rho ~ -10
#          MuQ ~ -50
#
#=============================================================================================


import math
from random import random
import numpy as np
#import matplotlib.pyplot as plt
from scipy import optimize
import copy
import logging

pi = math.pi ; hbarc = 197.3269788; ln= np.log


#physical values
fp = 92.2; mpi = 140.; mN = 939.; mNp = 1535.; mw = 783. ; mr = 775.; me=0.511 ; mm=105.6; mu_0=923

#parameter sets
m0=700.0 ;g1 = 7.829574386331634 ;g2 = 14.293782629281743;mbar2= 19.461938600144112 * fp**2 ;lam = 35.7773844533009 ;lam6 = 14.00979135313087/fp**2 ;gw = 7.302392457542381;gp = 8.11996858169054;

#m0=600.0;g1 = 8.500;g2 = 14.964;mbar2 = 22.556*fp**2;lam = 40.7456 ;lam6 = 15.8835*fp**-2 ;gw  = 9.1271;gp = 7.845;
#m0=550.0;g1 = 8.7852 ;g2 = 15.249;mbar2 = 22.753*fp**2 ;lam = 41.2997 ;lam6 = 16.2404*fp**-2 ;gw = 10.2135;gp = 7.6173;
#m0=500;g1=9.02;g2=15.5;mbar2=22.9*fp**2; lam=42.3; lam6=16.9/fp**2; gw=11.3; gp=7.3

#for convinience
G=g1+g2;g=g1-g2 # for reference, G>0 g<0 

mub=mu_0 #i dont know why

#===============================================================================
#functions to define mass:

def M(s):  # real positive
    if s<0:
        logging.info("sigma is negative")
    return np.sqrt((G*s)**2 + 4* m0**2 )

def m(s,sign):  #function to calculate in medium mass of nucleon, in: MeV out: MeV

    ans= 0.5*((M(s)+ sign*g*s))

    if ans<0:
        logging.critical("negative mass found")

    return ans

#===============================================================================

def kf(m,mu): #function to calculate fermi momentum, in: MeV,MeV out: MeV

    if (mu**2-m**2)>0:
        return np.sqrt(mu**2-m**2) 
    else:
        return 0

#===============================================================================

#===============================================================================
#New!
def pos( m,mu ): # pos:=(k+mu/m), unphysical errors caused due to "zero mass" and "negative Mu" are removed

    k=kf(m,mu)
    
    if m==0:
        return 1 #since formulas are of the form m**n ln (pos) , when m = 0 the whole term is zero
        
    pos= (k+mu)/m
    
    if pos<0: #negative value in log is due to Mu<0, which case, there should be no contribution to  Pfg
        logging.info ("negative value found in log (returned 0)")
        pos=1

    return pos

#===============================================================================

def Pfg(m,mu): #function to calculate pressure of fermi gas, in: MeV,MeV out: MeV

    k=kf(m,mu)

    if k>0:
        return (2/3 * k**3 * mu - m**2 * mu * k + m**4 *ln(pos(m,mu)) )/(8*pi**2)
    else:
        return 0

#===============================================================================

#===============================================================================

def dmds(s,sign): #sign = 1,-1 # Function to calculate dm/ds, in: MeV/fm, out: MeV/fm

      return 0.5*( G**2*s / M(s) + sign* g) #verified #when s>0, always valid
#===============================================================================

#===============================================================================

def dpdm(m,mu): #Function to calculate dp/dm, in: MeV,MeV out:
 k=kf(m,mu)
 
 if k>0:
     return (-4*m*mu*k + 4 *m**3* ln(pos(m,mu))/(8*pi**2))
 else:  #maybe this isnt needed
     return 0 

#===============================================================================

#===============================================================================

def dpdmu (m,mu): #calculate density i.e derivate of Pressure (of fermigas) wrt chemical potential 
    k=kf(m,mu)
    return (kf(m,mu)**3)/(3*pi**2)

def densityfg(m,mu):
    return (dpdmu(m,mu))

#===============================================================================

#===============================================================================
def getp(mub,s,w,r,muq): #calculates Pressure from mean feilds and chemical potentials

    muP = mub + muq - gw*w - 0.5*gp*r
    muN = mub       - gw*w + 0.5*gp*r
    muL = -muq # from charge conservation

    mp = 0.5 * (M(s)+g*s)
    mn = 0.5 * (M(s)-g*s)  # we see mn>mp only when s<0  

    P_fg = Pfg(mp,muP) + Pfg(mn,muP) + Pfg(mp,muN) + Pfg(mn,muN) #Fermi-Gas type pressure term from baryons 
    P_phi= 0.5* (mw*w)**2 + 0.5* (mr*r)**2+ 0.5* mbar2*s*s - 0.25*lam*s**4 + lam6/6 * s**6 +mpi**2*fp*s #pressure due to meson interactions 
    P_l  = Pfg(me,muL) + Pfg(mm,muL) #Contribution from electron and muon

    P= P_fg+ P_phi + P_l #total pressure is the sum of contributions

    return(P)
#===============================================================================

#===============================================================================
def getbden(mub,s,w,r,muq): #function to calculate baryon density from mean feilds and chemical potentials

    mp = 0.5 * ((((G*s)**2)+4*m0**2)**0.5+g*s)
    mn = 0.5 * ((((G*s)**2)+4*m0**2)**0.5-g*s)

    mup=mub+muq-gw*w-0.5*gp*r
    mun=mub-gw*w+0.5*gp*r

    return( dpdmu(mp,mup)+dpdmu(mn,mup)+dpdmu(mp,mun)+dpdmu(mn,mun) )
#===============================================================================

#To simplify formulas: 
#===============================================================================
def dVds(s):
    return mbar2*s-lam*s**3+lam6*s**5+mpi**2*fp
def dVdw(w):
    return mw**2 * w
def dVdr(r):
    return mr**2 * r
#===============================================================================

#self consistency and charge neutrality conditions:
#===============================================================================
def dPds(var):
    s=var[0]
    w=var[1]
    r=var[2]
    muq=var[3]
    #in-medium Mass of positive and negative parity nucleons
    mp = m(s,1)
    mn = m(s,-1)
    #Effective Chemical potential for proton and neutron
    mup = (mub+ 0.5* muq - gw * w + 0.5 * (muq - gp*r)) 
    mun = (mub+ 0.5* muq - gw * w - 0.5 * (muq - gp*r)) 

    return dVds(s) + dpdm(mp,mup)*dmds(s,1) + dpdm(mp,mun)*dmds(s,1) + dpdm(mn,mup)*dmds(s,-1) + dpdm(mn,mun)*dmds(s,-1)

def dPdw(var):
    s=var[0]
    w=var[1]
    r=var[2]
    muq=var[3]
    #in-medium Mass of positive and negative parity nucleons
    mp = m(s,1)
    mn = m(s,-1)
    #Effective Chemical potential for proton and neutron
    mup = (mub+ 0.5* muq - gw * w + 0.5 * (muq - gp*r)) 
    mun = (mub+ 0.5* muq - gw * w - 0.5 * (muq - gp*r)) 

    return dVdw(w)  - gw * ( dpdmu(mp,mup)+dpdmu(mp,mun)+dpdmu(mn,mup)+dpdmu(mn,mun)) #dmupdw=dmundw=-gw

def dPdr(var):
    s=var[0]
    w=var[1]
    r=var[2]
    muq=var[3]
    #in-medium Mass of positive and negative parity nucleons
    mp = m(s,1)
    mn = m(s,-1)
    #Effective Chemical potential for proton and neutron
    mup = (mub+ 0.5* muq - gw * w + 0.5 * (muq - gp*r)) 
    mun = (mub+ 0.5* muq - gw * w - 0.5 * (muq - gp*r)) 

    return dVdr(r) - 0.5* gp* (dpdmu(mp,mup)-dpdmu(mp,mun)+dpdmu(mn,mup)-dpdmu(mn,mun))

def dPdmuq(var):
    s=var[0]
    w=var[1]
    r=var[2]
    muq=var[3]
    #in-medium Mass of positive and negative parity nucleons
    mp = m(s,1)
    mn = m(s,-1)
    #Effective Chemical potential for proton and neutron
    mup = (mub+ 0.5* muq - gw * w + 0.5 * (muq - gp*r)) 
    mun = (mub+ 0.5* muq - gw * w - 0.5 * (muq - gp*r)) 

    return dpdmu(mp,mup)+ dpdmu(mn,mup) + densityfg(me,muq) + densityfg(mm,muq)
#===============================================================================
    
#function to be solved numerically
#===============================================================================
def myFunction (var): 
    F=np.zeros(4)
    F[0]= dPds(var)
    F[1]= dPdw(var)
    F[2]= dPdr(var)
    F[3]= dPdmuq(var)
    return F
#===============================================================================

#===============================================================================
def getrand (): #get random numbers to use as initial guess for solution:
   R=np.empty(4)
   R[0]=random()*100
   R[1]=random()*50
   R[2]=-50+random()*100
   R[3]=random()*-100
   return R
#===============================================================================

def getguess(N): #if N goes from 1 to 46,000,000 we will check each value once (wasteful but whatever)
    R=np.empty(4)
    R[0]=92- (N % 92) # sigma goes from 1 to 100 MeV
    N/=92
    R[1]= N%50
    N/=50
    R[2]=(N%100)-50
    N/=100
    R[3]=(N%100)-100
    return R
#===============================================================================

# SOLVING THE SYSTEM OF EQUATIONS

#===============================================================================

error=0.001 #error to confirm numerical solution is actially a solution to the system
var = np.zeros(4) #array where meanfields and chemical potentials are stored var=(s,w,r,muq)

#P=np.empty(0)
#BDEN=np.empty(0)

N= 46000000 #N is the number of times the loop runs,  take ~10 days to complete  

f = open(r"C:\Users\hken\Desktop\data\datasys1.txt", "a") #file to save solutions of type 1
f2 = open(r"C:\Users\hken\Desktop\data\datasys2.txt", "a") #file to save solutions of type 2

f.write(f"Computing: N={N}\n")
for i in range (N): 
 Guess = getguess(N)  #initial guess is a random number
 G0=copy.deepcopy(Guess)
 
 MUB=np.empty(0)    #numpy arrays to store values of fields
 SIGMA=np.empty(0)
 W=np.empty(0)
 RHO=np.empty(0)
 MUQ=np.empty(0)
 
 for t in range (100): #find sigma(muB), omega(muB), rho(muB), Muq(muB) by solving F using Guess.
    mub= mu_0 + 5 + t
    #sol = optimize.fsolve(myFunction,Guess) #solves system of equations using fsolve
    sol = optimize.root(myFunction,Guess) #solves system of equations using root
    #solve system of equations using minimize     
    d=np.linalg.norm(myFunction(sol.x)) # difference between sol and true solution
    flag=0
    if (d<error and sol.x[0]>0 and sol.x[3]>-100): #selects numerically valid solutions, with non negative sigma and charge chemical potential bigger than -100MeV to ensure validity of model
        if (sol.x[1]>0.00001 and sol.x[2]>0.00001 or sol.x[2]<-0.00001 ): # selects solutions with non-trivial values of omega and rho feilds   
            flag=1
            MUB=np.append(MUB,mub)
            SIGMA=np.append(SIGMA,sol.x[0])
            W=np.append(W,sol.x[1])
            RHO=np.append(RHO,sol.x[2])
            MUQ=np.append(MUQ,sol.x[3]) 
            Guess = np.array(sol.x) #next guess (may remove later)
    else:
        break

if flag==1: #write to file
    f2.write(f"flag:{flag}\n")
    f2.write(f"guess:{G0}\n")
    f2.write(f"{MUB} {SIGMA} {W} {RHO} {MUQ}\n")
    del MUB #deletes array to save memory for next solution
    del SIGMA
    del W
    del RHO
    del MUQ
#===============================================================================



print("code excuted")