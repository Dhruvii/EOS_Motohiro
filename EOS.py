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
import random
import numpy as np
#import matplotlib.pyplot as plt
from scipy import optimize

pi = math.pi ; hbarc = 197.3269788; ln= np.log


#physical values
fp = 92.2; mpi = 140.; mN = 939.; mNp = 1535.; mw = 783. ; mr = 775.; me=0.511 ; mm=105.6; mu_0=923

#parameter sets
m0=700.0 ;g1 = 7.829574386331634 ;g2 = 14.293782629281743;mbar2= 19.461938600144112 * fp**2 ;lam = 35.7773844533009 ;lam6 = 14.00979135313087/fp**2 ;gw = 7.302392457542381;gp = 8.11996858169054;

#m0=600.0;g1 = 8.500;g2 = 14.964;mbar2 = 22.556*fp**2;lam = 40.7456 ;lam6 = 15.8835*fp**-2 ;gw  = 9.1271;gp = 7.845;
#m0=550.0;g1 = 8.7852 ;g2 = 15.249;mbar2 = 22.753*fp**2 ;lam = 41.2997 ;lam6 = 16.2404*fp**-2 ;gw = 10.2135;gp = 7.6173;
#m0=500;g1=9.02;g2=15.5;mbar2=22.9*fp**2; lam=42.3; lam6=16.9/fp**2; gw=11.3; gp=7.3

#for convinience
G=g1+g2;g=g1-g2;
#===============================================================================

def m(s,sign):  #function to calculate in medium mass of nucleon, in: MeV out: MeV

    if sign==+1:

        return 0.5 * ((((G*s)**2)+4*m0**2)**0.5+g*s)

    elif sign==-1:

        return 0.5 * ((((G*s)**2)+4*m0**2)**0.5-g*s)

#===============================================================================

#===============================================================================

def kf(m,mu): #function to calculate fermi momentum, in: MeV,MeV out: MeV

    if (mu**2-m**2)>0:

        k = (mu**2-m**2)**(0.5)

        return k

    else:

        return 0

#===============================================================================

#===============================================================================

def Pfg(m,mu): #function to calculate pressure of fermi gas, in: MeV,MeV out: MeV

    k=kf(m,mu)

    if k>0:

        return (2/3 * k**3 * mu - m**2 * mu * k + m**4 *ln ((mu+k)/m))/(8*pi**2)

    else:

        return 0

#===============================================================================

#===============================================================================

def dmds(s,sign): #sign = 1,-1 # Function to calculate dm/ds, in: MeV/fm, out: MeV/fm

    if sign==1:

      return 0.5*(((G*s)**2 + 4*m0**2)**(-0.5) * G**2*s + g)

    elif sign==-1:

      return 0.5*(((G*s)**2 + 4*m0**2)**(-0.5) * G**2*s - g)

#===============================================================================



#===============================================================================

def dpdm(m,mu): #Function to calculate dp/dm, in: MeV,MeV out:

 k=kf(m,mu)

 if k>0:

     return (-4*m*mu*k + 4 *m**3* ln((k+mu)/m)/(8*pi**2))

 else:

     return 0

#===============================================================================



#===============================================================================

def dpdmu (m,mu):

    k=kf(m,mu)

    if k>0:

        return (kf(m,mu)**3)/(3*pi**2)

    else:

        return 0
    
def densityfg(m,mu):

    return (dpdmu(m,mu))

#===============================================================================



#===============================================================================

def getp(mub,s,w,r,muq):

    muP=mub+muq-gw*w-0.5*gp*r
    muN=mub-gw*w+0.5*gp*r

    mp = 0.5 * ((((G*s)**2)+4*m0**2)**0.5+g*s)
    mn = 0.5 * ((((G*s)**2)+4*m0**2)**0.5-g*s)

    P_fg = Pfg(mp,muP)+Pfg(mn,muP)+Pfg(mp,muN)+Pfg(mn,muN)
    P_phi= 0.5* (mw*w)**2+ 0.5* (mr*r)**2+ 0.5* mbar2*s*s - 0.25*lam*s**4 + lam6/6 * s**6 +mpi**2*fp*s
    P_l= + Pfg(me,-muq) +Pfg(mm,-muq)

    P=P_fg+P_phi + P_l

    #print(P_fg,P_phi,P_l)
    return(P)
    
#===============================================================================

#===============================================================================

def getbden(mub,s,w,r,muq):

    mp = 0.5 * ((((G*s)**2)+4*m0**2)**0.5+g*s)
    mn = 0.5 * ((((G*s)**2)+4*m0**2)**0.5-g*s)

    mup=mub+muq-gw*w-0.5*gp*r
    mun=mub-gw*w+0.5*gp*r

    return( dpdmu(mp,mup)+dpdmu(mn,mup)+dpdmu(mp,mun)+dpdmu(mn,mun) )

#===============================================================================





#===============================================================================

def myFunction (var):

 s=var[0]
 w=var[1]
 p=var[2]
 muq=var[3]
# in-medium Mass of positive and negative parity nucleons
 mp = m(s,1)
 mn = m(s,-1)
#Chemical potential for positive and negative parity nucleons
 mup = (mub+ 0.5* muq - gw * w + 0.5 * (muq - gp*p))
 mun = (mub+ 0.5* muq - gw * w - 0.5 * (muq - gp*p))

 F= np.empty(4)
#self consistency for sigma
 F[0]= mbar2*s-lam*s**3+lam6*s**5+mpi**2*fp+dpdm(mp,mup)*dmds(s,1)+dpdm(mp,mun)*dmds(s,1)+dpdm(mn,mup)*dmds(s,-1)+dpdm(mn,mun)*dmds(s,-1)
 #self consistency for omega
 F[1]= mw**2 * w - gw* (dpdmu(mp,mup)+dpdmu(mp,mun)+dpdmu(mn,mup)+dpdmu(mn,mun))
 #self consistency for rho
 F[2]= mr**2* p - 0.5* gp* (dpdmu(mp,mup)-dpdmu(mp,mun)+dpdmu(mn,mup)-dpdmu(mn,mun))
 #charge neutrality + beta equilibrium
 F[3]= -densityfg(muq,me)- densityfg(muq,mm) + dpdmu(mp,mup)+dpdmu(mn,mup)
 return F
#===============================================================================

def getrand ():
   R=np.empty(4)
   R[0]=random()*92
   R[1]=random()*50
   R[2]=-50+random()*100
   R[3]=random()*-100
   return R
   
#===============================================================================


#===============================================================================


# SOLVING THE SYSTEM OF EQUATIONS

#===============================================================================


error=0.01 #error to confirm solution is actially a solution
var = np.array([0.0,0.0,0.0,0.0])

#Guess = np.array([30,8,-12,-5]) #m0=700
#Guess = np.array([60,8,-8,-10]) #m0=600
#Guess = np.array([20,18,-10,-200])


#P=np.empty(0)
#BDEN=np.empty(0)

N=10 #N is the number of times random values are tried
f = open("C:\Users\hken\Desktop\data\dat.txt", "a")
f.write("running file: N=",N)
for i in range (N): 
 Guess = getrand()  #initial guess is a random number
 flag  = 0          #starts as 0, becomes 1 when a good data set is obtained
 MUB=np.empty(0)
 SIGMA=np.empty(0)
 W=np.empty(0)
 RHO=np.empty(0)
 MUQ=np.empty(0)

 for t in range (100): #find sigma(muB), omega(muB), rho(muB), Muq(muB) by solving F using Guess.
  mub= mu_0 + t
  #sol = optimize.fsolve(myFunction,Guess) #solves system of equations using fsolve
  f.write("guess:",Guess)
  sol = optimize.root(myFunction,Guess)#(.x) #solves system of equations using root
  ##sol = optimize.minimize(myFunction,Guess) #solves system of equations using minimize     
  d=np.linalg.norm(myFunction(sol.x)) # difference between sol and true solution
  if (d<error and sol.x[0]>0 and sol.x[1]>0 and sol.x[3]>-100): #only selects 'good' solutions
          MUB=np.append(MUB,mub)
          SIGMA=np.append(SIGMA,sol.x[0])
          W=np.append(W,sol.x[1])
          RHO=np.append(RHO,sol.x[2])
          MUQ=np.append(MUQ,sol.x[3])
          flag=flag*1
          Guess = np.array(sol.x) #next guess (may remove)
  else:
          flag=flag*0
          f.write("flag:0 ")
 if flag==1:
    f.write("flag:1")
    f.write(MUB,SIGMA,W,RHO,MUQ)
    f.write("end.") 
 del MUB #delete array 
 del SIGMA
 del W
 del RHO
 del MUQ


#===============================================================================