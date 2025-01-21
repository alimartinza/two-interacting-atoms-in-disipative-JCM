from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import time
import os
import tkinter as tk
import pandas as pd

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
n=tensor(qeye(2),qeye(2),num(3))
sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,4])))
a=tensor(qeye(2),qeye(2),destroy(3))
sm1=tensor(sigmam(),qeye(2),qeye(3))
sp1=tensor(sigmap(),qeye(2),qeye(3))
sz1=tensor(sigmaz(),qeye(2),qeye(3))
sx1=tensor(sigmax(),qeye(2),qeye(3))
sm2=tensor(qeye(2),sigmam(),qeye(3))
sp2=tensor(qeye(2),sigmap(),qeye(3))
sz2=tensor(qeye(2),sigmaz(),qeye(3))
sx2=tensor(qeye(2),sigmax(),qeye(3))

e=basis(2,0)
gr=basis(2,1)

e0=tensor(e,basis(3,0))
g1=tensor(gr,basis(3,1))

ee0=tensor(e,e,basis(3,0)) #0
ee1=tensor(e,e,basis(3,1)) #1
ee2=tensor(e,e,basis(3,2)) #2

eg0=tensor(e,gr,basis(3,0)) #3
ge0=tensor(gr,e,basis(3,0)) #6

eg1=tensor(e,gr,basis(3,1)) #4
ge1=tensor(gr,e,basis(3,1)) #7

eg2=tensor(e,gr,basis(3,2)) #5
ge2=tensor(gr,e,basis(3,2)) #8

gg0=tensor(gr,gr,basis(3,0)) #9
gg1=tensor(gr,gr,basis(3,1)) #10
gg2=tensor(gr,gr,basis(3,2)) #11

#Definimos los parametros del problema
# '''----ENERGIAS JCM SIMPLE----'''
# w_0=1
# g=0.001*w_0
# def E_jcm1(n_:int,delta:list,x:float):
#     return [0.5*np.sqrt((delta-x*(2*n_-1))**2+4*g**2*n_),-0.5*np.sqrt((delta-x*(2*n_-1))**2+4*g**2*n_)]
# delta=np.linspace(-7*g,7*g,100000)
# colors=colormaps['plasma'](np.linspace(0,1,10))
# labels=['$E^1_\pm$','$E^2_\pm$','$E^3_\pm$']
# fig=plt.figure(figsize=(8,6))
# ax=fig.add_subplot()
# ax.plot(delta/g,-delta*100/2,color='black',label='$E^0$')
# for i,x in enumerate(np.linspace(0,2*g,10)):
#     ax.plot(delta/g,E_jcm1(2,delta,x)[0]*100,color=colors[i])
#     ax.plot(delta/g,E_jcm1(2,delta,x)[1]*100,color=colors[i])
# ax.grid()
# ax.legend()
# ax.set_xlim(delta[0]/g,delta[-1]/g)
# ax.set_xlabel('$\Delta/g$',size=15)
# ax.set_ylabel('Energia u.a.',size=15)
# ax.tick_params(labelsize=10)
# plt.show()
# plt.close(fig)

'''----ENERGIAS JCM DOBLE----'''
t_final=100000
steps=100000
w_0=1
g=0.001*w_0
p=0.005*g
k=2*g
x=0
J=0

gamma=0.1*g
acoplamiento='lineal'
if acoplamiento=='lineal':
    a=1/2
elif acoplamiento=='bs':
    a=1
else:
    print(f"Acoplamietno tiene que ser lineal o bs pero es {acoplamiento}")
    exit()

def beta_n(n_:int,k:float,J:float,x:float):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def rabi_freq(n_:int,j1:int,j2:int,d:float,g:float,k:float,J:float,x:float):
    return omega_general(n_,j2,d,g,k,J,x)-omega_general(n_,j1,d,g,k,J,x)

d=np.linspace(-15*g,15*g,20000)
# E=[[E00],[E11,E12,E13],[E21,E22,E23,E24],...,[En1,En2,En3,En4]]
# E=[[-d+J],[1/2*(x-d)+k+np.sqrt(2*g**2+(k-J+d/2-x/2)**2),1/2*(x-d)+k-np.sqrt(2*g**2+(k-J+d/2-x/2)**2),(-2*k-J)*np.ones_like(d)],[-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos(theta_n(2)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+2*np.pi)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+4*np.pi)/3),(x-J-2*k)*np.ones_like(d)]]
# E_jcm=[[1/2*np.sqrt(4*g**2+(d-x)**2),-1/2*np.sqrt(4*g**2+(d-x)**2)],[1/2*np.sqrt(2*4*g**2+(d-3*x)**2),-1/2*np.sqrt(2*4*g**2+(d-3*x)**2)]]

chi_list=np.linspace(0,5*g,10)
colors=colormaps['inferno'](np.linspace(0,1,len(chi_list)+2))
labels=['$\Omega_{21}$','$\Omega_{32}$','$\Omega_{31}$']
fig=plt.figure(figsize=(16,6))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
# ax3=fig.add_subplot(133)
ax=[ax1,ax2]
# for m,j in enumerate([[1,2],[2,3],[1,3]]):
for l,x in enumerate(chi_list):
    y12=[100*rabi_freq(2,1,2,delta,g,k,J,x) for delta in d]
    y23=[100*rabi_freq(2,2,3,delta,g,k,J,x) for delta in d]
    y31=[100*rabi_freq(2,1,3,delta,g,k,J,x) for delta in d]
    ax[0].plot(d/g,y12,color=colors[l])
    ax[1].plot(d/g,y23,color=colors[l])
    ax[1].plot(d/g,y31,color=colors[l])
ax[0].set_xlabel('$\Delta/g$')
ax[0].set_xlim(1/g*d[0],1/g*d[-1])
ax[1].set_xlim(1/g*d[0],1/g*d[-1])
ax1.grid()
ax2.grid()
# ax3.grid()
ax1.set_ylabel('Energia u.a.')
plt.show()

'''----frecuencias para diferentes k----'''

d=np.linspace(-15*g,15*g,20000)

k_list=np.linspace(0,5*g,10)
colors=colormaps['inferno'](np.linspace(0,1,len(k_list)+2))
labels=['$\Omega_{21}$','$\Omega_{32}$','$\Omega_{31}$']
fig=plt.figure(figsize=(16,6))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
# ax3=fig.add_subplot(133)
ax=[ax1,ax2]
# for m,j in enumerate([[1,2],[2,3],[1,3]]):
x=0
for l,k in enumerate(k_list):
    y12=[100*rabi_freq(2,1,2,delta,g,k,J,x) for delta in d]
    y23=[100*rabi_freq(2,2,3,delta,g,k,J,x) for delta in d]
    y31=[100*rabi_freq(2,1,3,delta,g,k,J,x) for delta in d]
    ax[0].plot(d/g,y12,color=colors[l])
    ax[1].plot(d/g,y23,color=colors[l])
    ax[1].plot(d/g,y31,color=colors[l])
ax[0].set_xlabel('$\Delta/g$')
ax[0].set_xlim(1/g*d[0],1/g*d[-1])
ax[1].set_xlim(1/g*d[0],1/g*d[-1])
ax1.grid()
ax2.grid()
# ax3.grid()
ax1.set_ylabel('Energia u.a.')
plt.show()


# fig=plt.figure(figsize=(8,6))
# ax=fig.add_subplot()
# # ax.set_title("Relación de dispersión",size=20)
# ax.plot(d/g,E_jcm[0][0]*200,linestyle="dashed",color="black",label="$2E_{JC}^{(1)}$")
# ax.plot(d/g,E_jcm[0][1]*200,linestyle="dashed",color="black")

# ax.plot(d/g,E_jcm[1][0]*200,linestyle="dashed",color="red",label="$2E_{JC}^{(2)}$")
# ax.plot(d/g,E_jcm[1][1]*200,linestyle="dashed",color="red")

# ax.plot(d/g,E[0][0]*100,color="black",label='$E^{(0)}$')
# ax.plot(d/g,E[1][0]*100,color="green",label='$E_1^{(1)}$')
# ax.plot(d/g,E[1][1]*100,color="green",label='$E_2^{(1)}$')
# ax.plot(d/g,E[1][2]*100,color="lime",label='$E_3^{(1)}$')

# ax.plot(d/g,E[2][0]*100,color="red",label='$E_1^{(2)}$')
# ax.plot(d/g,E[2][1]*100,color="orange",label='$E_2^{(2)}$')
# ax.plot(d/g,E[2][2]*100,color="yellow",label='$E_3^{(2)}$')
# ax.plot(d/g,E[2][3]*100,color="grey",label='$E_4^{(2)}$')
# ax.set_xlim(-10,10)
# ax.set_xlabel("$\Delta/g$")
# ax.set_ylabel("Energia")
# ax.legend(loc="upper right")
# ax.grid()
# plt.show()

