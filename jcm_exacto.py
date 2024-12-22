from qutip import *
import numpy as np
import matplotlib.pyplot as plt
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

t_final=100000
steps=100000
w_0=1
g=0.001*w_0
p=0.005*g
k=0
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

def beta_n(n_:int):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)
def gamma_n(n_:int,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))
def eta_n(n_:int,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))
def Q_n(n_:int):
    return gamma_n(n_)/3-beta_n(n_)*beta_n(n_)/9
def R_n(n_):
    return 1/54*(9*beta_n(n_)*gamma_n(n_)-27*eta_n(n_)-2*beta_n(n_)*beta_n(n_)*beta_n(n_))
def theta_n(n_):
    return np.arccos(R_n(n_)/np.sqrt(-Q_n(n_)**3))

d=np.linspace(-10*g,10*g,100000)
# E=[[E00],[E11,E12,E13],[E21,E22,E23,E24],...,[En1,En2,En3,En4]]
E=[[-d+J],[1/2*(x-d)+k+np.sqrt(2*g**2+(k-J+d/2-x/2)**2),1/2*(x-d)+k-np.sqrt(2*g**2+(k-J+d/2-x/2)**2),(-2*k-J)*np.ones_like(d)],[-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos(theta_n(2)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+2*np.pi)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+4*np.pi)/3),(x-J-2*k)*np.ones_like(d)]]
E_jcm=[[1/2*np.sqrt(4*g**2+d**2),-1/2*np.sqrt(4*g**2+d**2)],[1/2*np.sqrt(2*4*g**2+d**2),-1/2*np.sqrt(2*4*g**2+d**2)]]

plt.title("Relación de dispersión",size=20)
plt.plot(d/g,E_jcm[0][0]*200,linestyle="dashed",color="black",label="$2E_{JC}^{(1)}$")
plt.plot(d/g,E_jcm[0][1]*200,linestyle="dashed",color="black")

plt.plot(d/g,E_jcm[1][0]*200,linestyle="dashed",color="red",label="$2E_{JC}^{(2)}$")
plt.plot(d/g,E_jcm[1][1]*200,linestyle="dashed",color="red")

plt.plot(d/g,E[0][0]*100,color="black",label='$E^{(0)}$')
plt.plot(d/g,E[1][0]*100,color="green",label='$E_1^{(1)}$')
plt.plot(d/g,E[1][1]*100,color="green",label='$E_2^{(1)}$')
plt.plot(d/g,E[1][2]*100,color="lime",label='$E_3^{(1)}$')

plt.plot(d/g,E[2][0]*100,color="red",label='$E_1^{(2)}$')
plt.plot(d/g,E[2][1]*100,color="orange",label='$E_2^{(2)}$')
plt.plot(d/g,E[2][2]*100,color="yellow",label='$E_3^{(2)}$')
plt.plot(d/g,E[2][3]*100,color="grey",label='$E_4^{(2)}$')
plt.xlim(-10,10)
plt.xlabel("$\Delta/g$")
plt.ylabel("Energia")
plt.legend(loc="upper right")
plt.grid()
plt.show()

