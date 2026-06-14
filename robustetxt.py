from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors


from jcm_lib import simu_unit_y_disip,simu_unit,simu_disip


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

#DEFINIMOS LOS VECTORES DE LA BASE
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

# from mpl_toolkits.mplot3d import axes3d


SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.14)
plt.rc('figure.subplot',bottom=0.11)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)
plt.rc('figure.subplot',wspace=0.07)


script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

# def beta_n(n_:int,k:float,J:float,x:float):
#     return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

# def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

# def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

# def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

# def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

# def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

# def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
#     return 2*np.sqrt(-2*Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)




w_0=1
g=0.01*w_0

# gamma_list=[0.01*g,0.1*g,0.25*g]
# p=0.005*g

d=2*g
x=2*g

J=0*g
k=0*g
border=0.02

#'eg0-ge0+gg1','w(2)','eg0+ge0','eg0'
psi0Name='eg2'

# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez3t fg delta.txt')
# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta-chi\{psi0Name} k={k/g}g J={J/g}g delta-chi.txt')
# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g chi.txt')
#delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g k.txt')
# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta.txt')
# delta_fg_10T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez10t fg delta.txt')
def try_load(file):
    try:
        return np.loadtxt(file)
    except:
        print('no se encontro',file)
        return np.zeros((4,406))
delta_fg_chi_3T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g chi3t.txt')
delta_fg_chi_2T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g chi2t.txt')
delta_fg_chi_1T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g chi1t.txt')

delta_fg_delta_3T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta 3t.txt')
delta_fg_delta_2T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta 2t.txt')
delta_fg_delta_1T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta 1t.txt')


delta_fg_k_3T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g k3t.txt')
delta_fg_k_2T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g k2t.txt')
delta_fg_k_1T=try_load(rf'C:\Users\alima\Estudios\Tesis\two-interacting-atoms-in-disipative-JCM\robusteces\tcm\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g k1t.txt')
for i in [1,2,3]:
    delta_fg_delta_3T[i][delta_fg_delta_3T[i]>np.pi]+=-2*np.pi
    delta_fg_delta_3T[i][delta_fg_delta_3T[i]<-np.pi]+=2*np.pi

    delta_fg_delta_2T[i][delta_fg_delta_2T[i]>np.pi]+=-2*np.pi
    delta_fg_delta_2T[i][delta_fg_delta_2T[i]<-np.pi]+=2*np.pi

    delta_fg_delta_1T[i][delta_fg_delta_1T[i]>np.pi]+=-2*np.pi
    delta_fg_delta_1T[i][delta_fg_delta_1T[i]<-np.pi]+=2*np.pi

    delta_fg_chi_3T[i][delta_fg_chi_3T[i]>np.pi]+=-2*np.pi
    delta_fg_chi_3T[i][delta_fg_chi_3T[i]<-np.pi]+=2*np.pi

    delta_fg_chi_2T[i][delta_fg_chi_2T[i]>np.pi]+=-2*np.pi
    delta_fg_chi_2T[i][delta_fg_chi_2T[i]<-np.pi]+=2*np.pi

    delta_fg_chi_1T[i][delta_fg_chi_1T[i]>np.pi]+=-2*np.pi
    delta_fg_chi_1T[i][delta_fg_chi_1T[i]<-np.pi]+=2*np.pi

    delta_fg_k_3T[i][delta_fg_k_3T[i]>np.pi]+=-2*np.pi
    delta_fg_k_3T[i][delta_fg_k_3T[i]<-np.pi]+=2*np.pi

    delta_fg_k_2T[i][delta_fg_k_2T[i]>np.pi]+=-2*np.pi
    delta_fg_k_2T[i][delta_fg_k_2T[i]<-np.pi]+=2*np.pi

    delta_fg_k_1T[i][delta_fg_k_1T[i]>np.pi]+=-2*np.pi
    delta_fg_k_1T[i][delta_fg_k_1T[i]<-np.pi]+=2*np.pi

colors=mpl.colormaps['plasma'](np.linspace(0,1,3+1))

fig_rob_delta=plt.figure(figsize=(8,6),dpi=120)
ax_rob_delta=fig_rob_delta.add_subplot()
ax_rob_delta.set_xlim(np.min(delta_fg_delta_2T[0])/g,np.max(delta_fg_delta_2T[0])/g)
ax_rob_delta.set_ylim(np.min(delta_fg_delta_3T[3])/np.pi-border,np.max(delta_fg_delta_3T[3])/np.pi+border)
ax_rob_delta.hlines(0,np.min(delta_fg_delta_3T[0])/g,np.max(delta_fg_delta_3T[0])/g,colors='grey',linestyles='dashed',alpha=0.5)
ax_rob_delta.set_xlabel('$\Delta/g$')
ax_rob_delta.set_ylabel('$\delta \phi/\pi$')
ax_rob_delta.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)
# ax_rob_delta.vlines([-x/2/g,3/2*x/g],-0.1,0.02,colors='grey',linestyles='dashed')

for i in range(1,len(delta_fg_delta_3T)):
    ax_rob_delta.plot(delta_fg_delta_3T[0]/g,delta_fg_delta_3T[i]/np.pi,color=colors[i-1])
    ax_rob_delta.plot(delta_fg_delta_2T[0]/g,delta_fg_delta_2T[i]/np.pi,color=colors[i-1],linestyle='dashed')
    ax_rob_delta.plot(delta_fg_delta_1T[0]/g,delta_fg_delta_1T[i]/np.pi,color=colors[i-1],linestyle='dashdot')


fig_rob_chi=plt.figure(figsize=(8,6))
ax_rob_chi=fig_rob_chi.add_subplot()
ax_rob_chi.set_xlim(np.min(delta_fg_chi_2T[0])/g,np.max(delta_fg_chi_2T[0])/g)
ax_rob_chi.set_ylim(np.min(delta_fg_chi_3T[3])/np.pi-border,np.max(delta_fg_chi_3T[3])/np.pi+border)
ax_rob_chi.hlines(0,np.min(delta_fg_chi_3T[0])/g,np.max(delta_fg_chi_3T[0])/g,colors='grey',linestyles='dashed',alpha=0.5)
ax_rob_chi.set_xlabel('$\chi/g$')
ax_rob_chi.set_ylabel('$\delta \phi/\pi$')
ax_rob_chi.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)
# ax_rob_chi.vlines([-x/2/g,3/2*x/g],-0.1,0.02,colors='grey',linestyles='dashed')

for i in range(1,len(delta_fg_chi_3T)):
    ax_rob_chi.plot(delta_fg_chi_3T[0]/g,delta_fg_chi_3T[i]/np.pi,color=colors[i-1])
    ax_rob_chi.plot(delta_fg_chi_2T[0]/g,delta_fg_chi_2T[i]/np.pi,color=colors[i-1],linestyle='dashed')
    ax_rob_chi.plot(delta_fg_chi_1T[0]/g,delta_fg_chi_1T[i]/np.pi,color=colors[i-1],linestyle='dashdot')


fig_rob_k=plt.figure(figsize=(8,6))
ax_rob_k=fig_rob_k.add_subplot()
ax_rob_k.set_xlim(np.min(delta_fg_k_2T[0])/g,np.max(delta_fg_k_2T[0])/g)
ax_rob_k.set_ylim(np.min(delta_fg_k_3T[3])/np.pi-border,np.max(delta_fg_k_3T[3])/np.pi+border)
ax_rob_k.hlines(0,np.min(delta_fg_k_3T[0])/g,np.max(delta_fg_k_3T[0])/g,colors='grey',linestyles='dashed',alpha=0.5)
ax_rob_k.set_xlabel('$k/g$')
ax_rob_k.set_ylabel('$\delta \phi/\pi$')
ax_rob_k.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)
# ax_rob_k.vlines([-x/2/g,3/2*x/g],-0.1,0.02,colors='grey',linestyles='dashed')

for i in range(1,len(delta_fg_k_3T)):
    ax_rob_k.plot(delta_fg_k_3T[0]/g,delta_fg_k_3T[i]/np.pi,color=colors[i-1])
    ax_rob_k.plot(delta_fg_k_2T[0]/g,delta_fg_k_2T[i]/np.pi,color=colors[i-1],linestyle='dashed')
    ax_rob_k.plot(delta_fg_k_1T[0]/g,delta_fg_k_1T[i]/np.pi,color=colors[i-1],linestyle='dashdot')

plt.show()