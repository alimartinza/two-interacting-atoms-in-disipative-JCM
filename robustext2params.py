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
w_0=1
g=0.001*w_0

# gamma_list=[0.01*g,0.1*g,0.25*g]
# p=0.005*g

d=0*g
x=5*g

J=0*g
k=2.5*g


#'eg0-ge0+gg1','w(2)','eg0+ge0','eg0'
psi0Name='gg2'

# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez3t fg delta.txt')
delta_fg=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta-chi\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta-chi.txt')
# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g chi.txt')
# delta_fg_3T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta.txt')
# delta_fg_10T=np.loadtxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez10t fg delta.txt')

colors=mpl.colormaps['inferno'](np.linspace(0,1,len(delta_fg)+1))

fig_rob=plt.figure(figsize=(8,6))
ax_rob=fig_rob.add_subplot()
# ax_rob3t.set_xlim(np.min(delta_fg_3T[0])/g,np.max(delta_fg_3T[0])/g)
# ax_rob3t.set_ylim(-0.6,0.6)
ax_rob.hlines(0,np.min(delta_fg[0])/g,np.max(delta_fg[0])/g,colors='grey',linestyles='dashed',alpha=0.5)
ax_rob.set_xlabel('$\Delta/g$')
ax_rob.set_ylabel('$\delta \phi/\pi$')
ax_rob.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)

xmin=-0.5
xmax=0.5
# print(delta_fg_3T[0])
x=[0, 0.0005, 0.001, 0.0025, 0.0035, 0.005]
markes=['o','v','s','x','^','.',' ']
for i in range(int(len(delta_fg)/2)):
    ax_rob.scatter(delta_fg[2*i]/g,delta_fg[2*i+1]/np.pi,color=colors[2*i],marker='.')

    #lineas en funcion del detunning
    ax_rob.plot([3*x[i]/g-2*(k-J)/g,3*x[i]/g-2*(k-J)/g],[xmin+0.1,xmax-0.1],linestyle='dashed',marker='$3$',color=colors[2*i],alpha=0.8)
    ax_rob.plot([x[i]/g+2*(k-J)/g,x[i]/g+2*(k-J)/g],[xmin,xmax],linestyle='dashed',marker='$1$',color=colors[2*i],alpha=0.8)
    ax_rob.plot([2*x[i]/g,2*x[i]/g],[xmin+0.05,xmax-0.05],linestyle='dashed',marker='$2$',color=colors[2*i],alpha=0.8)

plt.show()