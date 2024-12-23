"""Heatmap: eje x es gt (tiempo adimensional), eje y es algun parametro, por ejemplo delta, y eje z (color) es la concurrence.
Idea sacada de un paper que paso fernando. Quiero ver como variando los parametros puedo encontrar o evidar sudden death effect en la concurrencia."""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors


from jcm_lib import simu_unit_y_disip,anim_univsdis


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


SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU


def concurrence_heatmap_delta(psi0,psi0Name):
    steps=2000
    t_final=10*steps
    w_0=1
    g=0.001*w_0
    J=0
    p=0.005*g
    x=0
    gamma=0.1*g
    k=0
    alpha=0

    gt=np.linspace(0,g*t_final,steps)
    delta=np.linspace(-3*g,3*g,129)
    delta_ticks=np.linspace(-3,3,129)

    concu_u=np.zeros((len(delta),steps))
    concu_d=np.zeros((len(delta),steps))

    for i,d in enumerate(delta):
        fg_u,fg_d,concu_u[i],concu_d[i]=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,alpha,psi0,t_final,steps)
        
    fig=plt.figure(figsize=(16,9))
    fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig.add_subplot(121)
    ax_d=fig.add_subplot(122,sharey=ax_u)
    ax_u.set_title('Unitario')
    ax_d.set_title('Disipativo')
    ax_u.set_xlabel('gt')
    ax_d.set_xlabel('gt')
    ax_u.set_ylabel('$\Delta/g$')
    c0 = ax_u.pcolor(gt, delta_ticks, concu_u, shading='auto', cmap='jet',vmin=0)
    contour_u = ax_u.contourf(gt, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    c1 = ax_d.pcolor(gt, delta_ticks, concu_d, shading='auto', cmap='jet',vmin=0)
    contour_d = ax_d.contourf(gt, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    fig.colorbar(c0, ax=ax_u,label="Concurrence")
    fig.colorbar(c1, ax=ax_d,label="Concurrence")

    plt.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\2\alpha=0 x=0 k=0 j=0 gamma=0.1g barrido delta\{psi0Name} concu heatmap SDE.png')
    plt.close()

def concurrence_heatmap_x(psi0,psi0Name):
    steps=2000
    t_final=10*steps
    w_0=1
    J=0
    g=0.001*w_0
    p=0.005*g
    d=0
    gamma=0.1*g
    k=0

    gt=np.linspace(0,g*t_final,steps)
    delta=np.linspace(-3*g,3*g,129)
    delta_ticks=np.linspace(-3,3,129)

    concu_u=np.zeros((len(delta),steps))
    concu_d=np.zeros((len(delta),steps))

    for i,d in enumerate(delta):
        fg_u,fg_d,concu_u[i],concu_d[i]=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
        
    fig=plt.figure(figsize=(16,9))
    fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig.add_subplot(121)
    ax_d=fig.add_subplot(122,sharey=ax_u)
    ax_u.set_title('Unitario')
    ax_d.set_title('Disipativo')
    ax_u.set_xlabel('gt')
    ax_d.set_xlabel('gt')
    ax_u.set_ylabel('$\Delta/g$')
    c0 = ax_u.pcolor(gt, delta_ticks, concu_u, shading='auto', cmap='jet',vmin=0)
    contour_u = ax_u.contourf(gt, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    c1 = ax_d.pcolor(gt, delta_ticks, concu_d, shading='auto', cmap='jet',vmin=0)
    contour_d = ax_d.contourf(gt, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    fig.colorbar(c0, ax=ax_u,label="Concurrence")
    fig.colorbar(c1, ax=ax_d,label="Concurrence")

    plt.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\2\x=0 k=0 j=0 gamma=0.1g barrido delta\{psi0Name} concu heatmap SDE.png')
    plt.close()


#concurrence_heatmap((eg1+ge1).unit(),'eg1+ge1')

for psi0,psi0Name in zip([gg1,eg0,(eg0+ge0).unit(),(eg0-ge0).unit(),eg1,(eg1+ge1).unit(),(eg1-ge1).unit(),(eg0+ge0+gg1).unit(),(ee0-gg2).unit(),(ee0+gg2).unit(),(ee0+eg1+ge1+gg2).unit()],['gg1','eg0','eg0+ge0','eg0-ge0','eg1','eg1+ge1','eg1-ge1','w','ee0-gg2','ee0+gg2','ee0+eg1+ge1+gg2']):
    concurrence_heatmap_delta(psi0,psi0Name)
