"""La idea de este script es hacer una animacion temporal del canberra. En el eje x e y ponemos
parametros normales -por ejemplo- k y delta, y en la animacion mostramos como evoluciona 
temporalmente el canberra. es decir, lo que tendriamos que hacer es calcular el canberra para
diferentes tiempos e ir animando."""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from jcm_lib import simu_unit_y_disip,canberra

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

w_0=1
J=0
g=0.001*w_0
p=0.005*g
gamma=0.1*g
psi0=(gg1+eg0+ge0).unit()
psi0Name="w"
steps=3000
t_final=50000
kappa=np.linspace(0,2*g,21)
delta=np.linspace(0,2*g,21)


def canberra_anim_delta_vs_kappa(psi0,psi0Name:str,steps:int,t_final:int,delta:list,kappa:list,chi:list,frames:int,anim_time:float):
    if len(chi)!=4:
        print(f"Por ahora solo tenemos la posibilidad de plotear para 4 chis, pero tu lista de chis es de {len(chi)}")
        exit()
    if frames >steps:
        print("ERROR: la cantidad de frames de la animacion no puede ser mayor a la cantidad de steps en la simulacion")
        exit()

    w_0=1
    J=0
    g=0.001*w_0
    p=0.005*g
    gamma=0.1*g

    x=chi[0]
    tot_iters=len(delta)*len(kappa)
    iteracion=0
    k_ax0, delta_ax0 = np.meshgrid(kappa,delta,sparse=True)
    zs0=np.zeros((len(kappa),len(delta),frames))
    for i1,d in enumerate(delta):
        for i2,k in enumerate(kappa):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs0[i1][i2]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 1/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    x=chi[1]
    tot_iters=len(delta)*len(kappa)
    iteracion=0
    k_ax1, delta_ax1 = np.meshgrid(kappa,delta,sparse=True)
    zs1=np.zeros((len(kappa),len(delta),frames))
    for i3,d in enumerate(delta):
        for i4,k in enumerate(kappa):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs1[i3][i4]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 2/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    x=chi[2]
    tot_iters=len(delta)*len(kappa)
    iteracion=0
    k_ax2, delta_ax2 = np.meshgrid(kappa,delta,sparse=True)
    zs2=np.zeros((len(kappa),len(delta),frames))
    for i5,d in enumerate(delta):
        for i6,k in enumerate(kappa):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs2[i5][i6]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 3/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    x=chi[3]
    tot_iters=len(delta)*len(kappa)
    iteracion=0
    k_ax3, delta_ax3 = np.meshgrid(kappa,delta,sparse=True)
    zs3=np.zeros((len(kappa),len(delta),frames))
    for i7,d in enumerate(delta):
        for i8,k in enumerate(kappa):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs3[i7][i8]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 4/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")



    #color entre z.min() y z.max()
    fig=plt.figure(figsize=(16,9))
    fig.suptitle(f"$\psi_0$={psi0Name}")
    ax0=fig.add_subplot(221)
    ax0.set_title("$\chi=0$")
    ax1=fig.add_subplot(222,sharey=ax0)
    ax1.set_title("$\chi=0.5g$")
    ax2=fig.add_subplot(223,sharex=ax0)
    ax2.set_title("$\chi=g$")
    ax3=fig.add_subplot(224,sharey=ax2)
    ax3.set_title("$\chi=2g$")
    z_max =  max([zs0.flatten().max(),zs1.flatten().max(),zs2.flatten().max(),zs3.flatten().max()])
    #plotear el pcolormesh()


    # Create animation
    #condicion inicial
    c0 = ax0.pcolor(k_ax0/g, delta_ax0/g, zs0[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c1 = ax1.pcolor(k_ax1/g, delta_ax1/g, zs1[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c2 = ax2.pcolor(k_ax2/g, delta_ax2/g, zs2[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c3 = ax3.pcolor(k_ax3/g, delta_ax3/g, zs3[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    # Define an update function that modifies the contour plot data efficiently
    def update(frame):
        c0.set_array(zs0[:,:,frame].ravel())
        c1.set_array(zs1[:,:,frame].ravel())
        c2.set_array(zs2[:,:,frame].ravel())
        c3.set_array(zs3[:,:,frame].ravel())
        return [c0,c1,c2,c3]
    ax0.set_xlabel("$k/g$")
    ax0.set_ylabel("$\Delta/g$")
    fig.colorbar(c0, ax=ax1)

    anim = FuncAnimation(fig, update, frames=frames, interval=25,blit=True)
    anim.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} canberra k vs delta chis.gif", writer='pillow')

    plt.show()

def canberra_anim_delta_vs_chi(psi0,psi0Name:str,steps:int,t_final:int,delta:list,chi:list,kappa:list,frames:int,anim_time:float):
    if len(kappa)!=4:
        print(f"Por ahora solo tenemos la posibilidad de plotear para 4 k'es, pero tu lista de k'es es de {len(chi)}")
        exit()
    if frames >steps:
        print("ERROR: la cantidad de frames de la animacion no puede ser mayor a la cantidad de steps en la simulacion")
        exit()

    w_0=1
    J=0
    g=0.001*w_0
    p=0.005*g
    gamma=0.1*g

    k=kappa[0]
    tot_iters=len(delta)*len(chi)
    iteracion=0
    x_ax0, delta_ax0 = np.meshgrid(chi,delta,sparse=True)
    zs0=np.zeros((len(chi),len(delta),frames))
    for i1,d in enumerate(delta):
        for i2,x in enumerate(chi):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs0[i1][i2]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 1/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    k=kappa[1]
    tot_iters=len(delta)*len(chi)
    iteracion=0
    x_ax1, delta_ax1 = np.meshgrid(chi,delta,sparse=True)
    zs1=np.zeros((len(chi),len(delta),frames))
    for i3,d in enumerate(delta):
        for i4,x in enumerate(chi):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs1[i3][i4]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 2/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    k=kappa[2]
    tot_iters=len(delta)*len(chi)
    iteracion=0
    x_ax2, delta_ax2 = np.meshgrid(chi,delta,sparse=True)
    zs2=np.zeros((len(chi),len(delta),frames))
    for i5,d in enumerate(delta):
        for i6,x in enumerate(chi):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs2[i5][i6]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 3/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    k=kappa[3]
    tot_iters=len(delta)*len(chi)
    iteracion=0
    x_ax3, delta_ax3 = np.meshgrid(chi,delta,sparse=True)
    zs3=np.zeros((len(chi),len(delta),frames))
    for i7,d in enumerate(delta):
        for i8,x in enumerate(chi):
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
            zs3[i7][i8]=canberra(fg_u,fg_d,temporal=True)[::int(steps/frames)]
            iteracion+=1
            print("Lap 4/4")
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")



    #color entre z.min() y z.max()
    fig=plt.figure(figsize=(16,9))
    fig.suptitle(f"$\psi_0$={psi0Name}")
    ax0=fig.add_subplot(221)
    ax0.set_title(f"$k={kappa[0]}$")
    ax1=fig.add_subplot(222,sharey=ax0)
    ax1.set_title(f"$k={kappa[1]}$")
    ax2=fig.add_subplot(223,sharex=ax0)
    ax2.set_title(f"$k={kappa[2]}$")
    ax3=fig.add_subplot(224,sharey=ax2)
    ax3.set_title(f"$k={kappa[3]}$")
    z_max =  max([zs0.flatten().max(),zs1.flatten().max(),zs2.flatten().max(),zs3.flatten().max()])
    #plotear el pcolormesh()


    # Create animation
    #condicion inicial
    c0 = ax0.pcolor(x_ax0/g, delta_ax0/g, zs0[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c1 = ax1.pcolor(x_ax1/g, delta_ax1/g, zs1[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c2 = ax2.pcolor(x_ax2/g, delta_ax2/g, zs2[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    c3 = ax3.pcolor(x_ax3/g, delta_ax3/g, zs3[:,:,0], cmap='plasma', vmin=0, vmax=z_max)
    # Define an update function that modifies the contour plot data efficiently
    def update(frame):
        c0.set_array(zs0[:,:,frame].ravel())
        c1.set_array(zs1[:,:,frame].ravel())
        c2.set_array(zs2[:,:,frame].ravel())
        c3.set_array(zs3[:,:,frame].ravel())
        return [c0,c1,c2,c3]
    ax0.set_xlabel("$chi/g$")
    ax0.set_ylabel("$\Delta/g$")
    fig.colorbar(c0, ax=ax1)
    # simtime s=frames*interv ms=frames*interv/1000 s --> interv ms = simtime s/frames = simtime *1000 ms/frames
    anim = FuncAnimation(fig, update, frames=len(zs0[:,:,0]), interval=anim_time*1000/len(zs0[:,:,0]),blit=True)
    anim.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} canberra delta vs chi varios k.gif", writer='pillow')

    plt.show()

canberra_anim_delta_vs_chi(psi0,psi0Name,steps,t_final,np.linspace(0,2*g,21),np.linspace(0,2*g,21),[0,0.5*g,g,2*g],300,7)