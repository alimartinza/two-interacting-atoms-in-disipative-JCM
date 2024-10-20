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

psi0=(ee0+gg2).unit()
psi0Name="ee0+gg2"
steps=3000
t_final=50000
w_0=1
J=0
g=0.001*w_0
p=0.005*g
delta=np.linspace(0,2*g,21)
gamma=0.1*g
kappa=np.linspace(0,2*g,21)

x=0
tot_iters=len(delta)*len(kappa)
iteracion=0
k_ax0, delta_ax0 = np.meshgrid(kappa,delta,sparse=True)
zs0=np.zeros((len(kappa),len(delta),steps))
for i1,d in enumerate(delta):
    for i2,k in enumerate(kappa):
        fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
        zs0[i1][i2]=canberra(fg_u,fg_d,temporal=True)
        iteracion+=1
        print("Lap 1/4")
        print(f"Aprox. Progress {iteracion*100/tot_iters}%")

x=0.5*g
tot_iters=len(delta)*len(kappa)
iteracion=0
k_ax1, delta_ax1 = np.meshgrid(kappa,delta,sparse=True)
zs1=np.zeros((len(kappa),len(delta),steps))
for i3,d in enumerate(delta):
    for i4,k in enumerate(kappa):
        fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
        zs1[i3][i4]=canberra(fg_u,fg_d,temporal=True)
        iteracion+=1
        print("Lap 2/4")
        print(f"Aprox. Progress {iteracion*100/tot_iters}%")

x=g
tot_iters=len(delta)*len(kappa)
iteracion=0
k_ax2, delta_ax2 = np.meshgrid(kappa,delta,sparse=True)
zs2=np.zeros((len(kappa),len(delta),steps))
for i5,d in enumerate(delta):
    for i6,k in enumerate(kappa):
        fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
        zs2[i5][i6]=canberra(fg_u,fg_d,temporal=True)
        iteracion+=1
        print("Lap 3/4")
        print(f"Aprox. Progress {iteracion*100/tot_iters}%")

x=2*g
tot_iters=len(delta)*len(kappa)
iteracion=0
k_ax3, delta_ax3 = np.meshgrid(kappa,delta,sparse=True)
zs3=np.zeros((len(kappa),len(delta),steps))
for i7,d in enumerate(delta):
    for i8,k in enumerate(kappa):
        fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
        zs3[i7][i8]=canberra(fg_u,fg_d,temporal=True)
        iteracion+=1
        print(f"Aprox. Progress {iteracion*100/tot_iters}%")



#color entre z.min() y z.max()
fig=plt.figure(figsize=(16,9))
fig.suptitle(f"$\psi_0$={psi0Name}")
ax0=fig.add_subplot(221)
ax0.set_title("$\chi=0$")
ax1=fig.add_subplot(222)
ax1.set_title("$\chi=0.5g$")
ax2=fig.add_subplot(223)
ax2.set_title("$\chi=g$")
ax3=fig.add_subplot(224)
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
    return [c0]
ax0.set_xlabel("$k/g$")
ax0.set_ylabel("$\Delta/g$")
fig.colorbar(c0, ax=ax1)

anim = FuncAnimation(fig, update, frames=steps/10, interval=10,blit=True)
anim.save(script_path+"\\"+"gifs"+"\\"+f"animation {psi0Name} canberra k vs delta chis.gif", writer='pillow')

plt.show()