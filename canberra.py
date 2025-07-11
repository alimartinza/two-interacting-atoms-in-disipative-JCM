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

# psi0=(ee0+gg2).unit()
# psi0Name="ee0+gg2"
# steps=3000
# t_final=50000
# w_0=1
# J=0
# g=0.001*w_0
# p=0.005*g
# delta=np.linspace(0,2*g,3)
# x=0
# gamma=0.1*g
# kappa=np.linspace(0,2*g,3)
# k_ax, delta_ax = np.meshgrid(kappa,delta,sparse=True)
# zs=np.zeros((len(kappa),len(delta)))
# for j,d in enumerate(delta):
#     for i,k in enumerate(kappa):  
#         fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)
#         zs[j][i]=canberra(fg_u,fg_d)

"""DELTA vs K con algunos CHI"""
psi0=(ee0+gg2).unit()
psi0Name="ee0+gg2"
steps=3000
t_final=50000
w_0=1
J=0
g=0.001*w_0
p=0.005*g
gamma=0.1*g

len_axis=41

kappa=np.linspace(0,2*g,len_axis)
delta=np.linspace(-2*g,2*g,len_axis)
save_frames=300


def canberra_sim_and_save_delta_vs_kappa(psi0,psi0Name,steps,t_final,w_0,J,g,p,gamma,kappa,delta,chi,save_frames):

    data=pd.DataFrame()

    tot_iters=len(delta)*len(kappa)
    iteracion=0
    # zs0=np.zeros((len(delta),len(kappa),frames))
    for d in delta:
        for k in kappa:
            param_name=f'{d/g}g;{k/g}g'
            fg_u,fg_d=simu_unit_y_disip(w_0,g,k,J,d,chi,gamma,p,psi0,t_final,steps)
            data[param_name]=canberra(fg_u,fg_d,temporal=True)[np.linspace(0, steps - 1, save_frames, dtype=int)]
            iteracion+=1
            print(f"Aprox. Progress {iteracion*100/tot_iters}%")

    data.to_csv(f'canberra delta {delta[0]}-{delta[-1]} vs kappa {kappa[0]}-{kappa[-1]} x={chi} {psi0Name}.csv')

def load_csv_canberra(filepath:str,chi:float):
    """Load .csv de canberra"""

    data=pd.read_csv(filepath)
    zs=np.empty((len(data.keys())-1,len(data[data.keys()[0]])))
    len_params=int(np.sqrt(len(data.keys())-1))
    params_inicial,params_final=data.keys()[1].replace("g","").split(';'),data.keys()[-1].replace("g","").split(';')
    delta_new=np.linspace(float(params_inicial[0])*g,float(params_final[0])*g,len_params)
    kappa_new=np.linspace(float(params_inicial[1])*g,float(params_final[1])*g,len_params)
    print()
    for i3,param_names in enumerate(data.keys()[1:]):
        zs[i3]=data[param_names]

    delta_ax, k_ax = np.meshgrid(delta_new,kappa_new,sparse=True)


    # #color entre z.min() y z.max()
    fig=plt.figure(figsize=(16,9))
    ax=fig.add_subplot()
    fig.suptitle(f"$\psi_0$={psi0Name} chi={chi/g}g")
    z_min, z_max = zs.min(), zs.max()
    #plotear el pcolormesh()
    c=ax.pcolor(delta_ax/g, k_ax/g, zs[:,-1].reshape((len_params,len_params)), cmap='plasma', vmin=0, vmax=z_max)
    ax.set_xlabel("$k/g$")
    ax.set_ylabel("$\Delta/g$")
    fig.colorbar(c, ax=ax)
    plt.show()

for x in [0.1*g,g,2*g]:
    canberra_sim_and_save_delta_vs_kappa(psi0,psi0Name,steps,t_final,w_0,J,g,p,gamma,kappa,delta,x,300)