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
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.14)
plt.rc('figure.subplot',bottom=0.122)
plt.rc('figure.subplot',right=1.0)
plt.rc('figure.subplot',top=0.95)
plt.rc('figure.subplot',wspace=0.07)


script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

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
    return 2*np.sqrt(-2*Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def energiasn1(j,g,d,x,k,J):
    if j==1: return (x-d)/2+k+np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==2: return (x-d)/2+k-np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==3: return -2*k-J
    else: 
        print('valor inesperado de j')
        exit()



def concurrence_heatmap_delta(psi0,psi0Name,x,J,k):
    w_0=1
    g=0.001*w_0
    
    gamma_list=[0.01*g,0.1*g,0.25*g]
    p=0.005*g
    
    d=0

    steps=3000
    T=2*np.pi/omega_general(1,1,d,g,k,J,x)

    # print(omega_general(1,2,d,g,k,J,x))
    t_final=5*T
    t=np.linspace(0,t_final,steps)
    delta=np.linspace(-5*g,25*g,150)
    delta_ticks=np.linspace(-5,25,150)

    delta_fg_3T=np.zeros((len(gamma_list),len(delta)))
    delta_fg_10T=np.zeros((len(gamma_list),len(delta)))

    concu_u=np.zeros((len(delta),steps))
    concu_d=np.zeros((len(gamma_list),len(delta),steps))

    for i,d in enumerate(delta):
        fg_u,concu_u[i]=simu_unit(w_0,g,k,J,d,x,1,psi0,t_final,steps)
        for j,gamma in enumerate(gamma_list):
            fg_d,concu_d[j][i]=simu_disip(w_0,g,k,J,d,x,gamma,p,1,psi0,t_final,steps)
            delta_fg_3T[j][i]=fg_d[int(len(delta)*3/t_final*T)]-fg_u[int(len(delta)*3/t_final*T)]
            delta_fg_10T[j][i]=fg_d[-1]-fg_u[-1]


    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez3t fg delta.txt',delta_fg_3T)
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez10t fg delta.txt',delta_fg_10T)

    fig_u=plt.figure(figsize=(8,6))
    fig_d=plt.figure(figsize=(8,6))
    # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig_u.add_subplot()
    ax_d=fig_d.add_subplot()
    ax_u.set_xlabel('$t/T_0$')
    ax_d.set_xlabel('$t/T_0$')
    ax_u.set_ylabel('$\Delta/g$')
    ax_d.set_ylabel('$\Delta/g$')
    c0 = ax_u.pcolor(t/T, delta_ticks, concu_u, shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_u = ax_u.contourf(t/T, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    c1 = ax_d.pcolor(t/T, delta_ticks, concu_d[2], shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_d = ax_d.contourf(t/T, delta_ticks, concu_d[2],levels=[0,0.01],colors='black',linewidths=1)
    ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    fig_d.colorbar(c1, ax=ax_d,shrink=0.7)
    # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    fig_u.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g gamma=0.25g concu delta uni.png')
    fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g gamma=0.25g concu delta dis.png')
  



def concurrence_heatmap_chi(psi0,psi0Name,d,J,k):
    w_0=1
    g=0.001*w_0
    
    gamma_list=[0.01*g,0.1*g,0.25*g]
    p=0.005*g
    
    x=0

    steps=3000
    T=2*np.pi/omega_general(1,1,d,g,k,J,x)

    # print(omega_general(1,2,d,g,k,J,x))
    t_final=5*T
    t=np.linspace(0,t_final,steps)
    chi=np.linspace(0,15*g,100)
    chi_ticks=np.linspace(0,15,100)

    delta_fg_3T=np.zeros((len(gamma_list),len(chi)))
    delta_fg_10T=np.zeros((len(gamma_list),len(chi)))

    concu_u=np.zeros((len(chi),steps))
    concu_d=np.zeros((len(gamma_list),len(chi),steps))

    for i,x in enumerate(chi):
        fg_u,concu_u[i]=simu_unit(w_0,g,k,J,d,x,1,psi0,t_final,steps)
        for j,gamma in enumerate(gamma_list):
            fg_d,concu_d[j][i]=simu_disip(w_0,g,k,J,d,x,gamma,p,1,psi0,t_final,steps)
            delta_fg_3T[j][i]=fg_d[int(len(chi)*3/t_final*T)]-fg_u[int(len(chi)*3/t_final*T)]
            delta_fg_10T[j][i]=fg_d[-1]-fg_u[-1]


    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g rebustez3t fg chi.txt',delta_fg_3T)
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g rebustez10t fg chi.txt',delta_fg_10T)

    fig_u=plt.figure(figsize=(8,6))
    fig_d=plt.figure(figsize=(8,6))
    # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig_u.add_subplot()
    ax_d=fig_d.add_subplot()
    ax_u.set_title('Unitario')
    ax_d.set_title('Disipativo')
    ax_u.set_xlabel('$t/T_0$')
    ax_d.set_xlabel('$t/T_0$')
    ax_u.set_ylabel('$\chi/g$')
    ax_d.set_ylabel('$\chi/g$')
    c0 = ax_u.pcolor(t/T, chi_ticks, concu_u, shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_u = ax_u.contourf(t/T, chi_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    c1 = ax_d.pcolor(t/T, chi_ticks, concu_d[2], shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_d = ax_d.contourf(t/T, chi_ticks, concu_d[2],levels=[0,0.01],colors='black',linewidths=1)
    ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    fig_d.colorbar(c1, ax=ax_d,shrink=0.7)
    # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    fig_u.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g gamma=0.25g concu chi uni.png')
    fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\chi\{psi0Name} d={d/g}g k={k/g}g J={J/g}g gamma=0.25g concu chi dis.png')
    plt.close()

def concurrence_heatmap_k(psi0,psi0Name,d,x):
    w_0=1
    g=0.001*w_0
    
    gamma_list=[0.01*g,0.1*g,0.25*g]
    p=0.005*g
    J=0
    k=0

    steps=3000
    T=2*np.pi/omega_general(1,1,d,g,k,J,x)
    J=15*g
    # print(omega_general(1,2,d,g,k,J,x))
    t_final=5*T
    t=np.linspace(0,t_final,steps)
    k_list=np.linspace(0,30*g,100)
    kj_ticks=np.linspace(-15,15,100)

    delta_fg_3T=np.zeros((len(gamma_list),len(k_list)))
    delta_fg_10T=np.zeros((len(gamma_list),len(k_list)))

    concu_u=np.zeros((len(k_list),steps))
    concu_d=np.zeros((len(gamma_list),len(k_list),steps))

    for i,k in enumerate(k_list):
        fg_u,concu_u[i]=simu_unit(w_0,g,k,J,d,x,1,psi0,t_final,steps)
        for j,gamma in enumerate(gamma_list):
            fg_d,concu_d[j][i]=simu_disip(w_0,g,k,J,d,x,gamma,p,1,psi0,t_final,steps)
            delta_fg_3T[j][i]=fg_d[int(len(k_list)*3/t_final*T)]-fg_u[int(len(k_list)*3/t_final*T)]
            delta_fg_10T[j][i]=fg_d[-1]-fg_u[-1]


    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g rebustez3t fg k.txt',delta_fg_3T)
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g rebustez10t fg k.txt',delta_fg_10T)

    fig_u=plt.figure(figsize=(8,6))
    fig_d=plt.figure(figsize=(8,6))
    # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig_u.add_subplot()
    ax_d=fig_d.add_subplot()
    ax_u.set_xlabel('$t/T_0$')
    ax_d.set_xlabel('$t/T_0$')
    ax_u.set_ylabel('$k/g$')
    ax_d.set_ylabel('$k/g$')
    c0 = ax_u.pcolor(t/T, kj_ticks, concu_u, shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_u = ax_u.contourf(t/T, kj_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    c1 = ax_d.pcolor(t/T, kj_ticks, concu_d[2], shading='auto', cmap='jet',vmin=0,vmax=1)
    contour_d = ax_d.contourf(t/T, kj_ticks, concu_d[2],levels=[0,0.01],colors='black',linewidths=1)
    ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    fig_d.colorbar(c1, ax=ax_d,shrink=0.7)
    # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    fig_u.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g gamma=0.25g concu k uni.png')
    fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\k\{psi0Name} d={d/g}g x={x/g}g J={J/g}g gamma=0.25g concu k dis.png')
    plt.close()



# g=0.001
# concurrence_heatmap_delta((eg1+ge1).unit(),'eg1+ge1',0.71*g,0.54*g,5.54*g)
g=0.001
# for params in [[0,0,0.5*g],[0,0,2.5*g]]:
#     for psi0,psi0Name in zip([(eg0+ge0).unit(),(eg1+ge1).unit(),(ee0+gg2).unit()],['eg0+ge0','eg1+ge1','ee0+gg2']):
#         concurrence_heatmap_delta(psi0,psi0Name,params[0],params[1],params[2])
# for params in [[0,0,0],[0,0,0.5*g],[0,0,2.5*g],[0.5*g,0,0],[5*g,0,0]]:
#     for psi0,psi0Name in zip([(eg0+ge0).unit(),(eg1+ge1).unit(),(ee0+gg2).unit()],['eg0+ge0','eg1+ge1','ee0+gg2']):
#         concurrence_heatmap_chi(psi0,psi0Name,params[0],params[1],params[2]) 
#         concurrence_heatmap_k(psi0,psi0Name,params[0],params[2],params[1])



psi0=(eg0+ge0).unit()
psi0Name='eg0+ge0'
concurrence_heatmap_delta(psi0,psi0Name,x=0,J=0,k=0)
concurrence_heatmap_delta(psi0,psi0Name,x=0.1*g,J=0,k=0)
concurrence_heatmap_delta(psi0,psi0Name,x=5*g,J=0,k=0)

concurrence_heatmap_chi(psi0,psi0Name,d=1*g,k=5*g,J=0)

concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=0*g)
concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=5*g)
concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=0.5*g)
concurrence_heatmap_k(psi0,psi0Name,d=1*g,x=0*g)
concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=0*g)

psi0=(eg1+ge1).unit()
psi0Name='eg1+ge1'
concurrence_heatmap_delta(psi0,psi0Name,x=0,J=0,k=0)
concurrence_heatmap_delta(psi0,psi0Name,x=0.1*g,J=0,k=0)

concurrence_heatmap_chi(psi0,psi0Name,d=1*g,k=0*g,J=0)

concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=0*g)
concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=5*g)
concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=0.5*g)
concurrence_heatmap_k(psi0,psi0Name,d=1*g,x=0*g)
concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=0*g)

# psi0=(ee0+gg2).unit()
# psi0Name='eg1+ge1+gg1'
# concurrence_heatmap_chi(psi0,psi0Name,d=5*g,k=5*g,J=0)
# concurrence_heatmap_chi(psi0,psi0Name,d=0*g,k=5*g,J=0)
# concurrence_heatmap_chi(psi0,psi0Name,d=5*g,k=0*g,J=0)

# concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=0*g,J=0)
# concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=5*g,J=0)
# concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=5*g,J=0)

# psi0=(eg1+ge1+ee0).unit()
# psi0Name='eg1+ge1+ee0'
# concurrence_heatmap_chi(psi0,psi0Name,d=5*g,k=5*g,J=0)
# concurrence_heatmap_chi(psi0,psi0Name,d=0*g,k=5*g,J=0)
# concurrence_heatmap_chi(psi0,psi0Name,d=5*g,k=0*g,J=0)

# concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=0*g,J=0)
# concurrence_heatmap_k(psi0,psi0Name,d=0*g,x=5*g,J=0)
# concurrence_heatmap_k(psi0,psi0Name,d=5*g,x=5*g,J=0)
