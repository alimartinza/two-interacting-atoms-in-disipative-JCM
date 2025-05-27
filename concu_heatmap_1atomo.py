from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence
from matplotlib import colormaps

e=basis(2,0)
gr=basis(2,1)

e0=tensor(e,basis(2,0)) #1
e1=tensor(e,basis(2,1)) #2
g0=tensor(gr,basis(2,0)) #3
g1=tensor(gr,basis(2,1)) #4

sz=tensor(sigmaz(),qeye(2))
sx=tensor(sigmax(),qeye(2))
sy=tensor(sigmay(),qeye(2))
sp=tensor(sigmap(),qeye(2))
sm=tensor(sigmam(),qeye(2))
a=tensor(qeye(2),destroy(2))

SMALL_SIZE = 15
MEDIUM_SIZE = 12.5
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.13)
plt.rc('figure.subplot',bottom=0.112)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)

w_0=1
g=0.001*w_0
def omega_n(n_:int,delta:float,chi:float):
    return np.sqrt((delta-chi*(2*n_-1))**2+4*g**2*n_)

def concurrence_heatmap_delta(psi0,psi0Name,x,delta_min=None,delta_max=None,unit:bool=True):
    w_0=1
    g=0.001*w_0

    if delta_min==None and delta_max==None:
        delta_min=-15*g
        delta_max=15*g
    else: None
    gamma=0.25*g
    p=0.005*g

    steps=6000
    # if psi0Name=='eg0+ge0':
    #     T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))
    # else:
    #     T=2*np.pi/np.abs(rabi_freq(2,1,2,delta,g,k,J,x))
    # print(omega_general(1,2,d,g,k,J,x))

    T=2*np.pi/np.abs(omega_n(1,0,0))
    t_final=7*T
    t=np.linspace(0,t_final,steps)
    delta=np.linspace(delta_min,delta_max,150)
    delta_ticks=np.linspace(delta_min/g,delta_max/g,150)

    # delta_fg_3T=np.zeros((len(gamma_list),len(delta)))
    # delta_fg_10T=np.zeros((len(gamma_list),len(delta)))

    concu_u=np.zeros((len(delta),steps))
    fg_u=np.zeros((len(delta),steps))
    concu_d=np.zeros((len(delta),steps))
    fg_d=np.zeros((len(delta),steps))
    for i,d in enumerate(delta):
        if unit==True: 
            H=x*a.dag()*a*a.dag()*a+d/2*sz + g*(a.dag()*sm+a*sp)

            sol_u=mesolve(H,psi0,t)
            # concu_u[i]=concurrence(sol_u.states)
            fg_u[i],_,_=fases(sol_u)
        else: None

        H=x*a.dag()*a*a.dag()*a+d/2*sz + g*(a.dag()*sm+a*sp) 
        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp] #operadores de colapso/lindblad
        sol_d=mesolve(H,psi0,t,c_ops=l_ops)
        # concu_d[i]=concurrence(sol_d.states)
        fg_d[i],_,_=fases(sol_d)
            # delta_fg_3T[j][i]=fg_d[int(len(delta)*3/t_final*T)]-fg_u[int(len(delta)*3/t_final*T)]
            # delta_fg_10T[j][i]=fg_d[-1]-fg_u[-1]


    # np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez3t fg delta.txt',delta_fg_3T)
    # np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g rebustez10t fg delta.txt',delta_fg_10T)
    # if unit==True:
    #     fig_u=plt.figure(figsize=(8,6))
    #     # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    #     ax_u=fig_u.add_subplot()
    #     ax_u.set_xlabel('$t/T_0$')
    #     ax_u.set_ylabel('$\Delta/g$')
    #     c0 = ax_u.pcolor(t/T, delta_ticks, concu_u, shading='auto', cmap='jet',vmin=0,vmax=1)
    #     contour_u = ax_u.contourf(t/T, delta_ticks, concu_u,levels=[0,0.01],colors='black',linewidths=1)
    #     ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    #     fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    #     fig_u.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} x={x/g}g gamma=0.25g concu delta uni.png')
       

    # else: None

    # fig_d=plt.figure(figsize=(8,6))
    # ax_d=fig_d.add_subplot()
    # ax_d.set_xlabel('$t/T_0$')
    # ax_d.set_ylabel('$\Delta/g$')
    # c1 = ax_d.pcolor(t/T, delta_ticks, concu_d, shading='auto', cmap='jet',vmin=0,vmax=1)
    # contour_d = ax_d.contourf(t/T, delta_ticks, concu_d,levels=[0,0.01],colors='black',linewidths=1)
    # ax_d.clabel(contour_d, fmt="%.1f",colors='red',fontsize=10)
    # fig_d.colorbar(c1, ax=ax_d,shrink=0.7)
    # # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    # fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} x={x/g}g gamma=0.25g concu delta dis.png')
    # plt.close('all')

    if unit==True:
        fig_u=plt.figure(figsize=(8,6))
        # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
        ax_u=fig_u.add_subplot()
        ax_u.set_xlabel('$t/T_0$')
        ax_u.set_ylabel('$\Delta/g$')
        c0 = ax_u.pcolor(t/T, delta_ticks, fg_u, shading='auto', cmap='jet')
        fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
        fig_u.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} x={x/g}g gamma=0.25g fg uni.png')
       

    else: None

    fig_d=plt.figure(figsize=(8,6))
    ax_d=fig_d.add_subplot()
    ax_d.set_xlabel('$t/T_0$')
    ax_d.set_ylabel('$\Delta/g$')
    c1 = ax_d.pcolor(t/T, delta_ticks, fg_d, shading='auto', cmap='jet')

    fig_d.colorbar(c1, ax=ax_d,shrink=0.7)
    # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} x={x/g}g gamma=0.25g fg dis.png')

    fig_delta=plt.figure(figsize=(8,6))
    ax_delta=fig_delta.add_subplot()
    ax_delta.set_xlabel('$t/T_0$')
    ax_delta.set_ylabel('$\Delta/g$')
    c1 = ax_delta.pcolor(t/T, delta_ticks, fg_u-fg_d, shading='auto', cmap='jet')

    fig_d.colorbar(c1, ax=ax_delta,shrink=0.7)
    # fig.colorbar(c1, ax=ax_d,label="Concurrence")

    fig_d.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\4\concu\{psi0Name} x={x/g}g gamma=0.25g  deltafg dis.png')
    plt.close('all')

# concurrence_heatmap_delta(e0,'e0',0)
# concurrence_heatmap_delta(e0,'e0',5*g)
# concurrence_heatmap_delta(g1,'g1',0)
# concurrence_heatmap_delta(g1,'g1',5*g)
# concurrence_heatmap_delta((e0+g1).unit(),'e0+g1',0)
# concurrence_heatmap_delta((e0+g1).unit(),'e0+g1',5*g)

x=5*g
d=0*g
gamma=0.25*g
p=0.005*g
steps=12000
T=2*np.pi/np.abs(omega_n(1,0,0))
t_final=20*T
t=np.linspace(0,t_final,steps)
psi0=(e0).unit()
psi0Name='e0'
H=x*a.dag()*a*a.dag()*a+d/2*sz + g*(a.dag()*sm+a*sp)

sol_u=mesolve(H,psi0,t)
# concu_u[i]=concurrence(sol_u.states)
fg_u,_,_,_=fases(sol_u)
l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp] #operadores de colapso/lindblad
sol_d=mesolve(H,psi0,t,c_ops=l_ops)
# concu_d[i]=concurrence(sol_d.states)
fg_d,_,_,_=fases(sol_d)
concu_u=concurrence(sol_u.states)
concu_d=concurrence(sol_d.states)
fig=plt.figure(figsize=(8,6))
fig.suptitle(f'$\psi_0={psi0Name} ; \chi={x/g}g ; \Delta={d/g}g$')
ax_fg=fig.add_subplot(121)
ax_concu=fig.add_subplot(122)
ax_fg.set_xlabel('$t/T_0$')
ax_fg.set_ylabel('$\phi/\pi$')
ax_fg.plot(t/T,fg_u/np.pi,color='black',label='unitario')
ax_fg.plot(t/T,fg_d/np.pi,color='red',label='disiaptivo')
ax_concu.plot(t/T,concu_u,color='black',label='unitario')
ax_concu.plot(t/T,concu_d,color='red',label='disiaptivo')
ax_fg.legend()
ax_concu.legend()
plt.grid()

plt.show()
