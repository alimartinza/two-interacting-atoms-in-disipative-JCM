from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from jcm_lib import fases,concurrence_ali
from entrelazamiento_lib import negativity_hor

import matplotlib as mpl
from matplotlib import cm
import os 

# Definiciones ----


## 1 Estados QUTIP ---- 

#DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
N_c=3
n=tensor(qeye(2),qeye(2),num(N_c))
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
a=tensor(qeye(2),qeye(2),destroy(N_c))
sm1=tensor(sigmam(),qeye(2),qeye(N_c))
sp1=tensor(sigmap(),qeye(2),qeye(N_c))
sz1=tensor(sigmaz(),qeye(2),qeye(N_c))
sx1=tensor(sigmax(),qeye(2),qeye(N_c))
sm2=tensor(qeye(2),sigmam(),qeye(N_c))
sp2=tensor(qeye(2),sigmap(),qeye(N_c))
sz2=tensor(qeye(2),sigmaz(),qeye(N_c))
sx2=tensor(qeye(2),sigmax(),qeye(N_c))

#DEFINIMOS LOS VECgORES DE LA BASE
e=basis(2,0)
gr=basis(2,1)

# e0=tensor(e,basis(N_c,0))
# g0=tensor(gr,basis(N_c,0))
# g1=tensor(gr,basis(N_c,1))
# sx=tensor(sigmax(),qeye(N_c))
# sy=tensor(sigmay(),qeye(N_c))
# sz=tensor(sigmaz(),qeye(N_c))
# sp=tensor(sigmap(),qeye(N_c))
# sm=tensor(sigmam(),qeye(N_c))


ee0=tensor(e,e,basis(N_c,0)) #0
ee1=tensor(e,e,basis(N_c,1)) #1
ee2=tensor(e,e,basis(N_c,2)) #2

eg0=tensor(e,gr,basis(N_c,0)) #3
ge0=tensor(gr,e,basis(N_c,0)) #6

eg1=tensor(e,gr,basis(N_c,1)) #4
ge1=tensor(gr,e,basis(N_c,1)) #7

eg2=tensor(e,gr,basis(N_c,2)) #5
ge2=tensor(gr,e,basis(N_c,2)) #8

gg0=tensor(gr,gr,basis(N_c,0)) #9
gg1=tensor(gr,gr,basis(N_c,1)) #10
gg2=tensor(gr,gr,basis(N_c,2)) #11


##2 Config matplotlib ----
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure.subplot',left=0.18)
plt.rc('figure.subplot',bottom=0.12)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)

## 3 Funciones ----
def beta_n(n_:int,k,J:float,x):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d,g:float,k,J:float,x,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d,g:float,k,J:float,x,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

def Q_n(n_:int,d,g:float,k,J:float,x):
    return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

def R_n(n_:int,d,g:float,k,J:float,x):
    return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

def theta_n(n_:int,d,g:float,k,J:float,x):
    return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

def En_(n_:int,j:int,d,g:float,k,J:float,x):
    return -beta_n(n_,k,J,x)/3+2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def Omega_n_ij(n_,i,j,d,g,k,J,x):
    return En_(n_,j,d,g,k,J,x)-En_(n_,i,d,g,k,J,x)

def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def rabi_freq(n_:int,j1:int,j2:int,d:float,g:float,k:float,J:float,x:float):
    return omega_general(n_,j2,d,g,k,J,x)-omega_general(n_,j1,d,g,k,J,x)

def energiasn1(j,g,d,x,k,J):
    if j==1: return (x-d)/2+k+np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==2: return (x-d)/2+k-np.sqrt(2*g**2+(k-J+(d-x)/2)**2)
    elif j==3: return -2*k-J
    else: 
        print('valor inesperado de j')
        exit()

def H_tcm(g,d,x,k,J):
    return x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2


script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
os.chdir(script_path)

def pr(estado):
    return estado.unit()*estado.unit().dag()


def vectorBloch(v1,v2,sol_states,steps,ciclos_bloch,T,t_final,points):
    sz_1=pr(v1)-pr(v2)
    sx_1=v1*v2.dag()+v2*v1.dag()
    sy_1=-1j*v1*v2.dag()+1j*v2*v1.dag()

    expect_sx_1=[expect(sx_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    expect_sy_1=[expect(sy_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    expect_sz_1=[expect(sz_1,sol_states[i]) for i in range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))]
    return [expect_sx_1,expect_sy_1,expect_sz_1]

def ket_to_bloch(v1,v2,ket):
    sz_1=pr(v1)-pr(v2)
    sx_1=v1*v2.dag()+v2*v1.dag()
    sy_1=-1j*v1*v2.dag()+1j*v2*v1.dag()
    
    return [expect(sx_1,ket),expect(sy_1,ket),expect(sz_1,ket)]

def l_ops(gamma,p):
    return [np.sqrt(gamma)*a,np.sqrt(p)*sm1,np.sqrt(p)*sm2]


# Planos de minima energia ----

from scipy.signal import argrelmin, argrelmax
w=1
g=0.01*w

def planos_minima_Omegaij(N_:int,d=np.linspace(-100*g,100*g,10000),x=np.linspace(0,15*g,75),k=np.linspace(-10*g,10*g,75),J=0):
    x_mesh,k_mesh=np.meshgrid(x,k)

    d_O12_min=np.zeros((3,len(x),len(k)))
    d_O23_min=np.zeros((3,len(x),len(k)))
    d_O31_min=np.zeros((3,len(x),len(k)))

    for i,x_i in enumerate(x):
        for j,k_j in enumerate(k):

            O12=np.abs(Omega_n_ij(N_,1,2,d,g,k_j,J,x_i))
            # fig,ax=plt.subplots()
            # plt.plot(d,O12)
            # plt.show()
            i_12_max=argrelmax(O12,order=1)
            i_12_min=argrelmin(O12,order=1)
            # print(i_12_min[0])
            if len(i_12_min[0])>1:
                d_O12_min[0,j,i]=d[i_12_min[0][0]]
                d_O12_min[1,j,i]=d[i_12_max[0][0]]
                d_O12_min[2,j,i]=d[i_12_min[0][1]]
            elif len(i_12_min[0])==0:
                search_order=2
                while len(i_12_min[0])==0 and search_order<5:
                    i_12_min=argrelmin(O12,order=search_order)
                    i_12_max=argrelmax(O12,order=search_order)
                    print(f'search order {search_order}')
                    print(i_12_min[0])
                    search_order+=1
                if len(i_12_min[0])==2:
                    d_O12_min[0,j,i]=d[i_12_min[0][0]]
                    d_O12_min[1,j,i]=d[i_12_max[0][0]]
                    d_O12_min[2,j,i]=d[i_12_min[0][1]]
                elif len(i_12_min[0])==1:
                    d_O12_min[0,j,i]=d[i_12_min[0][0]]
                    d_O12_min[1,j,i]=d[i_12_min[0][0]]
                    d_O12_min[2,j,i]=d[i_12_min[0][0]]
                else: 
                    print(f'Se encontraron mas minimos de lo esperado en O12 para x={x_i/g}g k={k_j/g}g...')
                    print(i_12_min[0])

            else:
                d_O12_min[0,j,i]=d[i_12_min[0][0]]
                d_O12_min[1,j,i]=d[i_12_min[0][0]]
                d_O12_min[2,j,i]=d[i_12_min[0][0]]

            O23=np.abs(Omega_n_ij(N_,2,3,d,g,k_j,J,x_i))
            i_23_max=argrelmax(O23,order=1)
            i_23_min=argrelmin(O23,order=1)
            # print(i_23_min)
            if len(i_23_min[0])>1:
                d_O23_min[0,j,i]=d[i_23_min[0][0]]
                d_O23_min[1,j,i]=d[i_23_max[0][0]]
                d_O23_min[2,j,i]=d[i_23_min[0][1]]
            elif len(i_23_min[0])==0:
                search_order=2
                while len(i_23_min[0])==0 and search_order<5:
                    i_23_min=argrelmin(O23,order=search_order)
                    i_23_max=argrelmax(O23,order=search_order)
                    search_order+=1
                if len(i_23_min[0])==2:
                    d_O23_min[0,j,i]=d[i_23_min[0][0]]
                    d_O23_min[1,j,i]=d[i_23_max[0][0]]
                    d_O23_min[2,j,i]=d[i_23_min[0][1]]
                elif len(i_23_min[0])==1:
                    d_O23_min[0,j,i]=d[i_23_min[0][0]]
                    d_O23_min[1,j,i]=d[i_23_min[0][0]]
                    d_O23_min[2,j,i]=d[i_23_min[0][0]]
                else: 
                    print(f'Se encontraron mas minimos de lo esperado en O23 para x={x_i/g}g k={k_j/g}g...')
                    print(i_23_min[0])
                    
            else:
                d_O23_min[0,j,i]=d[i_23_min[0][0]]
                d_O23_min[1,j,i]=d[i_23_min[0][0]]
                d_O23_min[2,j,i]=d[i_23_min[0][0]]
            
            O31=np.abs(Omega_n_ij(N_,3,1,d,g,k_j,J,x_i))
            i_31_min=argrelmin(O31,order=1)
            i_31_max=argrelmax(O31,order=1)
            if len(i_31_min[0])>1:
                d_O31_min[0,j,i]=d[i_31_min[0][0]]
                d_O31_min[1,j,i]=d[i_31_max[0][0]]
                d_O31_min[2,j,i]=d[i_31_min[0][1]]
            elif len(i_31_min[0])==0:
                search_order=2
                while len(i_31_min[0])==0 and search_order<5:
                    i_31_min=argrelmin(O31,order=search_order)
                    i_31_max=argrelmax(O31,order=search_order)
                    search_order+=1
                if len(i_31_min[0])==2:
                    d_O31_min[0,j,i]=d[i_31_min[0][0]]
                    d_O31_min[1,j,i]=d[i_31_max[0][0]]
                    d_O31_min[2,j,i]=d[i_31_min[0][1]]
                elif len(i_31_min[0])==1:
                    d_O31_min[0,j,i]=d[i_31_min[0][0]]
                    d_O31_min[1,j,i]=d[i_31_min[0][0]]
                    d_O31_min[2,j,i]=d[i_31_min[0][0]]
                else: 
                    print(f'Se encontraron mas minimos de lo esperado en O31 para x={x_i/g}g k={k_j/g}g...')
                    print(i_31_min[0])
            else:
                d_O31_min[0,j,i]=d[i_31_min[0][0]]
                d_O31_min[1,j,i]=d[i_31_min[0][0]]
                d_O31_min[2,j,i]=d[i_31_min[0][0]]

    # color_dimension = q_min # change to desired fourth dimension
    # minn, maxx = color_dimension.min(), color_dimension.max()
    # norm = mpl.colors.Normalize(minn, maxx)
    # m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    # m.set_array([])
    # fcolors = m.to_rgba(color_dimension)

    fig_12=plt.figure(figsize=(8,6))
    fig_12.suptitle(r'$\Omega^{(3)}_{12}$')
    ax_12=fig_12.add_subplot(111,projection='3d')
    ax_12.plot_surface(x_mesh/g,k_mesh/g,d_O12_min[0]/g,color='blue')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_12.plot_surface(x_mesh/g,k_mesh/g,d_O12_min[1]/g,color='red')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_12.plot_surface(x_mesh/g,k_mesh/g,d_O12_min[2]/g,color='green')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)

    fig_23=plt.figure(figsize=(8,6))
    fig_23.suptitle(r'$\Omega^{(3)}_{23}$')
    ax_23=fig_23.add_subplot(111,projection='3d')
    ax_23.plot_surface(x_mesh/g,k_mesh/g,d_O23_min[0]/g,color='blue')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_23.plot_surface(x_mesh/g,k_mesh/g,d_O23_min[1]/g,color='red')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_23.plot_surface(x_mesh/g,k_mesh/g,d_O23_min[2]/g,color='green')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)

    fig_31=plt.figure(figsize=(8,6))
    fig_31.suptitle(r'$\Omega^{(3)}_{31}$')
    ax_31=fig_31.add_subplot(111,projection='3d')
    ax_31.plot_surface(x_mesh/g,k_mesh/g,d_O31_min[0]/g,color='blue')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_31.plot_surface(x_mesh/g,k_mesh/g,d_O31_min[1]/g,color='red')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)
    ax_31.plot_surface(x_mesh/g,k_mesh/g,d_O31_min[2]/g,color='green')#,rstride=1, cstride=1, facecolor=fcolors,vmin=minn,vmax=maxx,shade=False)

    N=3

    ax_12.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-2)/g,color='black')
    ax_23.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-2)/g,color='black')
    ax_31.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-2)/g,color='black')

    ax_12.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-3)/g+2*k_mesh/g,color='black')
    ax_23.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-3)/g+2*k_mesh/g,color='black')
    ax_31.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-3)/g+2*k_mesh/g,color='black')

    ax_12.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-1)/g-2*k_mesh/g,color='black')
    ax_23.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-1)/g-2*k_mesh/g,color='black')
    ax_31.plot_surface(x_mesh/g,k_mesh/g,x_mesh*(2*N-1)/g-2*k_mesh/g,color='black')

    # cbar = fig_q.colorbar(m, ax=ax_q, shrink=0.5, aspect=5)
    # cbar.set_label('min(Q)')
    ax_12.set_xlabel(r'$\chi/g$')
    ax_12.set_ylabel(r'$k/g$')
    ax_12.set_zlabel(r'$\Delta/g$')

    ax_23.set_xlabel(r'$\chi/g$')
    ax_23.set_ylabel(r'$k/g$')
    ax_23.set_zlabel(r'$\Delta/g$')

    ax_31.set_xlabel(r'$\chi/g$')
    ax_31.set_ylabel(r'$k/g$')
    ax_31.set_zlabel(r'$\Delta/g$')

    plt.show()
    print('Finalizados los graficos de minima energia!')
    return None

## N=2 ----
# planos_minima_Omegaij(2)
## N=3 ----
# planos_minima_Omegaij(3)

# Dependencia FG ----

# cond_ini=1

# w_0=1
# g=0.01*w_0
# gamma=0.1*g
# p=0.005*g

# def ci(n:int,d,x,k,J):
#     if n==0:
#         psi0=(eg0+ge0).unit()
#         T=np.pi/(np.sqrt(2*g**2+(k-J+d/2-x/2)**2))
#     elif n==1:
#         psi0=(eg1+ge1).unit()
#         T=2*np.pi/np.abs(Omega_n_ij(2,1,2,d,g,k,J,x))
#     # elif n==2:
#     #     psi0=gg2
#     #     T=2*np.pi/Omega_n_ij(2,2,3,0,g,0,0,0)
#     # elif n==3:
#     #     psi0=tensor(gr,gr,basis(N_c,3))
#     #     T=2*np.pi/Omega_n_ij(2,2,3,0,g,0,0,0)
#     # elif n==4:
#     #     psi0=(eg2+ge2).unit()
#     #     T=2*np.pi/Omega_n_ij(2,2,3,0,g,0,0,0)
#     return psi0,T

## detuning ----

# x=0*g
# k=0
# J=0
# ciclos=15
# steps=300*ciclos


# delta_array=[0.00001*g,0.1*g,0.5*g,g,2*g,4*g,5*g,10*g]
# colors=mpl.colormaps['magma'](np.linspace(0,1,len(delta_array)+2))

# fig_fg=plt.figure(figsize=(8,6),dpi=120)
# ax_fg=fig_fg.add_subplot()
# # T0=2*np.pi/np.abs(Omega_n_ij(2,1,2,0,g,0,0,0))

# for i_delta,d in enumerate(delta_array):
#     psi0,T=ci(cond_ini,d,x,k,J)
#     # T=2*np.pi/np.abs(Omega_n_ij(2,1,2,d,g,k,J,x))
#     t_final=ciclos*T
            
#     t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#     sol_d=mesolve(H_tcm(g,d,x,k,J),psi0,t,c_ops=l_ops(gamma,p))

#     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#     ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_delta])

# ax_fg.set_xlabel(r'$t/T$')
# ax_fg.set_ylabel(r'$\phi_g/\pi$')
# ax_fg.set_xlim(0,ciclos)

# plt.grid()
# # fig_fg.savefig(rf'C:\Users\alima\Estudios\doc\doc latex\paper fg\figuras\dependencias\eg0+\detuning todo 0.png')

# plt.show()

# ## kerr ----

# for d in [0*g,0.5*g]:
#     k=0
#     J=0
#     ciclos=15
#     steps=300*ciclos

#     x_array=[0.00001*g,0.1*g,0.50001*g,g,2*g,4*g,5*g,10*g]
#     colors=mpl.colormaps['magma'](np.linspace(0,1,len(x_array)+2))
#     fig_fg=plt.figure(figsize=(8,6),dpi=120)
#     ax_fg=fig_fg.add_subplot()

#     for i_x,x in enumerate(x_array):
#         psi0,T=ci(cond_ini,d,x,k,J)
#         t_final=ciclos*T
                
#         t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#         sol_d=mesolve(H_tcm(g,d,x,k,J),psi0,t,c_ops=l_ops(gamma,p))

#         fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#         ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_x])
#     ax_fg.set_xlabel(r'$t/T$')
#     ax_fg.set_ylabel(r'$\phi_g/\pi$')
#     ax_fg.set_xlim(0,ciclos)

#     plt.grid()
#     # fig_fg.savefig(r'C:\Users\alima\Estudios\doc\doc latex\paper fg\figuras\dependencias\eg0+\kerr todo 0')
#     plt.show()

# ## dipole interaction ----

# x=0*g
# d=0
# J=0
# ciclos=15
# steps=300*ciclos

# k_array=[0.00001*g,0.1*g,0.5*g,g,2*g,4*g,5*g,10*g]
# colors=mpl.colormaps['magma'](np.linspace(0,1,len(k_array)+2))
# fig_fg=plt.figure(figsize=(8,6),dpi=120)
# ax_fg=fig_fg.add_subplot()

# for i_k,k in enumerate(k_array):
#     psi0,T=ci(cond_ini,d,x,k,J)    
#     t_final=ciclos*T
            
#     t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#     sol_d=mesolve(H_tcm(g,d,x,k,J),psi0,t,c_ops=l_ops(gamma,p))

#     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#     ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_k])

# ax_fg.set_xlabel(r'$t/T$')
# ax_fg.set_ylabel(r'$\phi_g/\pi$')
# ax_fg.set_xlim(0,ciclos)
# plt.grid()

# # fig_fg.savefig(rf'C:\Users\alima\Estudios\doc\doc latex\paper fg\figuras\dependencias\eg0+\interaccion todo 0')
# plt.show()

## acoplamiento ----

# d=0.1*g
# x=0
# k=0
# J=0
# ciclos=15
# steps=1000*ciclos

# gamma_array=[0,0.01*g,0.1*g,0.5*g,g]
# colors=mpl.colormaps['magma'](np.linspace(0,1,len(gamma_array)+2))
# fig_fg=plt.figure(figsize=(8,6),dpi=120)
# ax_fg=fig_fg.add_subplot()

# for i_gamma,gamma in enumerate(gamma_array):
#     psi0,T=ci(0,d,x,k,J)

#     t_final=ciclos*T
            
#     t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#     sol_d=mesolve(H_tcm(g,d,x,k,J),psi0,t,c_ops=l_ops(gamma,p=0))

#     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#     ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_gamma])
# ax_fg.set_xlabel(r'$t/T$')
# ax_fg.set_ylabel(r'$\phi_g/\pi$')
# ax_fg.set_xlim(0,ciclos)

# plt.grid()
# # fig_fg.savefig(r'C:\Users\alima\Estudios\doc\doc latex\paper fg\figuras\dependencias\eg0+\kerr todo 0')
# plt.show()

# #  BLOCH ----
# w_0=1
# g=0.01*w_0

#ESFERA DE BLOCH. LA IDEA ES ELEGIR ALGUNAS COMBINACIONES DE PARAMETROS PARA COMPARAR LAS TRAYECTORIAS 
#PARA LOS DIFERENTES CASOS.

# ciclos_bloch=40
# steps=3000*ciclos_bloch

# points=ciclos_bloch*50
# for p in [0.1*0.1*g]:
#     for gamma in [0.1*g]:
#         for x in [0*g]:
#             for tita in [0.*np.pi]:
#                 delta_array=[0.0000001*g]
#                 omega=np.sqrt(4*g**2+(0-x)**2)
#                 '''---Simulacion numerica---'''
#                 T=2*np.pi/omega
#                 t_final=40*T

#                 t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#                 fg_delta=np.zeros((len(delta_array),steps))
#                 fg_u_delta=np.zeros((len(delta_array),steps))
#                 fg_d_delta=np.zeros((len(delta_array),steps))

#                 vBloch_u_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#                 vBloch_d_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#                 vBloch_eigenvec_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))


#                 for i_delta,delta in enumerate(delta_array):
#                     print(f'delta #{i_delta}/{len(delta_array)}')
#                     # psi0=(e0+(1+1j)*g1).unit()#(np.sqrt(2+np.sqrt(2))/2*e0+1j*np.sqrt(2-np.sqrt(2))/2*g1).unit()
                    
#                     phi=0
#                     psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#                     H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)

#                     l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
                    
#                     sol_u=mesolve(H,psi0,t)
#                     sol_d=mesolve(H,psi0,t,c_ops=l_ops)

#                     fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#                     fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#                     fg_delta[i_delta]=fg_d-fg_u
#                     fg_d_delta[i_delta]=fg_d
#                     fg_u_delta[i_delta]=fg_u
#                 vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
#                 esfera1=Bloch()
#                 esfera1.make_sphere()
#                 colors=[mpl.colormaps['viridis'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['winter'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['magma'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))]

#                 esfera1.add_points(vBloch_u,'m',colors='black')
#                 esfera1.add_points(vBloch_eigevec,'m',colors=colors[1])
#                 esfera1.add_points(vBloch_d,'m',colors=colors[2])
#                 esfera1.render()
#                 # # esfera.save('bloch berry.png')
#                 esfera1.show()
#                 plt.show()

# FG con pocos delta para analizar casos ----

# fig_tau=plt.figure(figsize=(8,6))
# ax_tau=fig_tau.add_subplot()
# fig_fg=plt.figure(figsize=(8,6))
# ax_fg=fig_fg.add_subplot()
# esfera1=Bloch()
# esfera1.make_sphere()
# w_0=1
# g=0.01*w_0

# colors=['black','red','blue','green']
# i_color=0
# steps=3000*10
# # tita=0
# ciclos_bloch=10
# points=ciclos_bloch*50
# for p in [0.1*0.1*g]:
#     for gamma in [0.1*g]:
#         for x in [0*g]:
#             # for steps in [160,1600]:
#             delta_array=[2*g]
#             omega=np.sqrt(4*g**2+(delta_array[0]-x)**2)
#             '''---Simulacion numerica---'''
#             T=2*np.pi/omega
#             t_final=10*T
             
#             t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#             fg_delta=np.zeros((len(delta_array),steps))
#             fg_u_delta=np.zeros((len(delta_array),steps))
#             fg_d_delta=np.zeros((len(delta_array),steps))
#             # print(fg_delta[0])
#             N_u_delta=np.zeros((len(delta_array),steps))
#             N_d_delta=np.zeros((len(delta_array),steps))

#             vBloch_u_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#             vBloch_d_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))
#             vBloch_eigenvec_delta=np.zeros((3,len(delta_array),int(steps*ciclos_bloch*T/t_final/points)))

#             eigvals_death_t=np.full(len(delta_array),-1)
#             # eigvals_death_z=np.full(len(delta_array),0)
#             # negativity_revival_t=np.full(len(delta_array),-1)
#             # negativity_revival_z=np.full(len(delta_array),0)

#             for i_delta,delta in enumerate(delta_array):
#                 tita_rob=np.arctan2(delta-x,-2*g)
#                 tita=tita_rob
#                 print(f'{tita/np.pi}pi') 
#                 phi=0
#                 psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
#                 H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
#                 # print(psi0)
#                 # print(H*psi0)
#                 l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
                
#                 sol_u=mesolve(H,psi0,t)
#                 sol_d=mesolve(H,psi0,t,c_ops=l_ops)

#                 fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#                 fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

#                 fg_delta[i_delta]=fg_d-fg_u
#                 fg_d_delta[i_delta]=fg_d
#                 fg_u_delta[i_delta]=fg_u
#                 vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
#                 vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
#                 esfera1.add_points(vBloch_u,'m',colors='black')
#                 esfera1.add_points(vBloch_eigevec,'m',colors='red')
#                 esfera1.add_points(vBloch_d,'m',colors='yellow')

#                 label=fr'$\gamma={gamma/g :.2f}g, p={p/g :.2f}g$'
#                 dfg=fg_d-fg_u


#                 ax_fg.plot(t/T,fg_u/np.pi,color=colors[i_color],label=f'u')
#                 ax_fg.plot(t/T,fg_d/np.pi,color=colors[i_color],linestyle='dashed',label=f'd')
#                 # ax_fg.scatter(t/T,dfg,color=colors[i_color],label=f'd-u')
#                 # ax_fg.vlines(t_saltos,-2,2)
#                 # plt.show()

#                 # index_saltos=[]
#                 # t_saltos=[]

#                 # for i_tau in range(len(dfg)-1):
#                 #     if np.abs(dfg[i_tau+1]-dfg[i_tau])>3:
#                 #         index_saltos.append(i_tau)
#                 #         t_saltos.append(t[i_tau]/T)
#                 # dif_tiempo_salto=[]
#                 # for i in range(0,len(t_saltos)-2,2):
#                 #     dif_tiempo_salto.append(t_saltos[i+1]-t_saltos[i])
#                 # ax_tau.plot(dif_tiempo_salto,color=colors[i_color])
#                 # ax_tau.scatter(range(len(dif_tiempo_salto)),dif_tiempo_salto,color=colors[i_color],label=label)
#                 # i_color+=1
# esfera1.render()
# esfera1.show()
# # ax_tau.legend()

# ax_fg.legend()

# plt.show()

#  Graficos funcionables ----
'''-------------------------- GRAFICOS FUNCIONABLES --------------------------'''

# gamma=0.1*g
# p=0.1*0.1*g

# points=15000
# x=0*g
# delta=0*g
# # psi0=(e0+(1+1j)*g1).unit()#(np.sqrt(2+np.sqrt(2))/2*e0+1j*np.sqrt(2-np.sqrt(2))/2*g1).unit()
# tita=0
# phi=0
# psi0=np.cos(tita/2)*e0+np.exp(1j*phi)*np.sin(tita/2)*g1
# H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
# omega=np.sqrt(4*g**2+(delta-x)**2)
# '''---Simulacion numerica---'''
# T=2*np.pi/omega
# t_final=70*T
# steps=15000
# ciclos_bloch=70
# colors=[mpl.colormaps['viridis'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['winter'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points))))),mpl.colormaps['magma'](np.linspace(0,1,len(range(0,int(steps*ciclos_bloch*T/t_final),int(steps*ciclos_bloch*T/t_final/points)))))]

## Bloch unit puntero disip FG ----
'''-------------------------    BLOCH UNIT PUNTERO DISIP FG --------------------------'''

# fig_e=plt.figure(figsize=(8,6))

# ax_n=fig_e.add_subplot(321)
# ax_n_zoom=fig_e.add_subplot(322)
# ax_n.set_xlabel('$t/T$')
# ax_n.set_ylabel(r'$\mathcal{N}(\rho)$')
# # ax_e.set_ylabel(r'$E(\rho)$')
# ax_fg=fig_e.add_subplot(323,sharex=ax_n)
# ax_fg_zoom=fig_e.add_subplot(324,sharex=ax_n_zoom)

# ax_e=fig_e.add_subplot(325,sharex=ax_fg)
# ax_e_zoom=fig_e.add_subplot(326,sharex=ax_fg_zoom)
# ax_e.set_xlabel('$t/T$')
# ax_e.set_ylabel(r'$E(\rho)$')


# colors_fg=['blue','black','red']
# labels_fg=['u','d','d+']

# l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sm] #operadores de colapso/lindblad
# t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 


# sol_u=mesolve(H,psi0,t)
# sol_d=mesolve(H,psi0,t,c_ops=l_ops)

# fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
# fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)

# N_u=np.array([negativity_hor(sol_u.states[i],[0,1]) for i in range(len(sol_u.states))])
# N_d=np.array([negativity_hor(sol_d.states[i],[0,1]) for i in range(len(sol_d.states))])

# # C_u=concurrence_ali(sol_u.states)
# # C_d=concurrence_ali(sol_d.states)

# vBloch_u=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
# vBloch_eigevec=vectorBloch(e0,g1,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
# vBloch_d=vectorBloch(e0,g1,sol_d.states,steps,ciclos_bloch,T,t_final,points)
# esfera1=Bloch()
# esfera1.make_sphere()

# esfera1.add_points(vBloch_u,'m',colors='black')
# esfera1.add_points(vBloch_eigevec,'m',colors=colors[1])
# esfera1.add_points(vBloch_d,'m',colors=colors[2])

# # vBloch_u=vectorBloch(e1,g2,sol_u.states,steps,ciclos_bloch,T,t_final,points)
# # vBloch_eigevec=vectorBloch(e1,g2,psi_eig_d,steps,ciclos_bloch,T,t_final,points)
# # vBloch_d=vectorBloch(e1,g2,sol_d.states,steps,ciclos_bloch,T,t_final,points)
# esfera1.render()
# # esfera.save('bloch berry.png')
# esfera1.show()

# # esfera2=Bloch()
# # esfera2.make_sphere()

# # esfera2.add_points(vBloch_u,'m',colors='black')
# # esfera2.add_points(vBloch_eigevec,'m',colors=colors[1])
# # esfera2.add_points(vBloch_d,'m',colors=colors[2])
# # esfera2.render()
# # # esfera.save('bloch berry.png')
# # esfera2.show()

# zoom_steps=steps
# colors_e=mpl.colormaps['hot'](np.linspace(0,1,len(eigenvals_t_d[0])))
# for i1 in range(len(eigenvals_t_d[0])): 
#     if i1==2:
#         max_eig2=np.max(eigenvals_t_d[:,i1])
#         eigenvals_t_d[:,i1]=eigenvals_t_d[:,i1]/max_eig2
#         ax_e.text(0.6*t_final/T,0.75,"x{0:.2E}".format(max_eig2),color=colors_e[i1])
#     ax_e.plot(t/T,eigenvals_t_d[:,i1],color=colors_e[i1])
#     ax_e_zoom.plot(t[:zoom_steps]/T,eigenvals_t_d[:zoom_steps,i1],color=colors_e[i1])

# ax_n.plot(t/T,N_u,color='red',linestyle='dashed',label='N_u')
# ax_n.plot(t/T,N_d,color='green',linestyle='dashed',label='N_d')

# ax_n_zoom.plot(t[:zoom_steps]/T,N_u[:zoom_steps],color='red',linestyle='dashed',label='N_u')
# ax_n_zoom.plot(t[:zoom_steps]/T,N_d[:zoom_steps],color='green',linestyle='dashed',label='N_d')

# ax_fg.plot(t/T,fg_u,color=colors_fg[0],label=labels_fg[0])
# ax_fg.plot(t/T,fg_d,color=colors_fg[1],label=labels_fg[1])

# ax_fg_zoom.plot(t[:zoom_steps]/T,fg_u[:zoom_steps],color=colors_fg[0],label=labels_fg[0])
# ax_fg_zoom.plot(t[:zoom_steps]/T,fg_d[:zoom_steps],color=colors_fg[1],label=labels_fg[1])

# ax_fg.set_xlabel('$t/T$')
# ax_fg.set_ylabel(r'$\phi_g$')
# ax_fg.legend()
# ax_e.legend()
# plt.show()

# Simulacion simple FG ----

# gamma=0.1*g
# p=0.1*0.1*g
# x=2*g
# d=20/3*g

# k=0
# J=0


# points=15000
# T=2*np.pi/np.abs(rabi_freq(2,1,2,d,g,k,J,x))
# t_final=10*T

# t=np.linspace(0,t_final,points)

# rho_0=(eg1+ge1).unit()
# H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

# sol_u=mesolve(H,rho_0,t,c_ops=[])
# sol_d=mesolve(H,rho_0,t,c_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2])
# fg_u,arg,eigenvals_t_d,psi_eig_u = fases(sol_u)
# fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)
# fig_fg=plt.figure(figsize=(8,6),dpi=100)
# ax_fg=fig_fg.add_subplot()
# ax_fg.plot(t/T,fg_d/np.pi,color='red')
# ax_fg.plot(t/T,(fg_d-fg_u)/np.pi,color='red',linestyle='dashed')
# ax_fg.plot(t/T,fg_u/np.pi,color='black')
# plt.show()

#Condiciones iniciales tita N=1 ----
'''------------------   CONDICIONES INICIALES TITA  ---------------------------------------------------------'''
#ESFERA DE BLOCH Y HEATPLOT DE NEGATIVIDAD. BARREMOS EN UN ANGULO TITA (DE CONDICION INICIAL). MIRAMOS 1 CICLO DE EVOLUCION PARA CADA CONDICION INICIAL

# def heatplot(t,y,z_data:list,title:str,ylabel):
#     fig_u=plt.figure(figsize=(8,6))
#     fig_u.suptitle(title)
#     ax_u=fig_u.add_subplot()
#     ax_u.set_xlabel('$t/T$')
#     ax_u.set_ylabel(ylabel)
#     c0 = ax_u.pcolor(t/T, y, z_data, shading='auto', cmap='jet',vmin=0,vmax=0.5)
#     contour_u = ax_u.contourf(t/T, y, z_data,levels=[0,0.01],colors='black',linewidths=1)
#     ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
#     fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
#     # fig_u.savefig(rf'graficos\negativity\{psi0Name} {title} x={x/g}g k={k/g}g J={J/g}g neg delta dis.png')

# Parametros y Hamiltoniano
# w_0=1
# g=0.01*w_0
# gamma=0.1*g
# p=0.01*g

# x=0*g
# delta=0*g
# k=0
# J=0

# H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
# T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))

# # # Simulacion numerica
# num_ciclos=3
# t_final=num_ciclos*T
# steps=3000*num_ciclos
# num_tita=50
# tita_array=np.linspace(0,np.pi/2,num_tita)

# # #definimos los arrays de negatividad
# robustez_phi_p0=np.zeros((num_tita,steps))
# robustez_phi_p1=np.zeros((num_tita,steps))

# N_0=np.zeros((num_tita,steps))
# N_1=np.zeros((num_tita,steps))
# N_2=np.zeros((num_tita,steps))

# N_01=np.zeros((num_tita,steps))
# N_02=np.zeros((num_tita,steps))
# N_12=np.zeros((num_tita,steps))

# for j_tita,tita in enumerate(tita_array):
#     psi0=np.cos(tita/2)*(eg0+ge0).unit()+np.sin(tita/2)*gg1 
#     t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#     sol_u=mesolve(H,psi0,t)
#     sol_p=mesolve(H,psi0,t,c_ops=l_ops(gamma,p))
    
#     fg_u,arg,eigenvals_t_d,psi_eig_u = fases(sol_u)
#     fg_p,arg,eigenvals_t_d,psi_eig_d = fases(sol_p)

#     N_0[j_tita]=np.array([negativity_hor(sol_p.states[j],[1,0,0]) for j in range(len(sol_u.states))])
#     N_1[j_tita]=np.array([negativity_hor(sol_p.states[j],[0,1,0]) for j in range(len(sol_u.states))])
#     N_2[j_tita]=np.array([negativity_hor(sol_p.states[j],[0,0,1]) for j in range(len(sol_u.states))])

#     N_01[j_tita]=np.array([negativity_hor(sol_p.states[j].ptrace([0,1]),[1,0]) for j in range(len(sol_u.states))])
#     N_02[j_tita]=np.array([negativity_hor(sol_p.states[j].ptrace([0,2]),[1,0]) for j in range(len(sol_u.states))])
#     N_12[j_tita]=np.array([negativity_hor(sol_p.states[j].ptrace([1,2]),[1,0]) for j in range(len(sol_u.states))])
  
#     robustez_phi_p0[j_tita]=np.abs(fg_p)-np.abs(fg_u)
#     print(j_tita)
# np.savetxt(f'robusteces/tcm/N=1 tita 3t p={p/g}g',robustez_phi_p0[:,steps-1]/np.pi)
# np.savetxt(f'robusteces/tcm/N=1 tita 2t p={p/g}g',robustez_phi_p0[:,int(steps*2/3)]/np.pi)
# np.savetxt(f'robusteces/tcm/N=1 tita 1t p={p/g}g',robustez_phi_p0[:,int(steps/3)]/np.pi)

# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N0.txt',N_0)
# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N1.txt',N_1)
# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N2.txt',N_2)
# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N01.txt',N_01)
# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N02.txt',N_02)
# np.savetxt(f'robusteces/tcm/negatividad tita N=1 N12.txt',N_12)

# Condicion inicial tita N>1 ----

#Parametros y Hamiltoniano
# w_0=1
# g=0.01*w_0
# gamma=0.1*g
# p=0.01*g

# x=0*g
# delta=0*g
# k=0
# J=0

# H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
# T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))

# # # Simulacion numerica
# num_ciclos=3
# t_final=num_ciclos*T
# steps=3000*num_ciclos
# num_tita=50
# num_varphi=50

# tita_array=np.linspace(0,np.pi,num_tita)
# varphi_array=np.linspace(0,2*np.pi,num_varphi)
# # #definimos los arrays de negatividad
# robustez_phi_3t=np.zeros((num_tita,num_varphi))
# robustez_phi_2t=np.zeros((num_tita,num_varphi))
# robustez_phi_1t=np.zeros((num_tita,num_varphi))

# for j_t,tita in enumerate(tita_array):
#     for j_vp,varphi in enumerate(varphi_array):
#         psi0=np.cos(varphi)*np.sin(tita)*(ee0)+np.sin(varphi)*np.sin(tita)*(eg1+ge1).unit()+np.cos(tita)*(gg2)
#         t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

#         sol_u=mesolve(H,psi0,t)
#         sol_p=mesolve(H,psi0,t,c_ops=l_ops(gamma,p))
        
#         fg_u,arg,eigenvals_t_d,psi_eig_u = fases(sol_u)
#         fg_p,arg,eigenvals_t_d,psi_eig_d = fases(sol_p)

#         robustez_phi_3t[j_t,j_vp]=np.abs(fg_p[steps-1])-np.abs(fg_u[steps-1])
#         robustez_phi_2t[j_t,j_vp]=np.abs(fg_p[int(steps*2/3)])-np.abs(fg_u[int(steps*2/3)])
#         robustez_phi_1t[j_t,j_vp]=np.abs(fg_p[int(steps/3)])-np.abs(fg_u[int(steps/3)])
#         print(j_t*num_varphi+j_vp+1,'/',num_tita*num_varphi,flush=True)
# np.savetxt(f'robusteces/tcm/N=2 tita_varphi 3t.txt',robustez_phi_3t/np.pi)
# np.savetxt(f'robusteces/tcm/N=2 tita_varphi 2t.txt',robustez_phi_2t/np.pi)
# np.savetxt(f'robusteces/tcm/N=2 tita_varphi 1t.txt',robustez_phi_1t/np.pi)

# # Leer tita_varphi ----

# print(np.ones((5,2)))
# tita_varphi_data3T=np.loadtxt(f'robusteces/tcm/N=2 tita_varphi 3t.txt')
# tita_varphi_data2T=np.loadtxt(f'robusteces/tcm/N=2 tita_varphi 2t.txt')
# tita_varphi_data1T=np.loadtxt(f'robusteces/tcm/N=2 tita_varphi 1t.txt')
# num_tita=50
# num_varphi=50
# tita_array=np.linspace(0,np.pi,num_tita)
# varphi_array=np.linspace(0,2*np.pi,num_varphi)

# fig=plt.figure(figsize=(8,6),dpi=120)
# ax=fig.add_subplot()
# ax.set_ylabel(r'$\theta/\pi$')
# ax.set_xlabel(r'$\varphi/\pi$')
# c0 = ax.pcolor(varphi_array/np.pi, tita_array/np.pi, tita_varphi_data1T, shading='auto', cmap='jet')
# contour_u = ax.contourf(varphi_array/np.pi, tita_array/np.pi, tita_varphi_data1T,levels=[0,0.01],colors='black',linewidths=1)
# fig.colorbar(c0, ax=ax)
# plt.show()

# Condicion inicial perpendicular N=1 ----
'''------- BARRIDA DELTA ; CONDICION INICIAL PERPENDICULAR A LA DIRECCION DE ROTACION DEL HAMILTONIANO --------'''
#EL HAMILTONIANO DE 2X2 DEFINE UNA DIRECCION n=(g*sqrt(n),0,delta/2). PONEMOS LA CONDICION INICIAL QUE SEA 
#PERPENDICULAR A ESTA DIRECCION CALCULANDO EL ANGULO THETA DE LA DIRECCION n, SABIENDO QUE EL ANGULO POLAR
#phi=0 PORQUE EN n_y=0, DE ESTA MANERA LA TRAYECTORIA UNITARIA ES SIEMPRE POR UNA GEODESICA INDEPENDIENTEMENTE
#DE LA CONDICION DE RESONANCIA

def heatplot(t,y,z_data:list,title:str,ylabel):
    fig_u=plt.figure(figsize=(8,6))
    fig_u.suptitle(title)
    ax_u=fig_u.add_subplot()
    ax_u.set_xlabel('$t/T$')
    ax_u.set_ylabel(ylabel)
    c0 = ax_u.pcolor(t/T, y, z_data, shading='auto', cmap='jet',vmin=0,vmax=0.5)
    contour_u = ax_u.contourf(t/T, y, z_data,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    # fig_u.savefig(rf'graficos\negativity\{psi0Name} {title} x={x/g}g k={k/g}g J={J/g}g neg delta dis.png')


# w_0=1
# g=0.01*w_0
# gamma=0.1*g
# p=0.01*g

# k=0
# J=0
# ciclos=6
# color_trayectorias=[['darkred','red','lightcoral'],['navy','blue','cyan'],['darkgreen','limegreen','lime']]

# for x in [0]:
#     steps=3000*6
#     delta_array=np.linspace(-10*g,10*g,201)

#     fg_d_delta=np.zeros((len(delta_array),steps))
#     fg_u_delta=np.zeros((len(delta_array),steps))
#     N_0=np.zeros((len(delta_array),steps))
#     # N_1=np.zeros((len(delta_array),steps))
#     N_2=np.zeros((len(delta_array),steps))

#     N_01=np.zeros((len(delta_array),steps))
#     N_02=np.zeros((len(delta_array),steps))
#     # N_12=np.zeros((len(delta_array),steps))


#     omega0=2*2*g
#     T0=2*np.pi/omega0
#     esfera1=Bloch()
#     esfera1.make_sphere()
#     i_color=0
#     for i_delta,delta in enumerate(delta_array):
#         print(i_delta)
#         # T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))
#         t_final=ciclos*T0
            
#         t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
#         H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
#         H_evalv,H_evecs=H.eigenstates(sort='low') #'high') # supongo que high conviene si tenemos N>1 y low si N=1
#         # print(H_evalv)
#         succesfull_counts=0
#         indice_evecs=[0,0]
#         for i in range(len(H_evecs)):
#             # print(H_evecs[i])
#             if H_evecs[i].dag().tidyup()*(gg1+(eg0+ge0).unit())!=0: # or H_evecs[i].dag()*(eg1+ge1).unit()!=0 or H_evecs[i].dag()*ee0!=0:
#                 indice_evecs[succesfull_counts]=i
#                 succesfull_counts+=1
#             if succesfull_counts==2:
#                 break
#         psi0=(H_evecs[indice_evecs[0]]+H_evecs[indice_evecs[1]]).unit()
#         # print(psi0)
#         sol_u=mesolve(H,psi0,t)
#         sol_d=mesolve(H,psi0,t,c_ops=l_ops(gamma,p))


#         N_0[i_delta]=np.array([negativity_hor(sol_d.states[j],[1,0,0]) for j in range(len(sol_u.states))])
#         # N_1[i_delta]=np.array([negativity_hor(sol_d.states[j],[0,1,0]) for j in range(len(sol_u.states))])
#         N_2[i_delta]=np.array([negativity_hor(sol_d.states[j],[0,0,1]) for j in range(len(sol_u.states))])

#         N_01[i_delta]=np.array([negativity_hor(sol_d.states[j].ptrace([0,1]),[1,0]) for j in range(len(sol_u.states))])
#         N_02[i_delta]=np.array([negativity_hor(sol_d.states[j].ptrace([0,2]),[1,0]) for j in range(len(sol_u.states))])
#         # N_12[i_delta]=np.array([negativity_hor(sol_d.states[j].ptrace([1,2]),[1,0]) for j in range(len(sol_u.states))])
        
#         fg_u,arg,eigenvals_t_u,psi_eig_u = fases(sol_u)
#         fg_d,arg,eigenvals_t_d,psi_eig_d = fases(sol_d)
#         if i_delta in [0,50,100]:
            
#             # v_bloch_u=vectorBloch(H_evecs[indice_evecs[0]],H_evecs[indice_evecs[1]],sol_u.states,steps,ciclos,T,t_final,30*ciclos)
#             # v_bloch_d=vectorBloch(H_evecs[indice_evecs[0]],H_evecs[indice_evecs[1]],sol_d.states,steps,ciclos,T,t_final,30*ciclos)
#             # v_bloch_evec=vectorBloch(H_evecs[indice_evecs[0]],H_evecs[indice_evecs[1]],psi_eig_d,steps,ciclos,T,t_final,30*ciclos)
            
#             v_bloch_u=vectorBloch((eg0+ge0).unit(),gg1,sol_u.states,steps,ciclos,T0,t_final,30*ciclos)
#             v_bloch_d=vectorBloch((eg0+ge0).unit(),gg1,sol_d.states,steps,ciclos,T0,t_final,30*ciclos)
#             v_bloch_evec=vectorBloch((eg0+ge0).unit(),gg1,psi_eig_d,steps,ciclos,T0,t_final,30*ciclos)
            

#             esfera1.add_points(v_bloch_u,'m',colors=color_trayectorias[i_color][0])
#             esfera1.add_points(v_bloch_evec,'m',colors=color_trayectorias[i_color][1])
#             esfera1.add_points(v_bloch_d,'m',colors=color_trayectorias[i_color][2])
#             i_color+=1

#         fg_d_delta[i_delta]=np.abs(fg_d)-np.abs(fg_u)

#     esfera1.save(f'trayectorias perpendiculares n1 x={x/g}g.png')
#     esfera1.show()
#     plt.show()
#     esfera1.clear()

#     np.savetxt(f'robusteces/tcm/perpendicular N=1 3t0 x={x/g}g.txt',fg_d_delta[:,steps-1]/np.pi)
#     np.savetxt(f'robusteces/tcm/perpendicular N=1 2t0 x={x/g}g.txt',fg_d_delta[:,int(steps*2/3)]/np.pi)
#     np.savetxt(f'robusteces/tcm/perpendicular N=1 1t0 x={x/g}g.txt',fg_d_delta[:,int(steps/3)]/np.pi)

#     np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N0 x={x/g}g.txt',N_0)
#     # np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N1 x={x/g}g.txt',N_1)
#     np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N2 x={x/g}g.txt',N_2)
#     np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N01 x={x/g}g.txt',N_01)
#     np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N02 x={x/g}g.txt',N_02)
#     # np.savetxt(f'robusteces/tcm/negatividad N=1 t0 perp N12 x={x/g}g.txt',N_12)
   
# leer condicion inicial perpendicular (robustez y negatividad) ----

# w_0=1
# x=0
# g=0.01*w_0
# delta_array=np.linspace(-10*g,10*g,201)

# robustez_fg_3T=np.loadtxt(f'robusteces/tcm/perpendicular N=1 3t0 x={x/g}g.txt')%2
# robustez_fg_2T=np.loadtxt(f'robusteces/tcm/perpendicular N=1 2t0 x={x/g}g.txt')%2
# robustez_fg_1T=np.loadtxt(f'robusteces/tcm/perpendicular N=1 1t0 x={x/g}g.txt')%2
# fig_rob=plt.figure(figsize=(8,6),dpi=120)
# ax_rob=fig_rob.add_subplot()
# ax_rob.plot(delta_array,robustez_fg_3T,color='black',label=r'$3T_0$')
# ax_rob.plot(delta_array,robustez_fg_2T,color='black',linestyle='dashed',label=r'$2T_0$')
# ax_rob.plot(delta_array,robustez_fg_1T,color='black',linestyle='dashdot',label=r'$1T_0$')

# N0=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N0 x={x/g}g.txt')%2
# # N1=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N1 x={x/g}g.txt')
# N2=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N2 x={x/g}g.txt')%2

# N01=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N01 x={x/g}g.txt')%2
# N02=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N02 x={x/g}g.txt')%2
# # N12=np.loadtxt(f'robusteces/tcm/negatividad N=1 t0 perp N12 x={x/g}g.txt')

# fig_neg=plt.figure(figsize=(8,6),dpi=120)
# ax_N0=fig_neg.add_subplot(221)
# # ax_N1=fig_neg.add_subplot(322)
# ax_N2=fig_neg.add_subplot(222)

# ax_N01=fig_neg.add_subplot(223)
# ax_N02=fig_neg.add_subplot(224)
# # ax_N12=fig_neg.add_subplot(326)

# ciclos=10
# omega0=4*g
# T0=np.pi/omega0
# t_final=ciclos*T0
# steps=3000*6
# t=np.linspace(0,t_final,steps)
# c0 = ax_N0.pcolor(t/T0, delta_array, N0, shading='auto', cmap='Reds',vmin=0,vmax=1)
# # ax_N1.pcolor(t/T, delta_array, N1, shading='auto', cmap='Reds',vmin=0,vmax=1)
# ax_N2.pcolor(t/T0, delta_array, N2, shading='auto', cmap='Reds',vmin=0,vmax=1)

# ax_N01.pcolor(t/T0, delta_array, N01, shading='auto', cmap='Reds',vmin=0,vmax=1)
# ax_N02.pcolor(t/T0, delta_array, N02, shading='auto', cmap='Reds',vmin=0,vmax=1)
# # ax_N12.pcolor(t/T, delta_array, N12, shading='auto', cmap='Reds',vmin=0,vmax=1)

# fig_neg.colorbar(c0, ax=[ax_N0,ax_N2,ax_N01,ax_N02])
# plt.show()

# Chequeo estados perpendiculares ----
'''-------- CHEQUEO ESTADOS PERPENDICULARES AL HAMILTONIANO EN ESFERA DE BLOCH -------'''

# esfera=Bloch()
# esfera.make_sphere()
# delta=0
# x=0
# # delta_array=np.linspace(-10*g,10*g,201)
# tita=np.arctan2(delta-x,2*g)
# # h_vec=[[g]*len(delta_array),[0]*len(delta_array),delta_array/2-[x/2]*len(delta_array)]
# h_vec=[g,0,delta/2-x/2]
# h_vec=h_vec/np.sqrt(np.sum(h_vec_i**2 for h_vec_i in h_vec)) 
# esfera.add_vectors(h_vec,colors='black')

# phi=0
# psi_perp=np.cos(tita/2)*e0+np.sin(tita/2)*g1
# esfera.add_vectors(ket_to_bloch(e0,g1,psi_perp),colors='pink')
# H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)
# evals,ekets=H.eigenstates(phase_fix=0)
# colors_map=mpl.colormaps['viridis'](np.linspace(0,1,len(evals)))

# for iket,kets in enumerate(ekets):
#     esfera.add_points(ket_to_bloch(e0,g1,kets),'s',colors=colors_map[iket])
                
# esfera.render()
# esfera.show()
# plt.show()

# Graficos paper direcciones bloch ----
'''---------- GRAFICOS PARA PAPER CON DIRECCIONES Y SIMULACIONES ------------'''
# esfera=Bloch()
# esfera.make_sphere()
# delta=g
# x=0
# gamma=0.1*g
# p=0.01*g
# p0=0
# p1=0.01*g
# omega=np.sqrt(4*g**2+(delta-x)**2)

# # # Simulacion numerica
# num_ciclos=10
# steps=3000*num_ciclos

# T=2*np.pi/omega
# t_final=num_ciclos*T
# t=np.linspace(0,t_final,steps)
# # delta_array=np.linspace(-10*g,10*g,201)
# tita=np.arctan2(delta-x,-2*g)
# # h_vec=[[g]*len(delta_array),[0]*len(delta_array),delta_array/2-[x/2]*len(delta_array)]
# h_vec=[g,0,delta/2-x/2]
# h_vec=h_vec/np.sqrt(np.sum(h_vec_i**2 for h_vec_i in h_vec)) 
# esfera.add_vectors(h_vec,colors='black')

# phi=0
# psi_perp=np.cos(tita/2)*e0+np.sin(tita/2)*g1
# esfera.add_vectors(ket_to_bloch(e0,g1,psi_perp),colors='pink')
# H=x*a.dag()*a*a.dag()*a+delta/2*sz + g*(a.dag()*sm+a*sp)

# l_ops0=[np.sqrt(gamma)*a,np.sqrt(p0)*sm] #operadores de colapso/lindblad
# l_ops1=[np.sqrt(gamma)*a,np.sqrt(p1)*sm] #operadores de colapso/lindblad

# sol_u=mesolve(H,psi_perp,t)
# sol_p0=mesolve(H,psi_perp,t,c_ops=l_ops0)
# # sol_p1=mesolve(H,psi_perp,t,c_ops=l_ops1)
# fg_p0,arg,eigenvals_t_d,psi_eig_p0 = fases(sol_p0)
# # fg_p1,arg,eigenvals_t_d,psi_eig_p1 = fases(sol_p1)

# ciclos_bloch=num_ciclos
# points=ciclos_bloch*50

# vBloch_tita=vectorBloch(e0,g1,sol_u.states,steps,ciclos_bloch,T,t_final,points)
# esfera.add_points(vBloch_tita,'s',colors='black')

# vBloch_tita=vectorBloch(e0,g1,sol_p0.states,steps,ciclos_bloch,T,t_final,points)
# esfera.add_points(vBloch_tita,'s',colors='lightblue')

# vBloch_tita=vectorBloch(e0,g1,psi_eig_p0,steps,ciclos_bloch,T,t_final,points)
# esfera.add_points(vBloch_tita,'s',colors='blue')

# esfera.render()
# esfera.show()
# plt.show()

# Negatividad N=1 ----


# NEGATIVIDAD N>=2 ----

# 1 CORRER ----

# N_c=3
# steps=3000*4


# w0=1
# g=0.01*w0

# gamma=0.1*g
# p=0.05*gamma

# J=0

# T0=2*np.pi/np.abs(Omega_n_ij(2,1,2,0,g,0,0,0))
# t_final=10*T0
# t=np.linspace(0,t_final,steps)


# delta=np.linspace(-15*g,15*g,251)


# N_0=np.zeros((len(delta),steps))
# N_1=np.zeros((len(delta),steps))
# N_2=np.zeros((len(delta),steps))

# N_01=np.zeros((len(delta),steps))
# N_02=np.zeros((len(delta),steps))
# N_12=np.zeros((len(delta),steps))


# # concu_d=np.zeros((len(delta),steps))
# for rho_0,rho_0Name in zip([(eg1+ge1).unit(),(eg1+ge1).unit(),gg1],['eg0','eg1','gg1']):
#     for x in [0,2*g]:
#         for k in [0]:
#             for i,d in enumerate(delta):
#                 print(f'x={x/g}g, k={k/g}g, {i}/{len(delta)} -> {i*100/len(delta)}%',flush=True)


#                 H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2


#                 sol=mesolve(H,rho_0,t,c_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2])

#                 # fg_total,arg_tot,eigenvals_tot_t,psi_puntero=fases(sol)


#                 for j in range(len(sol.states)):
#                     N_0[i,j]=negativity_hor(sol.states[j],[1,0,0])
#                     # N_1[i,j]=negativity_hor(sol.states[j],[0,1,0])
#                     N_2[i,j]=negativity_hor(sol.states[j],[0,0,1])

#                     N_01[i,j]=negativity_hor(sol.states[j].ptrace([0,1]),[1,0])
#                     N_02[i,j]=negativity_hor(sol.states[j].ptrace([0,2]),[1,0])
#                     # N_12[i,j]=negativity_hor(sol.states[j].ptrace([1,2]),[1,0])
                    
#                 if gamma=='00' and p==0:
#                     np.savetxt(f'negativity/N0 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_0)
#                     # np.savetxt(f'negativity/N1 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_1)
#                     np.savetxt(f'negativity/N2 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_2)

#                     np.savetxt(f'negativity/N01 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_01)
#                     np.savetxt(f'negativity/N02 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_02)
#                     # np.savetxt(f'negativity/N12 unit {rho_0Name} x={x/g}g k={k/g}g.txt',N_12)
#                 else:
#                     np.savetxt(f'negativity/N0 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_0)
#                     # np.savetxt(f'negativity/N1 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_1)
#                     np.savetxt(f'negativity/N2 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_2)

#                     np.savetxt(f'negativity/N01 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_01)
#                     np.savetxt(f'negativity/N02 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_02)
#                     # np.savetxt(f'negativity/N12 {rho_0Name} gamma={gamma/g}g x={x/g}g k={k/g}g.txt',N_12)
#                 with open(rf'negativity/{rho_0Name} limites tcm.txt','a') as limites_file:
#                     limites_file.write('\n')
#                     limites_file.write(f'{k/g},{x/g},{J/g},{delta[0]/g},{delta[-1]/g},{len(delta)}')


# 2 LEER una combinacion de parametros ---- 

g=0.01
k=0*g
x=2*g

rho_0Name='eg1'
N0=np.loadtxt(f'negativity/N0 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')
# N1=np.loadtxt(f'negativity/N1 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')
N2=np.loadtxt(f'negativity/N2 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')

N01=np.loadtxt(f'negativity/N01 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')
N02=np.loadtxt(f'negativity/N02 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')
# N12=np.loadtxt(f'negativity/N12 {rho_0Name} gamma=0.1g x={x/g}g k={k/g}g.txt')

steps=3000*4
# T0=2*np.pi/np.abs(Omega_n_ij(2,1,2,0,g,0,0,0))
omega0=4*g
T0=2*np.pi/omega0
t_final=10*T0
t=np.linspace(0,t_final,steps)

delta=np.linspace(-15,15,251)


def heatplot(t_data:list,x_data:list,z_data:list,title:str,psi0Name:str):
    fig_u=plt.figure(figsize=(8,6))
    # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
    ax_u=fig_u.add_subplot()
    fig_u.suptitle(title)
    ax_u.set_xlabel('$gt$')
    ax_u.set_ylabel('$\Delta/g$')
    c0 = ax_u.pcolor(t_data, x_data, z_data, shading='auto', cmap='Reds',vmin=0,vmax=1)
    contour_u = ax_u.contourf(t_data, x_data, z_data,levels=[0,0.01],colors='black',linewidths=1)
    ax_u.clabel(contour_u, fmt="%.1f",colors='red',fontsize=10)
    fig_u.colorbar(c0, ax=ax_u,shrink=0.7)
    # fig_u.savefig(rf'graficos\negativity\{psi0Name} {title} x={x/g}g k={k/g}g J={J/g}g neg delta dis.png')

fig=plt.figure(figsize=(8,6),dpi=120)
fig.subplots_adjust(
    left=0.13,
    right=1,
    bottom=0.12,
    top=0.95,
    wspace=0.1,
    hspace=0.1   
)

ax_N0=fig.add_subplot(2,2,1)
ax_N2=fig.add_subplot(2,2,2)
ax_N01=fig.add_subplot(2,2,3)
ax_N02=fig.add_subplot(2,2,4)

ax_N01.set_xlabel('$t/T_0$')
ax_N02.set_xlabel('$t/T_0$')
ax_N0.set_ylabel('$\Delta/g$')
ax_N01.set_ylabel('$\Delta/g$')

ax_N0.set_title(r'$\mathcal{N}_{A}$',y=0.05,color='black', fontsize=20)
ax_N2.set_title(r'$\mathcal{N}_{C}$',y=0.05,color='black', fontsize=20)
ax_N01.set_title(r'$\mathcal{N}_{A:B}$',y=0.05,color='black', fontsize=20)
ax_N02.set_title(r'$\mathcal{N}_{A:C}$',y=0.05,color='black', fontsize=20)



ax_N0.tick_params(axis='x', labelbottom=False)
ax_N2.tick_params(axis='x', labelbottom=False)

ax_N2.tick_params(axis='y', labelleft=False)
ax_N02.tick_params(axis='y', labelleft=False)

ax_N0.set_yticks(np.arange(-15,15,1),minor=True)
ax_N2.set_yticks(np.arange(-15,15,1),minor=True)
ax_N01.set_yticks(np.arange(-15,15,1),minor=True)
ax_N02.set_yticks(np.arange(-15,15,1),minor=True)

c0 = ax_N0.pcolor(t/T0, delta, N0, shading='auto', cmap='Reds',vmin=0,vmax=1)
c0 = ax_N2.pcolor(t/T0, delta, N2, shading='auto', cmap='Reds',vmin=0,vmax=1)
c0 = ax_N01.pcolor(t/T0, delta, N01, shading='auto', cmap='Reds',vmin=0,vmax=1)
c0 = ax_N02.pcolor(t/T0, delta, N02, shading='auto', cmap='Reds',vmin=0,vmax=1)
fig.colorbar(c0, ax=[ax_N0,ax_N2,ax_N01,ax_N02])
plt.show()

## 3 LEER 3x3 cada particion por separado ---- 
# from matplotlib import cm


# def heatplot3x3(negativity_data,title:str,rho_0Name:str,yline:bool):

#     x_max=np.max(negativity_data)
#     x_min=np.min(negativity_data)

#     fig_0, axes = plt.subplots(3, 3, figsize=(10, 8))
#     axes = axes.flatten()
#     # fig.suptitle(f"Concurrence $\psi_0$={psi0Name}")
#     fig_0.suptitle(title)
#     axes[0].annotate(r'$\chi$', xy=(2.95, 1.05), xycoords='axes fraction', xytext=(-0.05, 1.05),
#         arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12, ha='center')
#     axes[0].annotate(r'$k$', xy=(-0.07, -2), xycoords='axes fraction', xytext=(-0.1, 0.95),
#         arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12, va='center')
#     for l in range(9):
#         x_data=negativity_data[l,:,:]
#         if yline:
#             axes[l].axhline(2*x[l],linestyle='--',color='blue')
#             axes[l].axhline(x[l]+2*k[l],linestyle='--',color='blue')
#             axes[l].axhline(3*x[l]-2*k[l],linestyle='--',color='blue')
#         if l==0 or l==3 or l==6:
#             axes[l].set_ylabel('$\Delta/g$')
#         if l==6 or l==7 or l==8:
#             axes[l].set_xlabel('$gt$')
#         c0 = axes[l].pcolor(g*t, delta_ticks, x_data, shading='auto', cmap='magma',vmin=x_min,vmax=x_max)
#         # contour_0 = axes[l].contourf(g*t, delta_ticks, x_data,levels=[0,0.01],colors='black',linewidths=1)
#         # axes[l].clabel(contour_0, fmt="%.1f",colors='red',fontsize=10)
#         axes[l].set_xticklabels([])
#         axes[l].set_yticklabels([])

#     # fig_0.colorbar(c0, ax=ax_0,shrink=0.7)

#     # Adjust layout to make room for colorbar
#     plt.tight_layout()

#     # Add colorbar to the right of the figure
#     fig_0.subplots_adjust(right=0.9)
#     cbar_ax = fig_0.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#     fig_0.colorbar(cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=x_min, vmax=x_max)), cax=cbar_ax, label='N')
#     if gamma!=0:
#         fig_0.savefig(rf'graficos\negativity\3x3 dis {rho_0Name} {title} 012 012 1.png')
#     elif p!=0 and gamma==0:
#         fig_0.savefig(rf'graficos\negativity\3x3 dep {rho_0Name} {title} 012 012 005.png')
#     else:
#         fig_0.savefig(rf'graficos\negativity\3x3 {rho_0Name} {title} 012 012.png')

#     plt.show()


