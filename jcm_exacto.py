from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import axes3d

SMALL_SIZE = 15
MEDIUM_SIZE = 20
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

e0=tensor(e,basis(3,0))
g1=tensor(gr,basis(3,1))

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

# Definimos los parametros del problema
w_0=1
g=0.001*w_0

'''----ENERGIAS JCM SIMPLE----'''
def E_jcm1(n_:int,delta:list,x:float):
    return [x*(n_-1/2)**2+x/4+0.5*np.sqrt((delta-x*(n_-1/2))**2+4*g**2*n_),x*(n_-1/2)**2+x/4-0.5*np.sqrt((delta-x*(n_-1/2))**2+4*g**2*n_)]
# delta=np.linspace(-10*g,10*g,10000)
# colors=colormaps['plasma'](np.linspace(0,1,10))
# labels=['$E^1_\pm$','$E^2_\pm$','$E^3_\pm$']
# fig=plt.figure(figsize=(8,6))
# ax=fig.add_subplot()
# ax.plot(delta/g,-delta*100/2,color='black',label='$E^0$')
# for i,x in enumerate(np.linspace(0,2*g,10)):
#     ax.plot(delta/g,E_jcm1(2,delta,x)[0]*100,color=colors[i])
#     ax.plot(delta/g,E_jcm1(2,delta,x)[1]*100,color=colors[i])
# ax.grid()
# ax.legend()
# ax.set_xlim(delta[0]/g,delta[-1]/g)
# ax.set_xlabel('$\Delta/g$',size=15)
# ax.set_ylabel('Energia u.a.',size=15)
# ax.tick_params(labelsize=10)
# plt.show()
# plt.close(fig)

'''----ENERGIAS JCM DOBLE----'''
w_0=1
g=0.001*w_0

t_final=100000
steps=100000

acoplamiento='lineal'
if acoplamiento=='lineal':
    a=1/2
elif acoplamiento=='bs':
    a=1
else:
    print(f"Acoplamietno tiene que ser lineal o bs pero es {acoplamiento}")
    exit()

def beta_n(n_:int,k:float,J:float,x:float):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) + (n_ - 1)**(2*a)))

def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
    return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
    return 2*np.sqrt(-Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)

def rabi_freq(n_:int,j1:int,j2:int,d:float,g:float,k:float,J:float,x:float):
    return omega_general(n_,j2,d,g,k,J,x)-omega_general(n_,j1,d,g,k,J,x)

def grafico2d_chilist(k,J):
    d=np.linspace(-15*g,15*g,2000)
    # E=[[E00],[E11,E12,E13],[E21,E22,E23,E24],...,[En1,En2,En3,En4]]
    # E=[[-d+J],[1/2*(x-d)+k+np.sqrt(2*g**2+(k-J+d/2-x/2)**2),1/2*(x-d)+k-np.sqrt(2*g**2+(k-J+d/2-x/2)**2),(-2*k-J)*np.ones_like(d)],[-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos(theta_n(2)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+2*np.pi)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+4*np.pi)/3),(x-J-2*k)*np.ones_like(d)]]
    # E_jcm=[[1/2*np.sqrt(4*g**2+(d-x)**2),-1/2*np.sqrt(4*g**2+(d-x)**2)],[1/2*np.sqrt(2*4*g**2+(d-3*x)**2),-1/2*np.sqrt(2*4*g**2+(d-3*x)**2)]]

    chi_list=np.linspace(0,5*g,10)
    colors=colormaps['inferno'](np.linspace(0,1,len(chi_list)+2))
    labels=['$\Omega_{21}$','$\Omega_{32}$','$\Omega_{31}$']
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot()
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot()
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    for l,x in enumerate(chi_list):
        y12=[100*rabi_freq(2,1,2,delta,g,k,J,x) for delta in d]
        y23=[100*rabi_freq(2,2,3,delta,g,k,J,x) for delta in d]
        y31=[100*rabi_freq(2,1,3,delta,g,k,J,x) for delta in d]
        ax[0].plot(d/g,y12,color=colors[l])
        ax[1].plot(d/g,y23,color=colors[l])
        ax[1].plot(d/g,y31,color=colors[l])
    ax[0].set_xlabel('$\Delta/g$')
    ax[1].set_xlabel('$\Delta/g$')
    ax[0].set_xlim(1/g*d[0],1/g*d[-1])
    ax[1].set_xlim(1/g*d[0],1/g*d[-1])
    ax1.grid()
    ax2.grid()
    # ax3.grid()
    ax1.set_ylabel('Energia u.a.')
    ax2.set_ylabel('Energia u.a.')
    plt.show()

def grafico2d_dlist(k,J):
    d_list=np.linspace(-10*g,10*g,21)
    # E=[[E00],[E11,E12,E13],[E21,E22,E23,E24],...,[En1,En2,En3,En4]]
    # E=[[-d+J],[1/2*(x-d)+k+np.sqrt(2*g**2+(k-J+d/2-x/2)**2),1/2*(x-d)+k-np.sqrt(2*g**2+(k-J+d/2-x/2)**2),(-2*k-J)*np.ones_like(d)],[-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos(theta_n(2)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+2*np.pi)/3),-1/3*beta_n(2)+2*np.sqrt(-Q_n(2))*np.cos((theta_n(2)+4*np.pi)/3),(x-J-2*k)*np.ones_like(d)]]
    # E_jcm=[[1/2*np.sqrt(4*g**2+(d-x)**2),-1/2*np.sqrt(4*g**2+(d-x)**2)],[1/2*np.sqrt(2*4*g**2+(d-3*x)**2),-1/2*np.sqrt(2*4*g**2+(d-3*x)**2)]]

    chi_list=np.linspace(0,10*g,20000)
    colors=colormaps['inferno'](np.linspace(0,1,len(d_list)+2))
    labels=['$\Omega_{21}$','$\Omega_{32}$','$\Omega_{31}$']
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot()
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot()
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    for l,d in enumerate(d_list):
        y12=[100*rabi_freq(2,1,2,d,g,k,J,x) for x in chi_list]
        y23=[100*rabi_freq(2,2,3,d,g,k,J,x) for x in chi_list]
        y31=[100*rabi_freq(2,1,3,d,g,k,J,x) for x in chi_list]
        ax[0].plot(chi_list/g,y12,color=colors[l])
        ax[1].plot(chi_list/g,y23,color=colors[l])
        ax[1].plot(chi_list/g,y31,color=colors[l])
    ax[0].set_xlabel('$\chi/g$')
    ax[1].set_xlabel('$\chi/g$')
    ax[0].set_xlim(1/g*chi_list[0],1/g*chi_list[-1])
    ax[1].set_xlim(1/g*chi_list[0],1/g*chi_list[-1])
    ax1.grid()
    ax2.grid()
    # ax3.grid()
    ax1.set_ylabel('Energia u.a.')
    ax2.set_ylabel('Energia u.a.')
    plt.show()

'''----frecuencias para diferentes k----'''
def grafico2d_klist(J,x):
    d=np.linspace(-15*g,15*g,20000)

    k_list=np.linspace(0,5*g,10)
    colors=colormaps['inferno'](np.linspace(0,1,len(k_list)+2))
    labels=['$\Omega_{21}$','$\Omega_{32}$','$\Omega_{31}$']
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot()
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot()
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    x=0
    for l,k in enumerate(k_list):
        y12=[100*rabi_freq(2,1,2,delta,g,k,J,x) for delta in d]
        y23=[100*rabi_freq(2,2,3,delta,g,k,J,x) for delta in d]
        y31=[100*rabi_freq(2,1,3,delta,g,k,J,x) for delta in d]
        ax[0].plot(d/g,y12,color=colors[l])
        ax[1].plot(d/g,y23,color=colors[l])
        ax[1].plot(d/g,y31,color=colors[l])
    ax[0].set_xlabel('$\Delta/g$')
    ax[1].set_xlabel('$\Delta/g$')
    ax[0].set_xlim(1/g*d[0],1/g*d[-1])
    ax[1].set_xlim(1/g*d[0],1/g*d[-1])
    ax1.grid()
    ax2.grid()
    # ax3.grid()
    ax1.set_ylabel('Energia u.a.')
    ax2.set_ylabel('Energia u.a.')
    plt.show()

def grafico3d_delta_x(k,J):
    d=np.linspace(-15*g,15*g,200)
    chi_list=np.linspace(0,10*g,200)
    X,Y=np.meshgrid(d/g,chi_list/g)
    z1=np.zeros((len(chi_list),len(d)))
    z2=np.zeros((len(chi_list),len(d)))
    z3=np.zeros((len(chi_list),len(d)))
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(projection='3d')
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot(projection='3d')
    fig3=plt.figure(figsize=(8,6))
    ax3=fig3.add_subplot(projection='3d')
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2,ax3]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    for ld,delta in enumerate(d):
        for lx,x in enumerate(chi_list):
            z1[lx][ld]=100*rabi_freq(2,1,2,delta,g,k,J,x)
            z2[lx][ld]=100*rabi_freq(2,2,3,delta,g,k,J,x)
            z3[lx][ld]=100*rabi_freq(2,3,1,delta,g,k,J,x)
    cmap=colormaps['inferno']
    # surf1=ax1.plot_surface(X,Y,z1,cmap=cmap,lw=0,antialiased=False)
    # surf2=ax2.plot_surface(X,Y,z2,cmap=cmap,lw=0,antialiased=False)
    # surf3=ax3.plot_surface(X,Y,z3,cmap=cmap,lw=0,antialiased=False)
    # fig.colorbar(surf1,shrink=0.5)
    # fig2.colorbar(surf2,shrink=0.5)
    # fig3.colorbar(surf3,shrink=0.5)
    # ax1.view_init(30, -60)
    wire1=ax1.plot_wireframe(X,Y,z1,rstride=200,cstride=10)
    wire2=ax2.plot_wireframe(X,Y,z2,rstride=200,cstride=10)
    wire3=ax3.plot_wireframe(X,Y,z3,rstride=200,cstride=10)
    ax2.view_init(30, -60)
    ax3.view_init(30, -60)
    ax1.set_xlabel('$\Delta/g$')
    ax2.set_xlabel('$\Delta/g$')
    ax3.set_xlabel('$\Delta/g$')
    ax1.set_ylabel('$\chi/g$')
    ax2.set_ylabel('$\chi/g$')
    ax3.set_ylabel('$\chi/g$')
    plt.show()

def grafico3d_delta_k(x,J):
    d=np.linspace(-15*g,15*g,200)
    k_list=np.linspace(0,5*g,200)
    X,Y=np.meshgrid(d/g,k_list/g)
    z1=np.zeros((len(k_list),len(d)))
    z2=np.zeros((len(k_list),len(d)))
    z3=np.zeros((len(k_list),len(d)))
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(projection='3d')
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot(projection='3d')
    fig3=plt.figure(figsize=(8,6))
    ax3=fig3.add_subplot(projection='3d')
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2,ax3]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    for ld,delta in enumerate(d):
        for lk,k in enumerate(k_list):
            z1[lk][ld]=100*rabi_freq(2,1,2,delta,g,k,J,x)
            z2[lk][ld]=100*rabi_freq(2,2,3,delta,g,k,J,x)
            z3[lk][ld]=100*rabi_freq(2,3,1,delta,g,k,J,x)
    cmap=colormaps['inferno']
    surf1=ax1.plot_surface(X,Y,z1,cmap=cmap,lw=0,antialiased=False)
    surf2=ax2.plot_surface(X,Y,z2,cmap=cmap,lw=0,antialiased=False)
    surf3=ax3.plot_surface(X,Y,z3,cmap=cmap,lw=0,antialiased=False)
    fig.colorbar(surf1,shrink=0.5)
    fig2.colorbar(surf2,shrink=0.5)
    fig3.colorbar(surf3,shrink=0.5)
    ax1.view_init(30, -60)
    ax2.view_init(30, -60)
    ax3.view_init(30, -60)
    ax1.set_xlabel('$\Delta/g$')
    ax2.set_xlabel('$\Delta/g$')
    ax3.set_xlabel('$\Delta/g$')
    ax1.set_ylabel('$k/g$')
    ax2.set_ylabel('$k/g$')
    ax3.set_ylabel('$k/g$')
    plt.show()

def grafico3d_chi_k(d,J):
    chi_list=np.linspace(0,10*g,200)
    k_list=np.linspace(0,10*g,200)
    X,Y=np.meshgrid(chi_list/g,k_list/g)
    z1=np.zeros((len(k_list),len(chi_list)))
    z2=np.zeros((len(k_list),len(chi_list)))
    z3=np.zeros((len(k_list),len(chi_list)))
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(projection='3d')
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot(projection='3d')
    fig3=plt.figure(figsize=(8,6))
    ax3=fig3.add_subplot(projection='3d')
    # ax3=fig.add_subplot(133)
    ax=[ax1,ax2,ax3]
    # for m,j in enumerate([[1,2],[2,3],[1,3]]):
    for lx,x in enumerate(chi_list):
        for lk,k in enumerate(k_list):
            z1[lk][lx]=100*rabi_freq(2,1,2,d,g,k,J,x)
            z2[lk][lx]=100*rabi_freq(2,2,3,d,g,k,J,x)
            z3[lk][lx]=100*rabi_freq(2,3,1,d,g,k,J,x)
    cmap=colormaps['inferno']
    surf1=ax1.plot_surface(X,Y,z1,cmap=cmap,lw=0,antialiased=False)
    surf2=ax2.plot_surface(X,Y,z2,cmap=cmap,lw=0,antialiased=False)
    surf3=ax3.plot_surface(X,Y,z3,cmap=cmap,lw=0,antialiased=False)
    fig.colorbar(surf1,shrink=0.5)
    fig2.colorbar(surf2,shrink=0.5)
    fig3.colorbar(surf3,shrink=0.5)
    ax1.view_init(30, -30)
    ax2.view_init(30, -30)
    ax3.view_init(30, -30)
    ax1.set_xlabel('$\chi/g$')
    ax2.set_xlabel('$\chi/g$')
    ax3.set_xlabel('$\chi/g$')
    ax1.set_ylabel('$k/g$')
    ax2.set_ylabel('$k/g$')
    ax3.set_ylabel('$k/g$')
    plt.show()


# grafico2d_klist(2.5*g,0)

x=0*g
k=0*g
J=0*g
xmax=10
d=np.linspace(-xmax*g,xmax*g,10000)
# fig_real=plt.figure(figsize=(8,6))
# ax_real=fig_real.add_subplot()
# # ax_real.plot(d/g,[np.real(Q_n(2,delta,g,k,J,x)**3+R_n(2,delta,g,k,J,x)**2) for delta in d],color='blue')
# # ax_real.plot(d/g,[np.imag(Q_n(2,delta,g,k,J,x)**3+R_n(2,delta,g,k,J,x)**2) for delta in d],color='red')
# # ax_real.plot(d/g,[R_n(2,delta,g,k,J,x) for delta in d],color='black')
# # ax_real.plot(d/g,[Q_n(2,delta,g,k,J,x) for delta in d],color='blue')
# # ax_real.plot(d/g,[np.arccos(R_n(2,delta,g,k,J,x)/np.sqrt(-Q_n(2,delta,g,k,J,x)**3)) for delta in d],color='black')
# ax_real.plot(d/g,[np.sqrt(-Q_n(2,delta,g,k,J,x)**3) for delta in d],color='red')
# ax_real.plot(d/g,[R_n(2,delta,g,k,J,x) for delta in d],color='blue')

# # ax_real.plot(d/g,[R_n(2,delta,g,k,J,x)/np.sqrt(-Q_n(2,delta,g,k,J,x)**3) for delta in d],color='blue')
# ax_real.grid()
# plt.show()
E_jcm=[[1/2*np.sqrt(4*g**2+(d-x)**2),-1/2*np.sqrt(4*g**2+(d-x)**2)],[1/2*np.sqrt(2*4*g**2+(d-3*x)**2),-1/2*np.sqrt(2*4*g**2+(d-3*x)**2)]]
E=[-d+J,(x-d)/2+k+np.sqrt(2*g**2+(k-J+d/2-x/2)**2),(x-d)/2+k-np.sqrt(2*g**2+(k-J+d/2-x/2)**2),[-1/3*beta_n(2,k,J,x)+omega_general(2,1,delta,g,k,J,x) for delta in d],[-1/3*beta_n(2,k,J,x)+omega_general(2,2,delta,g,k,J,x) for delta in d],[-1/3*beta_n(2,k,J,x)+omega_general(2,3,delta,g,k,J,x) for delta in d]]
labels=['$E^0$','$E^1$','$E^1$','$E^2$','$E^2$','$E^2$']

colores=colormaps['inferno'](np.linspace(0,1,7))
fig_e=plt.figure(figsize=(8,6))
ax_e=fig_e.add_subplot()
ax_e.set_xlabel('$\Delta/g$')
ax_e.set_ylabel('Energia u.a.')
lines=[]
  
line0,=ax_e.plot(d/g,E[0],color=colores[0],label=labels[0])
line1,=ax_e.plot(d/g,E[1],color=colores[2],label=labels[1])
line2,=ax_e.plot(d/g,E[2],color=colores[2])
line3,=ax_e.plot(d/g,E[3],color=colores[4],label=labels[3])
line4,=ax_e.plot(d/g,E[4],color=colores[4])
line5,=ax_e.plot(d/g,E[5],color=colores[4])

line_ejcm0,=ax_e.plot(d/g,2*E_jcm1(1,d,x)[0],color=colores[2],label='$2E^1_{JC}$',ls='dashed')
ax_e.plot(d/g,2*E_jcm1(1,d,x)[1],color=colores[2],ls='dashed')
line_ejcm1,=ax_e.plot(d/g,2*E_jcm1(2,d,x)[0],color=colores[4],label='$2E^2_{JC}$',ls='dashed')
ax_e.plot(d/g,2*E_jcm1(2,d,x)[1],color=colores[4],ls='dashed')
ax_e.set_xlim(d[0]/g,d[-1]/g)
ax_e.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)
ax_e.legend(handles=[line_ejcm0,line_ejcm1,line0,line1,line3],loc='center right',framealpha=0.5)
ax_e.grid()
# fig_e.savefig('energias x=5.pdf')
plt.show()

# fig2=plt.figure(figsize=(8,6))
# ax2=fig2.add_subplot()
# ax2.ticklabel_format(style='scientific',scilimits=(-1,2),useMathText=True)
# ax2.grid()
# ax2.set_xlim(-xmax,xmax)
# lista=np.linspace(0,2*g,10)

# colors=colormaps['inferno'](np.linspace(0,1,len(lista)))

# for l,x in enumerate(lista):
#     E2=[[-1/3*beta_n(2,k,J,x)+omega_general(2,1,delta,g,k,J,x) for delta in d],[-1/3*beta_n(2,k,J,x)+omega_general(2,2,delta,g,k,J,x) for delta in d],[-1/3*beta_n(2,k,J,x)+omega_general(2,3,delta,g,k,J,x) for delta in d]]

#     ax2.plot(d/g,E2[0],color=colors[l])
#     ax2.plot(d/g,E2[1],color=colors[l])
#     ax2.plot(d/g,E2[2],color=colors[l])
# plt.show()
