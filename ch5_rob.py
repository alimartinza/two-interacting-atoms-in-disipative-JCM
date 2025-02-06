from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from jcm_lib import fases

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

acoplamiento='lineal'
if acoplamiento=='lineal':
    acop=1/2
elif acoplamiento=='bs':
    acop=1
else:
    print(f"Acoplamietno tiene que ser lineal o bs pero es {acoplamiento}")
    exit()

def beta_n(n_:int,k:float,J:float,x:float):
    return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=acop):
    return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=acop):
    return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

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



def f():
    if acoplamiento=='lineal':
        return 1
    elif acoplamiento=='bs':
        return sqrtN

def pr(estado):
    return estado.unit()*estado.unit().dag()

w_0=1
g=0.001*w_0

p=0.005*g

def robustez_delta(psi0,psi0Name,delta_min:float,delta_max:float,delta_steps:int,x:float,k:float,J:float,cluster_centers:list=None,cluster_radii:list=None,cluster_steps:list=None):

    fig_fg=plt.figure(figsize=(8,6))
    ax_fg=fig_fg.add_subplot()
    delta_list=np.linspace(delta_min,delta_max,delta_steps)
    if cluster_centers is not None:
        for l in range(len(cluster_centers)):
            delta_list=np.concat((delta_list,np.linspace(cluster_centers[l]-cluster_radii[l],cluster_centers[l]+cluster_radii[l],cluster_steps[l])))
    else:
        None

    deltafg=np.zeros((4,len(delta_list)))
    deltafg[0]=delta_list
    for j,gamma in enumerate([0.01*g,0.1*g,0.25*g]):

        colors=mpl.colormaps['plasma'](np.linspace(0,1,3+1))
        
        for i,delta in enumerate(delta_list):

            
            '''---Hamiltoniano---'''

            H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
                
            '''---Simulacion numerica---'''
            if psi0Name=='eg0+ge0':
                T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))
            else:
                T=2*np.pi/np.abs(rabi_freq(2,1,2,delta,g,k,J,x))
            t_final=3*T
            steps=2000

            l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
            t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
            sol_u=mesolve(H,psi0,t,c_ops=[])
            sol_d=mesolve(H,psi0,t,c_ops=l_ops)
            fg_u,arg,eigenvals_t_u = fases(sol_u)
            fg_d,arg,eigenvals_t_d = fases(sol_d)

            deltafg[j+1][i]=fg_d[-1]-fg_u[-1]

        ax_fg.scatter(deltafg[0]/g,deltafg[j+1]/np.pi,color=colors[j],marker='.')

    ax_fg.set_ylabel('$\delta\phi/\pi$',size=20)
    ax_fg.set_xlabel('$\Delta/g$',size=20)
    ax_fg.set_xlim(delta_min/g,delta_max/g)
    ax_fg.hlines(0,delta_min/g,delta_max/g,colors='grey',linestyles='dashed',alpha=0.5)
    # ax_fg.vlines([x/g*3-2*(k-J)/g,x/g+2*(k-J)/g],np.min(deltafg)/np.pi,np.max(deltafg)/np.pi,color='red',linestyles='dashed',alpha=0.5)
    ax_fg.ticklabel_format(style='sci',scilimits=(-2,2),useMathText=True)
    
    
    with open(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} limites.txt','a') as limites_file:
        limites_file.write('\n')
        limites_file.write(f'{k/g},{x/g},{J/g},{delta_min/g},{delta_max/g},{len(delta_list)},{cluster_centers},{cluster_radii},{cluster_steps}')

    fig_fg.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta img\{psi0Name} k={k/g}g x={x/g}g J={J/g}g.png')
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\delta\{psi0Name} k={k/g}g x={x/g}g J={J/g}g delta.txt',deltafg)
   
    plt.close('all')

def robustez_chi(psi0,psi0Name,chi_min:float,chi_max:float,chi_steps:int,delta:float,k:float,J:float,cluster_centers:list=None,cluster_radii:list=None,cluster_steps:list=None,saveword:str=""):
    fig_fg=plt.figure(figsize=(8,6))
    ax_fg=fig_fg.add_subplot()
    chi_list=np.linspace(chi_min,chi_max,chi_steps)
    
    if cluster_centers is not None:
        for l in range(len(cluster_centers)):
            chi_list=np.concat((chi_list,np.linspace(cluster_centers[l]-cluster_radii[l],cluster_centers[l]+cluster_radii[l],cluster_steps[l])))
        chi_list=np.sort(chi_list)
        # print(chi_list)
    else:
        None
    # chi_list sort por clusters
    deltafg=np.zeros((4,len(chi_list)))
    deltafg[0]=chi_list
    for j,gamma in enumerate([0.01*g,0.1*g,0.25*g]):

        colors=mpl.colormaps['plasma'](np.linspace(0,1,3+1))
        
        for i,x in enumerate(chi_list):

            
            '''---Hamiltoniano---'''

            H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
                
            '''---Simulacion numerica---'''
            if psi0Name=='eg0+ge0':
                T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))
            else:
                T=2*np.pi/np.abs(rabi_freq(2,1,2,delta,g,k,J,x))
            t_final=3*T
            steps=2000

            l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
            t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
            sol_u=mesolve(H,psi0,t,c_ops=[])
            sol_d=mesolve(H,psi0,t,c_ops=l_ops)
            fg_u,arg,eigenvals_t_u = fases(sol_u)
            fg_d,arg,eigenvals_t_d = fases(sol_d)

            deltafg[j+1][i]=fg_d[-1]-fg_u[-1]

        ax_fg.scatter(deltafg[0]/g,deltafg[j+1]/np.pi,color=colors[j],marker='.')

    ax_fg.set_ylabel('$\delta\phi/\pi$',size=20)
    ax_fg.set_xlabel('$\chi/g$',size=20)
    ax_fg.set_xlim(chi_list[0]/g,chi_list[-1]/g)
    ax_fg.hlines(0,chi_list[0]/g,chi_list[-1]/g,colors='grey',linestyles='dashed',alpha=0.5)
    # ax_fg.vlines([delta/g/3+2/3*(k-J)/g,delta/g-2*(k-J)/g],np.min(deltafg)/np.pi,np.max(deltafg)/np.pi,color='red',linestyles='dashed',alpha=0.5)
    ax_fg.ticklabel_format(style='sci',scilimits=(-2,2),useMathText=True)


    with open(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} {saveword}limites.txt','a') as limites_file:
        limites_file.write('\n')
        limites_file.write(f'{delta/g},{k/g},{J/g},{chi_min/g},{chi_max/g},{len(chi_list)}')

    fig_fg.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi img\{psi0Name} {saveword}d={delta/g}g k={k/g}g J={J/g}g.png')
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\chi\{psi0Name} {saveword}d={delta/g}g k={k/g}g J={J/g}g chi.txt',deltafg)
    plt.close('all')


def robustez_k(psi0,psi0Name,k_min:float,k_max:float,k_steps:int,delta:float,x:float,J:float,cluster_centers:list=None,cluster_radii:list=None,cluster_steps:list=None,saveword:str=""):
    fig_fg=plt.figure(figsize=(8,6))
    ax_fg=fig_fg.add_subplot()
    k_list=np.linspace(k_min,k_max,k_steps)
    
    if cluster_centers is not None:
        for l in range(len(cluster_centers)):
            k_list=np.concat((k_list,np.linspace(cluster_centers[l]-cluster_radii[l],cluster_centers[l]+cluster_radii[l],cluster_steps[l])))
        k_list=np.sort(k_list)
    else:
        None
    # k_list sort para cluster
    deltafg=np.zeros((4,len(k_list)))
    deltafg[0]=k_list
    for j,gamma in enumerate([0.01*g,0.1*g,0.25*g]):

        colors=mpl.colormaps['plasma'](np.linspace(0,1,3+1))
        
        for i,k in enumerate(k_list):

            
            '''---Hamiltoniano---'''

            H=x*n2 + delta/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
                
            '''---Simulacion numerica---'''
            if psi0Name=='eg0+ge0':
                T=np.pi/(np.sqrt(2*g**2+(k-J+delta/2-x/2)**2))
            else:
                T=2*np.pi/np.abs(rabi_freq(2,1,2,delta,g,k,J,x))
            t_final=3*T
            steps=2000

            l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
            t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
            sol_u=mesolve(H,psi0,t,c_ops=[])
            sol_d=mesolve(H,psi0,t,c_ops=l_ops)
            fg_u,arg,eigenvals_t_u = fases(sol_u)
            fg_d,arg,eigenvals_t_d = fases(sol_d)

            deltafg[j+1][i]=fg_d[-1]-fg_u[-1]

        ax_fg.scatter(deltafg[0]/g,deltafg[j+1]/np.pi,color=colors[j],marker='.')

    ax_fg.set_ylabel('$\delta\phi/\pi$',size=20)
    ax_fg.set_xlabel('$k/g$',size=20)
    ax_fg.set_xlim(k_list[0]/g,k_list[-1]/g)
    ax_fg.hlines(0,k_list[0]/g,k_list[-1]/g,colors='grey',linestyles='dashed',alpha=0.5)
    # ax_fg.vlines([delta/g/3+2/3*(k-J)/g,delta/g-2*(k-J)/g],np.min(deltafg)/np.pi,np.max(deltafg)/np.pi,color='red',linestyles='dashed',alpha=0.5)
    ax_fg.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)


    with open(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k\{psi0Name} {saveword}limites.txt','a') as limites_file:
        limites_file.write('\n')
        limites_file.write(f'{delta/g},{x/g},{J/g},{k_min/g},{k_max/g},{len(k_list)}')

    fig_fg.savefig(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k img\{psi0Name} {saveword}d={delta/g}g x={x/g}g J={J/g}g.png')
    np.savetxt(rf'D:\Estudios\Tesis\imagenes analisis\t-ordenado\5\robustez\k\{psi0Name} {saveword}d={delta/g}g x={x/g}g J={J/g}g k.txt',deltafg)
    plt.close('all')

psi0=(eg1+ge1).unit()
psi0Name='eg1+ge1 cluster'
robustez_delta(psi0,psi0Name,delta_min=-15*g,delta_max=15*g,delta_steps=302,x=0,k=2.5*g,J=0,cluster_centers=[0],cluster_radii=[5*g],cluster_steps=[152])
robustez_delta(psi0,psi0Name,delta_min=-10*g,delta_max=25*g,delta_steps=302,x=5*g,k=0,J=0,cluster_centers=[10*g],cluster_radii=[2.5*g],cluster_steps=[102])
robustez_delta(psi0,psi0Name,delta_min=-10*g,delta_max=15*g,delta_steps=302,x=5*g,k=2.5*g,J=0,cluster_centers=[0],cluster_radii=[2.5*g],cluster_steps=[102])
robustez_delta(psi0,psi0Name,delta_min=-10*g,delta_max=15*g,delta_steps=302,x=2*g,k=0,J=3*g,cluster_centers=[0],cluster_radii=[2.5*g],cluster_steps=[152])

# psi0=(gg2).unit()
# psi0Name='gg2'
# robustez_delta(psi0,psi0Name,-12*g,12*g,203,x=0,k=0,J=0)
# robustez_delta(psi0,psi0Name,-12*g,12*g,203,x=0,k=2.5*g,J=0)
# robustez_delta(psi0,psi0Name,-12*g,12*g,203,x=0,k=0.5*g,J=0)
# robustez_delta(psi0,psi0Name,-12*g,12*g,203,x=0.5*g,k=0,J=0)
# robustez_delta(psi0,psi0Name,-12*g,12*g,203,x=2.5*g,k=0,J=0)

# robustez_chi(psi0,psi0Name,-12*g,12*g,203,delta=0,k=0,J=0)
# robustez_chi(psi0,psi0Name,-12*g,12*g,203,delta=0.5*g,k=0,J=0)
# robustez_chi(psi0,psi0Name,-12*g,12*g,203,delta=2.5*g,k=0,J=0)
# robustez_chi(psi0,psi0Name,-12*g,12*g,203,delta=0,k=0.5*g,J=0)
# robustez_chi(psi0,psi0Name,-12*g,12*g,203,delta=0,k=2.5*g,J=0)

# robustez_k(gg2,'gg2',-12*g,12*g,203,delta=0,x=0*g,J=0,cluster_centers=[0],cluster_radii=[0.25*g],cluster_steps=[20])
# robustez_k(gg2,'gg2',-12*g,12*g,203,delta=2.5*g,x=0*g,J=0)
# robustez_k(gg2,'gg2',-12*g,12*g,203,delta=0.5*g,x=0*g,J=0)
# robustez_k(gg2,'gg2',-12*g,12*g,406,delta=0,x=2.5*g,J=0)
# robustez_k(gg2,'gg2',-12*g,12*g,406,delta=2.5*g,x=2.5*g,J=0)

