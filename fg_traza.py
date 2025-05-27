from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm
import matplotlib as mpl
import os
import time

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

e0=tensor(e,basis(3,0))
g0=tensor(gr,basis(3,0))
g1=tensor(gr,basis(3,1))
sx=tensor(sigmax(),qeye(3))
sy=tensor(sigmay(),qeye(3))
sz=tensor(sigmaz(),qeye(3))
sp=tensor(sigmap(),qeye(3))
sm=tensor(sigmam(),qeye(3))


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
plt.rc('figure.subplot',left=0.1)
plt.rc('figure.subplot',bottom=0.102)
plt.rc('figure.subplot',right=0.962)
plt.rc('figure.subplot',top=0.95)


script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU
os.chdir(script_path)
os.chdir('./reordenamiento')

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


'''#################################################################################################################################################
---------------------------------------------------ALPHA=1-------------------------------------------------------------------
####################################################################################################################################################'''

process_ti=time.process_time()

w0=1
g=0.001*w0


gamma=0.1*g#.1*g

x_list=[0.0001*g,0.5*g,1*g,2*g,5*g] #1*g va en orden ascendiente
d=1*g #1.1001*g#.5*g

k_list=[2.5*g,1*g,0.5*g,0.25*g,0*g] #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g
params=np.array([[0,0]])
for i in range(len(k_list)):
    for j in range(len(x_list)):
        params=np.append(params,[[x_list[j],k_list[i]]],axis=0)

params=np.delete(params,0,axis=0)


## CHEQUEAR BIEN CUALES SON LOS EJES DE CADA PARAMETRO Y PONER FLECHAS EN EL PLOT MOSTRANDO PARA DONDE CRECE CADA COSA.

psi0=(eg1+ge1).unit()  #gg1#(tensor(tensor(e,gr)+tensor(gr,gr),basis(3,0)+basis(3,1))).unit()#1/10*(gg0*gg0.dag()+(eg0+ge0).unit()*(eg0+ge0).unit().dag()+(eg0-ge0).unit()*(eg0-ge0).unit().dag()+gg1*gg1.dag()+ee0*ee0.dag()+(eg1+ge1).unit()*(eg1+ge1).unit().dag()+(eg1-ge1).unit()*(eg1-ge1).unit().dag()+gg2*gg2.dag()+(eg2+ge2).unit()*(eg2+ge2).unit().dag()+(eg2-ge2).unit()*(eg2-ge2).unit().dag())
psi0Name='eg1+ge1'
# prefijo=f'j d={d/g} x={x/g} k={k/g} J={J/g}'

steps=20000

# T=2*np.pi/energiasn1(1,g,d,x,k,J)
# T=2*np.pi/(omega_general(1,1,d,g,k,J,x))
# T=2*np.pi/(-beta_n(1,k,J,x)/3+omega_general(1,1,d,g,k,J,x))
# print(T)

# ciclos_bloch=5
# points=2000

acoplamiento='lineal'
def f():
    if acoplamiento=='lineal':
        return 1
    elif acoplamiento=='bs':
        return sqrtN

def pr(estado):
    return estado.unit()*estado.unit().dag()



fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')

fig_concu=plt.figure(figsize=(8,6))
fig_concu.suptitle('Concu')
fig_fg_ab=plt.figure(figsize=(8,6))
fig_fg_ab.suptitle('FG AB')
fig_fg_c=plt.figure(figsize=(8,6))
fig_fg_c.suptitle('FG C')
fig_fg_ac=plt.figure(figsize=(8,6))
fig_fg_ac.suptitle('FG AC')



for i,params in enumerate(params):
    x=params[0]
    k=0
    d=params[1]

    T=2*np.pi/omega_general(1,1,d,g,k,J,x)
    t_final=10*T

    '''##########---Hamiltoniano---##########'''

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    '''#######---Simulacion numerica---#######'''

    # try:
    #     ax_fg=fig_fg.add_subplot(3,3,i+1,sharex=ax_fg,sharey=ax_fg)
    # except:
    ax_fg=fig_fg.add_subplot(5,5,i+1)
    ax_fg.set_xlim(0,t_final/T)
    ax_fg.set_xticklabels([])
    ax_fg.set_yticklabels([])

    ax_concu=fig_concu.add_subplot(5,5,i+1)
    ax_concu.set_ylim(0,1)
    ax_concu.set_xlim(0,t_final/T)
    ax_concu.set_xticklabels([])
    ax_concu.set_yticklabels([])

    ax_fg_ab=fig_fg_ab.add_subplot(5,5,i+1)
    ax_fg_ab.set_xlim(0,t_final/T)
    ax_fg_ab.set_xticklabels([])
    ax_fg_ab.set_yticklabels([])

    ax_fg_c=fig_fg_c.add_subplot(5,5,i+1)
    ax_fg_c.set_xlim(0,t_final/T)
    ax_fg_c.set_xticklabels([])
    ax_fg_c.set_yticklabels([])

    ax_fg_ac=fig_fg_ac.add_subplot(5,5,i+1)
    ax_fg_ac.set_xlim(0,t_final/T)
    ax_fg_ac.set_xticklabels([])
    ax_fg_ac.set_yticklabels([])

    for gamma,color in zip([0,0.1*g],['black','red']):
        p=0.05*gamma
        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2] #OPERADORES DE COLAPSO

        t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

        sol=mesolve(H,psi0,t,c_ops=l_ops)

        def inferno(points:int):
            nrm = mpl.colors.Normalize(0, points)
            return mpl.cm.inferno(nrm(np.linspace(0,points,points)))

        '''########---Estado total---###########'''

        fg,arg,eigenvals_t = jcm.fases(sol)

        process_t0=time.process_time()
        print(f'{i}. SIMU Y CALCULO FG     ; {process_t0-process_ti}s ; -- ')

        ax_fg.plot(t/T,fg/np.pi,color=color)

        '''###---Atomo A-Atomo B---###'''

        atoms_states=np.empty_like(sol.states)
        for j in range(len(sol.states)):
            atoms_states[j]=sol.states[j].ptrace([0,1])  

        concu_ab=jcm.concurrence(atoms_states)
        fg_atoms,arg_atoms,eigenvals_t_atoms =jcm.fases(atoms_states)
        ax_concu.plot(t/T,concu_ab,color=color)
        ax_fg_ab.plot(t/T,fg_atoms/np.pi,color=color)

        '''##############---Cavidad---############'''

        cavity_states=np.empty_like(sol.states)
        for j in range(len(sol.states)):
            cavity_states[j]=sol.states[j].ptrace([2])

        fg_c,arg_cav,eigenvals_t_cav =jcm.fases(cavity_states)
        ax_fg_c.plot(t/T,fg_atoms/np.pi,color=color)


        '''#########---Atomo A-Cavidad----##########'''

        atom_acavity_states=np.empty_like(sol.states)
        for j in range(len(sol.states)):
            atom_acavity_states[j]=sol.states[j].ptrace([0,2])

        fg_ac,arg,eigenvals_t = jcm.fases(atom_acavity_states)
        ax_fg_ac.plot(t/T,fg_ac/np.pi,color=color)

sang=0.05
fig_fg.subplots_adjust(sang*1.5,sang*1.5,1-sang,1-sang,sang,sang)
ax_fg.annotate('$\chi$',xy=(0,sang),xycoords='figure fraction', xytext=(1-sang,sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
ax_fg.annotate('$\Delta$',xy=(sang,0),xycoords='figure fraction', xytext=(sang,1-sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
fig_fg.savefig(f'k=0 fg.png')

fig_concu.subplots_adjust(sang*1.5,sang*1.5,1-sang,1-sang,sang,sang)
ax_concu.annotate('$\chi$',xy=(0,sang),xycoords='figure fraction', xytext=(1-sang,sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
ax_concu.annotate('$\Delta$',xy=(sang,0),xycoords='figure fraction', xytext=(sang,1-sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
fig_concu.savefig(f'k=0 concu.png')

fig_fg_ab.subplots_adjust(sang*1.5,sang*1.5,1-sang,1-sang,sang,sang)
ax_fg_ab.annotate('$\chi$',xy=(0,sang),xycoords='figure fraction', xytext=(1-sang,sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
ax_fg_ab.annotate('$\Delta$',xy=(sang,0),xycoords='figure fraction', xytext=(sang,1-sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
fig_fg_ab.savefig(f'k=0 fg ab.png')

fig_fg_c.subplots_adjust(sang*1.5,sang*1.5,1-sang,1-sang,sang,sang)
ax_fg_c.annotate('$\chi$',xy=(0,sang),xycoords='figure fraction', xytext=(1-sang,sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
ax_fg_c.annotate('$\Delta$',xy=(sang,0),xycoords='figure fraction', xytext=(sang,1-sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
fig_fg_c.savefig(f'k=0 fg c.png')

fig_fg_ac.subplots_adjust(sang*1.5,sang*1.5,1-sang,1-sang,sang,sang)
ax_fg_ac.annotate('$\chi$',xy=(0,sang),xycoords='figure fraction', xytext=(1-sang,sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
ax_fg_ac.annotate('$\Delta$',xy=(sang,0),xycoords='figure fraction', xytext=(sang,1-sang),
                arrowprops=dict(arrowstyle="<-",color='black'))
fig_fg_ac.savefig(f'k=0 fg ac.png')

plt.show()