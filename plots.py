from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib 
import matplotlib as mpl
import os


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



psi0=(eg0+ge0).unit()  #gg1#(tensor(tensor(e,gr)+tensor(gr,gr),basis(3,0)+basis(3,1))).unit()#1/10*(gg0*gg0.dag()+(eg0+ge0).unit()*(eg0+ge0).unit().dag()+(eg0-ge0).unit()*(eg0-ge0).unit().dag()+gg1*gg1.dag()+ee0*ee0.dag()+(eg1+ge1).unit()*(eg1+ge1).unit().dag()+(eg1-ge1).unit()*(eg1-ge1).unit().dag()+gg2*gg2.dag()+(eg2+ge2).unit()*(eg2+ge2).unit().dag()+(eg2-ge2).unit()*(eg2-ge2).unit().dag())
psi0Name='eg0+ge0'

w0=1
g=0.001*w0

J=0*g
k=0*g

x=5*g
d=0*g
gamma=0.25*g
p=0.005*g
steps=8000

T=2*np.pi/np.abs(omega_general(1,2,0,g,0,0,0))
# T=2*np.pi/np.abs(rabi_freq(2,1,2,d,g,k,J,x))
t_final=15*T
t=np.linspace(0,t_final,steps)

H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2


sol_u=mesolve(H,psi0,t)
# concu_u[i]=concurrence(sol_u.states)
fg_u,_,_,_=jcm_lib.fases(sol_u)
l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
sol_d=mesolve(H,psi0,t,c_ops=l_ops)
# concu_d[i]=concurrence(sol_d.states)
fg_d,_,_,_=jcm_lib.fases(sol_d)

atoms_states_d=np.empty_like(sol_d.states)
for j in range(len(sol_d.states)):
    atoms_states_d[j]=sol_d.states[j].ptrace([0,1])
concu_d=jcm_lib.concurrence(atoms_states_d)

atoms_states_u=np.empty_like(sol_u.states)
for j in range(len(sol_u.states)):
    atoms_states_u[j]=sol_u.states[j].ptrace([0,1])
concu_u=jcm_lib.concurrence(atoms_states_u)

fig=plt.figure(figsize=(8,6))
fig.suptitle(f'$\psi_0={psi0Name} ; \chi={x/g}g ; \Delta={d/g}g ; k-J={(k-J)/g}g$')
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
