from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import jcm_lib as jcm

N_c=10
rho_c=coherent_dm(N_c,1+1j)
ee=basis([2,2],[0,0])
eg=basis([2,2],[0,1])
ge=basis([2,2],[1,0])
gg=basis([2,2],[1,1])

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

# def beta_n(n_:int,k:float,J:float,x:float):
#     return -(x*(n_**2+(n_-1)**2+(n_-2)**2)+J+2*k)

# def gamma_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return (x*(n_-1)**2-J+2*k)*(x*(n_-2)**2+x*n_**2+2*J)+(x*(n_-2)**2+d+J)*(x*n_**2-d+J)-2*g**2*(n_**(2*a)+(n_-1)**(2*a))

# def eta_n(n_:int,d:float,g:float,k:float,J:float,x:float,a:float=0.5):
#     return -(x*n_**2 - d + J)*(x*(n_ - 2)**2 + d + J)*(x*(n_ - 1)**2 - J + 2*k)+ 2*g**2*(x*(n_ - 2)**2*n_**(2*a) + x*n_**2*(n_ - 1)**(2*a) + d* (n_**(2*a) - (n_ - 1)**(2*a)) + J*(n_**(2*a) - (n_ - 1)**(2*a)))

# def Q_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return gamma_n(n_,d,g,k,J,x)/3-beta_n(n_,k,J,x)*beta_n(n_,k,J,x)/9

# def R_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return 1/54*(9*beta_n(n_,k,J,x)*gamma_n(n_,d,g,k,J,x)-27*eta_n(n_,d,g,k,J,x)-2*beta_n(n_,k,J,x)*beta_n(n_,k,J,x)*beta_n(n_,k,J,x))

# def theta_n(n_:int,d:float,g:float,k:float,J:float,x:float):
#     return np.arccos(R_n(n_,d,g,k,J,x)/np.sqrt(-Q_n(n_,d,g,k,J,x)**3))

# def omega_general(n_:int,j:int,d:float,g:float,k:float,J:float,x:float):
#     return 2*np.sqrt(-2*Q_n(n_,d,g,k,J,x))*np.cos((theta_n(n_,d,g,k,J,x)+2*(j-1)*np.pi)/3)
# print(1/2*(eg+ge)*(eg+ge).dag())
rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),rho_c)
# print(rho_0)
w0=1
g=0.001*w0


gamma=0*g       #.1*g
p=0.05*gamma

x=0         #1*g va en orden ascendiente
d=0.001*g        #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g

steps=10000

# T=2*np.pi/omega_general(1,1,d,g,k,J,x)
t_final=100/g

'''##########---Hamiltoniano---##########'''

H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2


t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2]

sol=mesolve(H,rho_0,t,c_ops=l_ops)

fg_total,arg_tot,eigenvals_tot_t=jcm.fases_mixta(sol)

atoms_states=np.empty_like(sol.states)
for j in range(len(sol.states)):
    atoms_states[j]=sol.states[j].ptrace([0,1])  

fg_spins,arg_spins,eigenvals_t=jcm.fases(atoms_states)

concu_ab=jcm.concurrence(atoms_states)

color='black'
fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')
ax_fg=fig_fg.add_subplot()
ax_fg.set_xlim(0,t_final*g)
ax_fg.plot(t*g,fg_total/np.pi,color=color)

fig_concu=plt.figure(figsize=(8,6))
fig_concu.suptitle('CONCU')
ax_concu=fig_concu.add_subplot()
ax_concu.set_xlim(0,g*t_final)
ax_concu.plot(g*t,concu_ab,color=color)

fig_fg_spins=plt.figure(figsize=(8,6))
fig_fg_spins.suptitle('FG SPINS')
ax_fg_spins=fig_fg_spins.add_subplot()
ax_fg_spins.set_xlim(0,g*t_final)
ax_fg_spins.plot(g*t,fg_spins/np.pi,color=color)

plt.show()
