import jcm_lib as jcm
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import time
import os

script_path= os.path.dirname(__file__)

N_c=4
steps=6000
g_t=10

w0=1
g=0.001*w0

x=0         #1*g va en orden ascendiente
w_q=1
d=0.01*g
w_r=w_q+d        #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g

#Matriz de cambio de base
# M=np.eye(4*N_c)
M=np.zeros((4*N_c,4*N_c))
M[0,3*N_c]=1
M[1,3*N_c+1]=1
M[2,N_c]=1/np.sqrt(2)
M[2,2*N_c]=1/np.sqrt(2)
M[3,N_c]=1/np.sqrt(2)
M[3,2*N_c]=-1/np.sqrt(2)

for ii in range(1,N_c-1):
    M[4*ii,3*N_c+1+ii]=1
for ii in range(1,N_c-1):
    M[4*ii+1,N_c+ii]=1/np.sqrt(2)
    M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,N_c+ii]=1/np.sqrt(2)
    M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
for ii in range(1,N_c-1):
    M[4*ii+2,ii-1]=1

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

#entrelazamiento unitario para 3x3
t_final=g_t/g

ci=0
if ci==0:
    psi_0=tensor((eg+ge).unit(),basis(N_c,1))
elif ci==1:
    psi_0=tensor(gg,basis(N_c,2))



modelo='TCM'
'''##########---Hamiltoniano---##########'''
if modelo=='TCM' or modelo=='1':
    #Hamiltoniano de TC
    H=w_r*n+x*n2 + w_q/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
elif modelo=='RABI' or modelo=='2':
    #Hamiltoniano de Rabi
    H=x*n2 + d/2*(sz1+sz2) + g*(sx1+sx2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
elif modelo=='SpinBoson' or modelo=='SB' or modelo=='3':
    #Hamiltoniano de "spin-boson"
    H=d/2*(sz1+sz2) + g*(sz1+sz2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
else:
    print('Este Modelo no existe. Modelo default es TCM.')
    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

t_0=time.time()

color='black'
t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')
ax_fg=fig_fg.add_subplot()
ax_fg.set_xlim(0,t_final*g)
        
fig_concu=plt.figure(figsize=(8,6))
fig_concu.suptitle('CONCU')
ax_concu=fig_concu.add_subplot()
ax_concu.set_xlim(0,t_final*g)

fig_3concu=plt.figure(figsize=(8,6))
fig_3concu.suptitle('3-CONCU')
ax_3concu=fig_3concu.add_subplot()
ax_3concu.set_xlim(0,t_final*g)

fig_fg_spins=plt.figure(figsize=(8,6))
fig_fg_spins.suptitle('FG SPINS')
ax_fg_spins=fig_fg_spins.add_subplot()
ax_fg_spins.set_xlim(0,t_final*g)
t_i=time.time()

sol=mesolve(H,psi_0,t)

t_i_cb=time.time()

for i in range(len(sol.states)):
    sol.states[i]=sol.states[i].transform(M)
    # sol.states[i].full()[np.abs(sol.states[i].full()) <=1e-8]=0 

t_f_cb=time.time()
print(t_f_cb-t_i_cb,'s tiempo de cambio de base')

#HOY QUIERO PROBAR SI LA FASE MIXTA ANDA UTILIZANDO MAS PASOS --> NO, NO ANDA AUN AUMENTANDO LOS PASOS. HAY QUE HACERA DEVUELTA.

fg_total,arg_tot,eigenvals_tot_t=jcm.fases(sol)
triconcu=jcm.triconcurrence(sol,0.5)
# fg_total_mixta,arg_tot,eigenvals_tot_t=jcm.fases_mixta(sol)

atoms_states=np.empty_like(sol.states)
for j in range(len(sol.states)):
    atoms_states[j]=sol.states[j].ptrace([0,1])

fg_spins,arg_spins,eigenvals_t=jcm.fases(atoms_states)
# fg_spins_mixta,arg_spins,eigenvals_t=jcm.fases_mixta(atoms_states)
concu_ab=jcm.concurrence(atoms_states)

ax_fg.plot(g*t,fg_total/np.pi,color=color)
ax_concu.plot(g*t,concu_ab,color=color)
ax_fg_spins.plot(g*t,fg_spins/np.pi,color=color)
ax_3concu.plot(g*t,triconcu,color='black')

fig_fg_spins.savefig(f'fg tcm/{N_c}x{N_c} fg spin {modelo} {ci} d={d/g}g plot.png')
fig_concu.savefig(f'fg tcm/{N_c}x{N_c} concu {modelo} {ci} d={d/g}g plot.png')
fig_fg.savefig(f'fg tcm/{N_c}x{N_c} fg {modelo} {ci} d={d/g}g plot.png')
fig_3concu.savefig(f'3concu/{N_c}x{N_c} 3concu {modelo} {ci} d={d/g}g plot.png')
t_c=time.time()
print(t_c-t_i,'s computo de simulacion, fg, concu y 3concu')
