from qutip import *
import numpy as np
import pandas as pd

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

w_0=1
g=0.001*w_0
p=0.005*g
k=0.1*g
x=0*g
d=0.1*g

gamma=0.1*g
J=0
t_final=25000
steps=2000

'''---Hamiltoniano---'''

H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

'''---Simulacion numerica---'''

t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
#DEFINIMOS LOS DISIPADORES SI HAY DISIPACION, Y SI NO, ENTONCES ESTA VACIO

psi0=eg0
sol=mesolve(H,psi0,t,c_ops=[],progress_bar=True)

atoms_states=np.empty_like(sol.states)
for j in range(len(sol.states)):
    atoms_states[j]=sol.states[j].ptrace([0,1])  


print(type(sol.states))
