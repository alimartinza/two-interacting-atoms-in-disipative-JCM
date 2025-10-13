import numpy as np
# from pymanopt.function import numpy
from qutip import *
import jcm_lib as jcm
import os
import time
import matplotlib.pyplot as plt
from entrelazamiento_lib import negativity_hor,roof_etanglement_bipartite,rankV_func,pure_tripartite_ent
from scipy.optimize import minimize,NonlinearConstraint

script_path = os.path.dirname(__file__) 

N_c=3
steps=6000
g_t=30

w0=1
g=0.001*w0

gamma=0*g

x=0         #1*g va en orden ascendiente
d=1*g       #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g

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
gm1=Qobj([[0,1,0],[1,0,0],[0,0,0]],dims=[[3],[3]])
gm2=Qobj([[0,-1j,0],[1j,0,0],[0,0,0]],dims=[[3],[3]])
gm3=Qobj([[1,0,0],[0,-1,0],[0,0,0]],dims=[[3],[3]])
gm4=Qobj([[0,0,1],[0,0,0],[1,0,0]],dims=[[3],[3]])
gm5=Qobj([[0,0,-1j],[0,0,0],[1j,0,0]],dims=[[3],[3]])
gm6=Qobj([[0,0,0],[0,0,1],[0,1,0]],dims=[[3],[3]])
gm7=Qobj([[0,0,0],[0,0,-1j],[0,1j,0]],dims=[[3],[3]])
gm8=Qobj([[1/np.sqrt(3),0,0],[0,1/np.sqrt(3),0],[0,0,-2/np.sqrt(3)]],dims=[[3],[3]])


# --- DIFERENTES ESTADOS INICIALES --- #
def simulacion(d:float):
    rho_0=tensor(gg,basis(N_c,2))

    t_final=g_t/g

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

    p=0.05*gamma

    sol=mesolve(H,rho_0,t)

    fg_total,arg_tot,eigenvals_tot_t=jcm.fases(sol)

    rho_01=np.empty_like(sol.states) #2x2
    rho_02=np.empty_like(sol.states)  #2x3
    rho_12=np.empty_like(sol.states) #2x3

    for j in range(len(sol.states)):
        rho_01[j]=sol.states[j].ptrace([0,1])
        rho_02[j]=sol.states[j].ptrace([0,2])
        rho_12[j]=sol.states[j].ptrace([1,2])

    return sol,rho_01,rho_02,rho_12,fg_total

sol,rho_01,rho_02,rho_12,fg_total=simulacion(d)

def rho_pauli(x1,x2,x3):
    return 1/2*qeye(2)+x1*sigmax()+x2*sigmay()+x3*sigmaz()

def rho_gellman(x1,x2,x3,x4,x5,x6,x7,x8):
    return 1/3*qeye(3)+x1*gm1+x2*gm2+x3*gm3+x4*gm4+x5*gm5+x6*gm6+x7*gm7+x8*gm8

def sep_dm(x:list):
    return tensor(rho_pauli(x[0],x[1],x[2]),rho_pauli(x[3],x[4],x[5]),rho_gellman(x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13]))

def const(x):
    return [x[0]**2+x[1]**2+x[2]**2,x[3]**2+x[4]**2+x[5]**2,x[6]**2+x[7]**2+x[8]**2+x[9]**2+x[10]**2+x[11]**2+x[12]**2+x[13]**2]

def jacob(x):
    return [[2*x[0],2*x[1],2*x[2],0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,2*x[3],2*x[4],2*x[5],0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,2*x[6],2*x[7],2*x[8],2*x[9],2*x[10],2*x[11],2*x[12],2*x[13]]]


def hess(x,v):
    return v[0]*np.diag([2]*3+[0]*11)+v[1]*np.diag([0]*3+[2]*3+[0]*8)+v[2]*np.diag([0]*6+[2]*8)

non_linear_const=NonlinearConstraint(const,0,1,jac=jacob,hess=hess)
# print(hess(0,[1,1,1]))

def inf_rel_entropy(rho):

    if rho.type == 'ket' or rho.type == 'bra':
        rho=ket2dm(rho)
    else:
        None

    def entropy_cond_sin_rho(x):
        return entropy_relative(rho,sep_dm(x[0],x[1:]))
    
    res=minimize(entropy_cond_sin_rho,np.zeros(14),method='trust-constr',constraints=non_linear_const,options={'verbose': 1})
    return res.x

# print()
res=inf_rel_entropy(sol.states[0])