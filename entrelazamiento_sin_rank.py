import numpy
from numpy.linalg import eigh
from pymanopt import Problem
from pymanopt.manifolds import UnitaryGroup
from pymanopt.optimizers import ConjugateGradient,SteepestDescent,ParticleSwarm,NelderMead
from pymanopt.function import autograd
# from pymanopt.function import numpy
import autograd.numpy as np
from autograd.numpy.linalg import grad_eigh
from qutip import *
import jcm_lib as jcm
import os
import time
import matplotlib.pyplot as plt


script_path= os.path.dirname(__file__)

N_c=3
steps=6000
g_t=30

w0=1
g=0.001*w0

gamma=0*g

x=0         #1*g va en orden ascendiente
d=0.001*g       #1.1001*g#.5*g

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

metodo=str(input('Metodo: '))

# --- DIFERENTES ESTADOS INICIALES --- #
def simulacion(d:float):
    rho_0=tensor((eg+ge).unit(),basis(N_c,1))

    t_final=g_t/g

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

    p=0.05*gamma

    sol=mesolve(H,rho_0,t)

    fg_total,arg_tot,eigenvals_tot_t=jcm.fases(sol)

    rho_01=np.empty_like(sol.states) #2x2
    for j in range(len(sol.states)):
        rho_01[j]=sol.states[j].ptrace([0,1])
    
    rho_02=np.empty_like(sol.states)  #2x3
    for j in range(len(sol.states)):
        rho_02[j]=sol.states[j].ptrace([0,2])

    rho_12=np.empty_like(sol.states) #2x3
    for j in range(len(sol.states)):
        rho_12[j]=sol.states[j].ptrace([1,2])

    return sol,rho_01,rho_02,rho_12,fg_total

sol,rho_01,rho_02,rho_12,fg_total=simulacion(d)
# print(sol.states[0])

# --- Optimized Entropy of entanglement ---
def tallis_entropy(psi_ij):
    '''
    Calculates de tallis entropy for q=2 (generalized concurrence) for a pure bipartite
    density matrix.
    
    Args:
        psi_ij: a bipartite pure state
        
    Returns:
        tallis entropy'''
    

# --- Stabilized Entropy Calculation ---
def entropy_of_entanglement_2x2(psi):
    row1 = psi[0:2]
    row2 = psi[2:4]
    a = np.sum(np.abs(row1)**2)
    d = np.sum(np.abs(row2)**2)
    b_conj = np.sum(np.conj(row1) * row2)
    mod_b_sq = np.abs(b_conj)**2

    discriminant = (a - d)**2 + 4 * mod_b_sq
    discriminant_sqrt = np.sqrt(np.maximum(discriminant, 0))  # CLIP NEGATIVES
    
    
    lambda1 = 0.5 * (1 + discriminant_sqrt)
    lambda2 = 0.5 * (1 - discriminant_sqrt)
    
    eigs = np.array([lambda1, lambda2])
    eigs = eigs[eigs > 1e-12]  # Filter near-zero
    
    eps = 1e-13
    return -np.sum(eigs * np.log2(eigs + eps))


def entropy_of_entanglement_2xN(psi):
    """
    Calculates the entanglement entropy of a pure state psi using an
    analytical formula for the eigenvalues of the 2x2 reduced density matrix.
    This is more efficient than numerical diagonalization.

    Args:
        psi: A 6x1 normalized pure state vector for a qubit-qudit system.

    Returns:
        The von Neumann entropy (in bits).
    """
    # The state is assumed to be normalized. The components c_ij correspond to
    # the qubit state i and qutrit state j.
    # row1 corresponds to qubit state |0>, with components c_00, c_01, c_02, ... , c_0N
    # row2 corresponds to qubit state |1>, with components c_10, c_11, c_12, ... , c_1N
    row1 = psi[0:N_c]
    row2 = psi[N_c:2*N_c]

    # The 2x2 reduced density matrix rho_A is [[a, b], [b*, d]].
    # a = |c_00|^2 + |c_01|^2 + |c_02|^2
    # d = |c_10|^2 + |c_11|^2 + |c_12|^2
    # b = c_00*c_10* + c_01*c_11* + c_02*c_12*
    a = np.sum(np.abs(row1)**2)
    d = np.sum(np.abs(row2)**2)

    # Manually implement vdot using autograd-safe primitives to avoid VJP error.
    # vdot(row1, row2) is sum(row1.conj() * row2)
    b_conj = np.sum(np.conj(row1) * row2) # This is b*
    mod_b_sq = np.abs(b_conj)**2

    # The eigenvalues are given by the analytical formula:
    # lambda = [ 1 +/- sqrt((a-d)^2 + 4|b|^2) ] / 2
    # Note: trace(rho_A) = a + d = 1 for a normalized state psi.
    discriminant_sqrt = np.sqrt((a - d)**2 + 4 * mod_b_sq)
    
    # Calculate the two eigenvalues
    lambda1 = 0.5 * (1 + discriminant_sqrt)
    lambda2 = 0.5 * (1 - discriminant_sqrt)

    # Create a small array of the two eigenvalues.
    eigs = np.array([lambda1, lambda2])
    
    # Filter out numerical noise (very small or negative eigenvalues).
    eigs = eigs[eigs > 1e-12]
    
    # Calculate von Neumann entropy: S = -sum(p * log2(p))
    # Add a small epsilon for numerical stability to prevent log(0) errors,
    # which can occur with pure states.
    eps = 1e-16
    return -np.sum(eigs * np.log2(eigs + eps))


#primero hacemos para la condicion inicial, y con esta tomamos la semilla y vamos 
#calculando la optimizacion paso a paso

def rankV_func(rho):
    eigvals, eigvecs = eigh(rho) #primero buscamos los autovalores y autovectores de la matriz dada por dos razones
    eigvals[eigvals<0]=0
    eigvals = np.clip(eigvals, 0.0, 1.0)
    keep = eigvals > 1e-12
    if not np.any(keep):
        raise ValueError("rho has zero effective rank")
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    rank = len(eigvals)
    V = np.dot(eigvecs, np.diag(np.sqrt(eigvals))) #la segunda y mas fundamental es para encontrar esta matriz que vendria a ser la descomposicion espectral de rho: rho=V^{\dagger}V. Esta matriz V nos sirve para ir transformandola con la matriz unitaria arbitraria e ir buscando el minimo de entropia
    return rank, V

def roof_etanglement_bipartite(rank,V,initial_point=None,dims=0):
    '''Calcula el entrelazamiento de un sistema bipartito mixto de dimensiones 
    arbitrarias 2xm usando convex roof de la entropia de von neuman.
    Parametros:
        rho: matriz densidad a la que se le quiere calcular el entrelazamiento. tiene que ser un ndarray no un qobj y tiene que ser de [2xN]x[2xN] dimensiones (la dimension 2 tiene que ir primero)
    Opcional:
        initial_point: el punto por el que se quiere comenenzar la optimizacion. Esta
        pensado para utilizarse en caso de tener una evolucion temporal y querer calcular el entrelazamiento en funcion del tiempo.'''

    # --- Manifold Definition ---
    # print(rank)
    # print(V)
    manifold = UnitaryGroup(rank) #definimos el grupo sobre el cual queremos hacer la optimizacion rho'=U^\dag V^\dag V U o algo asi. Usamos el grupo U(rank) porque eliminamos los autovalores que son 0 que no van a sumar a la entropia y esto nos ahorra tiempo de computo al tener que optimizar en un espacio menor
    @autograd(manifold)    
    # --- Cost Function with Numerical Differentiation ---
    def cost(U):
        # print(U)
        W = np.dot(V,U)
        total_entanglement = 0
        for i in range(rank):
            w_i = W[:, i]
            p_i = np.sum(np.abs(w_i)**2)
            if p_i < 1e-10:  # STRICTER THRESHOLD
                continue
            psi_i = w_i / np.sqrt(p_i)
            if dims==0:
                ent = entropy_of_entanglement_2xN(psi_i)
            elif dims==1:
                ent = entropy_of_entanglement_2x2(psi_i)
            else:
                print('incorrect dims for choosing entropy function')
                exit()

            total_entanglement += p_i * ent
        return total_entanglement

    # Create problem with numerical differentiation
    problem = Problem(
        manifold=manifold,
        cost=cost 
    )

    # Use identity matrix as initial point
    # initial_point = np.eye(rank)
    optimizer = ParticleSwarm(verbosity=1,max_iterations=5,population_size=25)
    # optimizer= SteepestDescent(verbosity=1,max_iterations=1000)
    # optimizer= ConjugateGradient(beta_rule="HagerZhang",verbosity=1,max_iterations=1000)
    # optimizer = NelderMead(verbosity=1)
    result = optimizer.run(problem,initial_point=initial_point)

    result = optimizer.run(problem,initial_point=initial_point)
    
    return result

# ---- rho_01 ----- #
print('--------- rho_01 ------------')
# step_size_list=[1e-10,1e-12,1e-14,1e-16]
# resultcost_stepsize=np.zeros((4,steps))
# for j,min_step in enumerate(step_size_list):
rank0_01,V0_01=rankV_func(rho_01[0].full())
result0_01=roof_etanglement_bipartite(rank0_01,V0_01,None,dims=1)
# print(rank0_01)
# print('V0_01',V0_01)

resultcost_01=[result0_01.cost]
resultrank_01=[rank0_01]

oldrank_01=rank0_01
oldpoint_01=result0_01.point
concu_01=jcm.concurrence(rho_01)

for i in range(1,len(rho_01)):
    rank_i_01,V_i_01=rankV_func(rho_01[i].full())

    # if rank_i_01==oldrank_01:
    #     initial_point=oldpoint_01
    # else: initial_point= None
    initial_point=None

    result_i_01=roof_etanglement_bipartite(rank_i_01,V_i_01,initial_point=initial_point ,dims=1)
    resultcost_01=np.append(resultcost_01,result_i_01.cost)
    resultrank_01=np.append(resultrank_01,rank_i_01)
    oldrank_01=rank_i_01
    oldpoint_01=result_i_01.point
    # resultcost_stepsize[j]=resultcost_01


# # ---- rho_02 ---- #
print('--------- rho_02 ------------')

print('--------- rho_02 ------------')

rank0_02,V0_02=rankV_func(rho_02[0].full())
result0_02=roof_etanglement_bipartite(rank0_02,V0_02,initial_point=initial_point)

resultcost_02=[result0_02.cost]
resultrank_02=[rank0_02]

oldrank_02=rank0_02
oldpoint_02=result0_02.point

for i in range(1,len(rho_02)):
    rank_i_02,V_i_02=rankV_func(rho_02[i].full())
    # if rank_i_01==oldrank_01:
    #     initial_point=oldpoint_01
    # else: initial_point= None
    initial_point=None
    result_i_02=roof_etanglement_bipartite(rank_i_02,V_i_02,initial_point=initial_point)
    resultcost_02=np.append(resultcost_02,result_i_02.cost)
    oldrank_02=rank_i_02
    oldpoint_02=result_i_02.point


# --- rho_12 ---#
print('--------- rho_12 ------------')

rank0_12,V0_12=rankV_func(rho_12[0].full())
result0_12=roof_etanglement_bipartite(rank0_12,V0_12,initial_point=None)


resultcost_12=[result0_12.cost]
resultrank_12=[rank0_12]

oldrank_12=rank0_12
oldpoint_12=result0_12.point


for i in range(1,len(rho_12)):
    rank_i_12,V_i_12=rankV_func(rho_12[i].full())
    # if rank_i_01==oldrank_01:
    #     initial_point=oldpoint_01
    # else: initial_point= None
    initial_point=None
    result_i_12=roof_etanglement_bipartite(rank_i_12,V_i_12,initial_point=initial_point)
    resultcost_12=np.append(resultcost_12,result_i_12.cost)
    oldrank_12=rank_i_12
    oldpoint_12=result_i_12.point

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

# --- MEdida tripartita --- #
def pure_tripartite_ent(E_01:list,E_02:list,E_12:list,alpha:float):
    Q=(E_01**alpha+E_02**alpha+E_12**alpha)
    A=(Q*(Q-E_01)*(Q-E_02)*(Q-E_12))**0.5
    return A

A_1=pure_tripartite_ent(resultcost_01,resultcost_02,resultcost_12,1)
A_05=pure_tripartite_ent(resultcost_01,resultcost_02,resultcost_12,0.5)

t_final=g_t/g
t=np.linspace(0,t_final,steps)

# metodo=input()
header=f'Metodo: {metodo}; steps: {steps}; gt_f= {g_t}; N_c= {N_c}; detunning = {d/g}g; x= {x/g}g; k= {k/g}g, J={J/g}g'


with open('CRE data/data_log.txt', "r+", encoding="utf-8") as f:
        # Read the first line only
        first_line = f.readline()
        parts = first_line.strip().split()

        # Get the number at the end
        try:
            current_number = int(parts[-1])
        except ValueError:
            raise ValueError("The last element of the first line is not a number.")

        # Increment
        new_number = current_number + 1

        # --- Update first line in place ---
        parts[-1] = str(new_number)
        new_first_line = " ".join(parts) + "\n"

        # Go back to start and overwrite only the first line
        f.seek(0)
        f.write(new_first_line)

        # Move to end of file to append
        f.seek(0, 2)
        f.write(f"{new_number}: "+header+"\n")


numpy.savetxt(f'CRE data/{new_number} resultcost.txt',[resultcost_01,resultcost_02,resultcost_12],header='E_01; E_02; E_12')


fig1=plt.figure(figsize=(8,6))
fig1.suptitle(r'$A(|\psi\rangle) ; \alpha=1$')
ax1=fig1.add_subplot()
ax1.set_xlim(0,g_t)
ax1.plot(g*t,A_1)

fig2=plt.figure(figsize=(8,6))
fig2.suptitle(r'$A(|\psi\rangle) ; \alpha=1/2$')
ax2=fig2.add_subplot()
ax2.set_xlim(0,g_t)
ax2.plot(g*t,A_05)

fig_ent=plt.figure(figsize=(8,6))
fig_ent.suptitle('Entrelazamientos reducidos')
ax_ent=fig_ent.add_subplot()
ax_ent.set_xlim(0,g_t)
ax_ent.plot(g*t,resultcost_01,color='red',label=r'$E_{01}$')#+f'step: {step_size:.1f}')   
# ax_ent.plot(g*t,concu_01,color='black',label='concu_01')
ax_ent.plot(g*t,resultcost_02,color='blue',label=r'$E_{02}$')
ax_ent.plot(g*t,resultcost_12,color='green',label=r'$E_{12}$')
# ax_ent.plot(g*t,A_05)
# ax_ent.plot(g*t,A_1)

plt.legend()


fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')
ax_fg=fig_fg.add_subplot()
ax_fg.set_xlim(0,t_final*g)
ax_fg.plot(g*t,fg_total/np.pi,color='black')

plt.show()