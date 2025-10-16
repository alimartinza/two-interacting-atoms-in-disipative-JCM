import numpy as np
from numpy.linalg import eigh
from pymanopt import Problem
from pymanopt.manifolds import UnitaryGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers import SteepestDescent
from pymanopt.function import autograd
from autograd import numpy as anp
from autograd.numpy.linalg import eigvalsh
from qutip import *
import jcm_lib as jcm
import os
import time
import matplotlib.pyplot as plt


script_path= os.path.dirname(__file__)

N_c=3
steps=6000
g_t=10


cond_inic=0
modelos='TCM'

w0=1
g=0.001*w0


gamma=[0*g,0.1*g]       #.1*g
i_gamma=0
if i_gamma==0:
    disip='uni'
elif i_gamma==1:
    disip='dis'
else:
    print('i_gamma no esta en el rango deseado')
gamma=gamma[i_gamma]

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

# --- DIFERENTES ESTADOS INICIALES --- #
def simulacion(ci:int,d:float,modelo:str):
    if ci==0:
        #CAVIDAD EN FOCK CON NUMERO BIEN DEFINIDO PURO
        fase_pura=True
        fotones=1
        rho_0=tensor((eg+ge).unit(),basis(N_c,fotones))
    elif ci==1:
        #CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO PERO SIN COHERENCIAS ENTRE ESTADOS REDUCIDOS ATOMICOS PURO
        fase_pura=True
        rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0)+basis(N_c,1)))
        # rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0))+tensor(ee,basis(N_c,1))) 
    elif ci==3:
        #CAVIDAD COHERENTE PURO
        fase_pura=True
        rho_0=tensor(1/2*(eg+ge)*(eg+ge).dag(),coherent_dm(N_c,2)) 
    elif ci==4:
        #CAVIDAD EN FOCK CON NUMERO BIEN DEFINIDO PURO
        fase_pura=True
        fotones=2
        eef=tensor(ee,basis(N_c,fotones-2))
        ggf=tensor(gg,basis(N_c,fotones))
        rho_0=(eef+ggf).unit()*(eef+ggf).unit().dag()
    elif ci==5:
        #CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO PURO
        fase_pura=True
        rho_0=ket2dm(tensor((ee+gg).unit(),basis(N_c,0)+basis(N_c,1)))
        # rho_0=ket2dm(tensor((eg+ge).unit(),basis(N_c,0))+tensor(ee,basis(N_c,1)))
    elif ci==6:
        #CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO Y CON COHERENCIAS ENTRE ESTADOS REDUCIDOS ATOMICOS PURO
        fase_pura=True
        rho_0=ket2dm(tensor(((eg+ge).unit()+gg).unit(),basis(N_c,1)))
    elif ci==7:
        #CAVIDAD EN FOCK CON NUMERO NO BIEN DEFINIDO Y CON COHERENCIAS ENTRE ESTADOS REDUCIDOS ATOMICOS PURO
        fase_pura=True
        rho_0=ket2dm(tensor(((eg+ge).unit()+gg+ee).unit(),basis(N_c,1)))
    else:
        print('porfavor elegir una condicion inicial que este suporteada. Por default ci=0')
        ci=0

    t_final=g_t/g


    '''##########---Hamiltoniano---##########'''
    if modelo=='TCM' or modelo=='1':
        #Hamiltoniano de TC
        H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    elif modelo=='RABI' or modelo=='2':
        #Hamiltoniano de Rabi
        H=x*n2 + d/2*(sz1+sz2) + g*(sx1+sx2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    elif modelo=='SpinBoson' or modelo=='SB' or modelo=='3':
        #Hamiltoniano de "spin-boson"
        H=d/2*(sz1+sz2) + g*(sz1+sz2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    else:
        print('Este Modelo no existe. Modelo default es TCM.')
        H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    print(modelo,' ci:',ci )

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

    p=0.05*gamma

    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2]

    sol=mesolve(H,rho_0,t,)

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


rho,rho_01,rho_02,rho_12,fg_total=simulacion(cond_inic,d,modelos)

# --- Optimized Entropy of entanglement ---
def tallis_entropy(psi_ij):
    '''
    Calculates de tallis entropy for q=2 (generalized concurrence) for a pure bipartite
    density matrix.
    
    Args:
        psi_ij: a bipartite pure state
        
    Returns:
        tallis entropy'''
    

def entropy_of_entanglement_2x2(psi):
    """
    Calculates the entanglement entropy of a pure state psi using an
    analytical formula for the eigenvalues of the 2x2 reduced density matrix.
    This is more efficient than numerical diagonalization.

    Args:
        psi: A 4x1 normalized pure state vector for a qubit-qudit system.

    Returns:
        The von Neumann entropy (in bits).
    """
    # The state is assumed to be normalized. The components c_ij correspond to
    # the qubit state i and qutrit state j.
    # row1 corresponds to qubit state |0>, with components c_00, c_01, c_02, ... , c_0N
    # row2 corresponds to qubit state |1>, with components c_10, c_11, c_12, ... , c_1N
    row1 = psi[0:2]
    row2 = psi[2:4]

    # The 2x2 reduced density matrix rho_A is [[a, b], [b*, d]].
    # a = |c_00|^2 + |c_01|^2 + |c_02|^2
    # d = |c_10|^2 + |c_11|^2 + |c_12|^2
    # b = c_00*c_10* + c_01*c_11* + c_02*c_12*
    a = anp.sum(anp.abs(row1)**2)
    d = anp.sum(anp.abs(row2)**2)

    # Manually implement vdot using autograd-safe primitives to avoid VJP error.
    # vdot(row1, row2) is sum(row1.conj() * row2)
    b_conj = anp.sum(anp.conj(row1) * row2) # This is b*
    mod_b_sq = anp.abs(b_conj)**2

    # The eigenvalues are given by the analytical formula:
    # lambda = [ 1 +/- sqrt((a-d)^2 + 4|b|^2) ] / 2
    # Note: trace(rho_A) = a + d = 1 for a normalized state psi.
    discriminant_sqrt = anp.sqrt((a - d)**2 + 4 * mod_b_sq)
    
    # Calculate the two eigenvalues
    lambda1 = 0.5 * (1 + discriminant_sqrt)
    lambda2 = 0.5 * (1 - discriminant_sqrt)

    # Create a small array of the two eigenvalues.
    eigs = anp.array([lambda1, lambda2])
    
    # Filter out numerical noise (very small or negative eigenvalues).
    eigs = eigs[eigs > 1e-12]
    
    # Calculate von Neumann entropy: S = -sum(p * log2(p))
    # Add a small epsilon for numerical stability to prevent log(0) errors,
    # which can occur with pure states.
    eps = 1e-13
    return -anp.sum(eigs * anp.log2(eigs + eps))


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
    a = anp.sum(anp.abs(row1)**2)
    d = anp.sum(anp.abs(row2)**2)

    # Manually implement vdot using autograd-safe primitives to avoid VJP error.
    # vdot(row1, row2) is sum(row1.conj() * row2)
    b_conj = anp.sum(anp.conj(row1) * row2) # This is b*
    mod_b_sq = anp.abs(b_conj)**2

    # The eigenvalues are given by the analytical formula:
    # lambda = [ 1 +/- sqrt((a-d)^2 + 4|b|^2) ] / 2
    # Note: trace(rho_A) = a + d = 1 for a normalized state psi.
    discriminant_sqrt = anp.sqrt((a - d)**2 + 4 * mod_b_sq)
    
    # Calculate the two eigenvalues
    lambda1 = 0.5 * (1 + discriminant_sqrt)
    lambda2 = 0.5 * (1 - discriminant_sqrt)

    # Create a small array of the two eigenvalues.
    eigs = anp.array([lambda1, lambda2])
    
    # Filter out numerical noise (very small or negative eigenvalues).
    eigs = eigs[eigs > 1e-12]
    
    # Calculate von Neumann entropy: S = -sum(p * log2(p))
    # Add a small epsilon for numerical stability to prevent log(0) errors,
    # which can occur with pure states.
    eps = 1e-16
    return -anp.sum(eigs * anp.log2(eigs + eps))


#primero hacemos para la condicion inicial, y con esta tomamos la semilla y vamos 
#calculando la optimizacion paso a paso

def rankV_func(rho, tol=1e-12):
    # Hermitian eigendecomposition
    eigvals, eigvecs = eigh(rho)
    # Numerical safety: clip tiny negatives and drop negligible modes
    eigvals = np.clip(eigvals, 0.0, 1.0)
    keep = eigvals > tol
    if not np.any(keep):
        raise ValueError("rho has zero effective rank")
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    V = eigvecs @ np.diag(np.sqrt(eigvals))
    if not np.isfinite(V).all():
        raise ValueError("Non-finite entries in V")
    rank = int(eigvals.size)
    return rank, V.astype(np.complex128, copy=False)

def roof_etanglement_bipartite(rank, V, dims=0, initial_point=None):
    """
    Convex-roof entanglement via unitary mixing on the UnitaryGroup(rank).
    This version *never* uses random initial points: it starts from Identity.
    """
    # --- Validate inputs ---
    if not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError(f"Invalid rank={rank}, expected positive int")

    V = np.asarray(V, dtype=np.complex128)
    if V.ndim != 2 or V.shape[1] != rank:
        raise ValueError(f"V must be (n, rank). Got {V.shape} with rank={rank}")
    if not np.isfinite(V).all():
        raise ValueError("V contains NaN/Inf")

    # Convert once for autograd
    V_anp = anp.array(V)

    # --- Manifold ---
    manifold = UnitaryGroup(rank)

    # --- Cost (use autograd-safe ops only) ---
    @autograd(manifold)
    def cost(U):
        """
        Calculates the average entanglement for an ensemble decomposition of rho.
        """
        if np.isnan(U.any()):
            print('la concha de tu madre')
            print(U)
            exit()

        W = V @ U
        total_entanglement = 0
        # nan_instances=0
        for i in range(rank):
            w_i = W[:, i]
            p_i = anp.sum(anp.abs(w_i)**2)
            # Guard against non-finite/zero
            if (not anp.isfinite(p_i)) or (p_i <= eps):
                continue
            psi_i = w_i / anp.sqrt(p_i + eps)

            # if anp.isnan(psi_i).any():
            #     print("NaN in psi_i", psi_i)
            #     nan_instances+=1
            # elif anp.isnan(p_i):
            #     print("NaN in p_i", p_i)
            #     nan_instances+=1
            # if nan_instances>10:
            #     print('muchos nans, revisar porque')
            #     exit()

            # Call the optimized entropy function
            if dims==0:
                ent = max(0,entropy_of_entanglement_2xN(psi_i))
            elif dims==1:
                ent = max(0,entropy_of_entanglement_2x2(psi_i))
            else:
                print('Incorrect dims for entanglement measure')
                exit()

            # Clamp negative tiny values due to roundoff
            ent = anp.maximum(0.0, ent)
            if not anp.isfinite(ent):
                return anp.array(1e30)

            total = total + p_i * ent

        return total

    # --- Optimization: force a clean identity start ---
    if initial_point is None:
        init = np.eye(rank, dtype=np.complex128)
    else:
        init = np.asarray(initial_point, dtype=np.complex128)
        if init.shape != (rank, rank):
            raise ValueError(f"initial_point must be ({rank},{rank}); got {init.shape}")
        if not np.isfinite(init).all():
            init = np.eye(rank, dtype=np.complex128)

    problem = Problem(manifold=manifold, cost=cost)
    optimizer = SteepestDescent(verbosity=0)
    try:
        if initial_point==None:
            initial_point=manifold.random_point()
    except: None

    result = optimizer.run(problem,initial_point=initial_point)

    return result



rank0_01,V0_01=rankV_func(rho_01[0].full())
result0_01=roof_etanglement_bipartite(rank0_01,V0_01,dims=1)

resultcost_01=[result0_01.cost]
resultrank_01=[rank0_01]

oldrank_01=rank0_01
oldpoint_01=result0_01.point
print('--------- rho_01 ------------')
for i in range(1,len(rho_01)):
    rank_i_01,V_i_01=rankV_func(rho_01[i].full())
    result_i_01=roof_etanglement_bipartite(rank_i_01,V_i_01,dims=1,initial_point=oldpoint_01 if rank_i_01==oldrank_01 else None)
    resultcost_01=np.append(resultcost_01,result_i_01.cost)
    oldrank_01=rank_i_01
    oldpoint_01=result_i_01.point


# --- rho_02 ---#

rank0_02,V0_02=rankV_func(rho_02[0].full())
result0_02=roof_etanglement_bipartite(rank0_02,V0_02)

resultcost_02=[result0_02.cost]
resultrank_02=[rank0_02]

oldrank_02=rank0_02
oldpoint_02=result0_02.point
print('--------- rho_02 ------------')

for i in range(1,len(rho_02)):
    rank_i_02,V_i_02=rankV_func(rho_02[i].full())
    result_i_02=roof_etanglement_bipartite(rank_i_02,V_i_02,initial_point=oldpoint_02 if rank_i_02==oldrank_02 else None)
    resultcost_02=np.append(resultcost_02,result_i_02.cost)
    oldrank_02=rank_i_02
    oldpoint_02=result_i_02.point


# --- rho_12 ---#
rank0_12,V0_12=rankV_func(rho_12[0].full())
result0_12=roof_etanglement_bipartite(rank0_12,V0_12)

resultcost_12=[result0_12.cost]
resultrank_12=[rank0_12]

oldrank_12=rank0_12
oldpoint_12=result0_12.point

print('--------- rho_12 ------------')

for i in range(1,len(rho_12)):
    rank_i_12,V_i_12=rankV_func(rho_12[i].full())
    result_i_12=roof_etanglement_bipartite(rank_i_12,V_i_12,initial_point=oldpoint_12 if rank_i_12==oldrank_12 else None)
    resultcost_12=np.append(resultcost_12,result_i_12.cost)
    oldrank_12=rank_i_12
    oldpoint_12=result_i_12.point


# --- MEdida tripartita --- #
def pure_tripartite_ent(E_01:list,E_02:list,E_12:list,alpha:float):
    Q=(E_01**alpha+E_02**alpha+E_12**alpha)/2
    A=(Q*(Q-E_01)*(Q-E_02)*(Q-E_12))**0.5
    return A

A_1=pure_tripartite_ent(resultcost_01,resultcost_02,resultcost_12,1)
A_05=pure_tripartite_ent(resultcost_01,resultcost_02,resultcost_12,0.5)

t_final=g_t/g
t=np.linspace(0,t_final,steps)

fig1=plt.figure(figsize=(8,6))
fig1.suptitle(r'$A(|\psi\rangle) ; \alpha=1$')
ax1=fig1.add_subplot()
ax1.set_xlim(0,g_t)
ax1.plot(t,A_1)

fig2=plt.figure(figsize=(8,6))
fig2.suptitle(r'$A(|\psi\rangle) ; \alpha=1/2$')
ax2=fig2.add_subplot()
ax2.set_xlim(0,g_t)
ax2.plot(t,A_05)

fig_ent=plt.figure(figsize=(8,6))
fig_ent.suptitle('Entrelazamientos reducidos')
ax_ent=fig_ent.add_subplot()
ax_ent.set_xlim(0,g_t)
ax_ent.plot(t,resultcost_01,color='red',label=r'$E_{01}$')
ax_ent.plot(t,resultcost_02,color='blue',label=r'$E_{02}$')
ax_ent.plot(t,resultcost_12,color='black',label=r'$E_{12}$')

fig_fg=plt.figure(figsize=(8,6))
fig_fg.suptitle('FG')
ax_fg=fig_fg.add_subplot()
ax_fg.set_xlim(0,t_final*g)
ax_fg.plot(g*t,fg_total/np.pi,color='black')

plt.show()
