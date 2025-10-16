import numpy as np
# from pymanopt.function import numpy
from qutip import *
import jcm_lib as jcm
import os
import time
import matplotlib.pyplot as plt
from entrelazamiento_lib import negativity_hor,roof_etanglement_bipartite,rankV_func,pure_tripartite_ent

script_path = os.path.dirname(__file__) 

N_c=3
steps=3000
g_t=30

w0=1
g=0.001*w0

gamma=0.1*g

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

# metodo=str(input('Metodo: '))

# --- DIFERENTES ESTADOS INICIALES --- #
def simulacion(d:float):
    rho_0=tensor((eg+ge).unit(),basis(N_c,1))

    t_final=g_t/g

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 

    p=0.05*gamma

    sol=mesolve(H,rho_0,t)#,c_ops=[p*sp1,p*sp2,gamma*a])

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



def negativ():
    '''En total, para un sistema tripartito, hay 6 negatividades que podemos intentar
    de observar. Las primeras 3 se corresponden con elegir alguna biparticion, por ejemplo
    B|AC, y entonces tomamos transposicion parcial sobre B, y calculamos la negatividad. Esto seria
    como ver el entrelazamiento entre Bob y el conjunto de (Alice+Charlie). 
    Las otras 3 opciones son tomar traza parcial sobre alguno de los 3, es decir, los 
    olvidamos o pensamos que se estan aislando, y vemos la negatividad que resta. Por
    ejemplo, supongamos que Bob se aisla del mundo, entonces tenemos rho_AC, al cual le
    calculamos la negatividad, y nos dice cual es el entrelazamiento entre Alice y Charlie 
    si Bob no participa o se rehusa a compartir su informacion (pero sigue afectando el experimento)'''
    N_0=np.zeros(len(sol.states)) #estas tres son las negatividades que se obtienen haciendo
    N_1=np.zeros(len(sol.states)) # transpocicion parcial del subsystema indicado. Por ejejmplo
    N_2=np.zeros(len(sol.states)) # el N_1 hace partial_transpose sobre el sistema 1 (atomo B) y calcula la negatividad

    # N_01=np.zeros(len(sol.states)) # haciendo trasposicion parcial sobre 0 y 1 a la vez
    # N_02=np.zeros(len(sol.states)) # 
    # N_12=np.zeros(len(sol.states))

    N_01_t0=np.zeros(len(sol.states)) # a diferencia de lo de arriba, esto toma primero traza parcial y deja estados
    N_02_t0=np.zeros(len(sol.states)) # bipartitos, a los cuales se les calcula la negatividad.
    N_12_t1=np.zeros(len(sol.states))
    # t0=time.process_time()
    for j in range(len(sol.states)):
        
        # print(sol.states[j])
        N_0[j]=negativity_hor(sol.states[j],[1,0,0])
        N_1[j]=negativity_hor(sol.states[j],[0,1,0])
        N_2[j]=negativity_hor(sol.states[j],[0,0,1])

        # N_01[j]=negativity_hor(sol.states[j],[1,1,0])
        # N_02[j]=negativity_hor(sol.states[j],[1,0,1])
        # N_12[j]=negativity_hor(sol.states[j],[0,1,1])

        N_01_t0[j]=negativity_hor(sol.states[j].ptrace([0,1]),[1,0])
        N_02_t0[j]=negativity_hor(sol.states[j].ptrace([0,2]),[1,0])
        N_12_t1[j]=negativity_hor(sol.states[j].ptrace([1,2]),[1,0])
        # print(time.process_time()-t0)

    t_final=g_t/g
    t=np.linspace(0,t_final,steps)
    fig1=plt.figure(figsize=(8,6))
    fig1.suptitle(r'$\mathcal{N}(\rho)$')
    ax1=fig1.add_subplot()
    ax1.set_xlim(0,g_t)
    ax1.plot(g*t,N_0,'.g',label='A|BC')
    ax1.plot(g*t,N_1,'.r',label='B|AC')
    ax1.plot(g*t,N_2,'.b',label='C|AB')

    ax1.legend()


    fig3=plt.figure(figsize=(8,6))
    fig3.suptitle(r'$\mathcal{N}(\rho_{ij})$')
    ax3=fig3.add_subplot()
    ax3.set_xlim(0,g_t)
    ax3.plot(g*t,N_01_t0,'.g',label='ij=AB')
    ax3.plot(g*t,N_02_t0,'.r',label='ij=AC')
    ax3.plot(g*t,N_12_t1,'.b',label='ij=BC')
    ax3.legend()
    plt.show()
    return None

negativ()

