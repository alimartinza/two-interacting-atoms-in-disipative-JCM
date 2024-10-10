from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from jcm_lib import plot_coherencias,entropy_vn,entropy_vn_atom,entropy_linear,concurrence,fases


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


SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

def simu_unit_y_disip(w_0,g,k,J,d,x,gamma,p,psi0,t_final:int=50000,steps:int=3000,acoplamiento:str='lineal'):
    #DEFINIMOS CUAL MODELO VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO

    def f():
        if acoplamiento=='lineal':
            return 1
        elif acoplamiento=='bs':
            return sqrtN
  
    def pr(estado):
        return estado.unit()*estado.unit().dag()

    '''---Hamiltoniano---'''

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    '''---Simulacion numerica---'''
    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    sol_u=mesolve(H,psi0,t,c_ops=[],progress_bar=True)
    sol_d=mesolve(H,psi0,t,c_ops=l_ops,progress_bar=True)
    fg_u,arg,eigenvals_t_u = fases(sol_u)
    fg_d,arg,eigenvals_t_d = fases(sol_d)

    #Hacemos un array de las coherencias y las completamos con el for
    coherencias_u={'0;1':[],'0;2':[],'0;3':[],'0;4':[],'0;5':[],'0;6':[],'0;7':[],'0;8':[],'0;9':[],'0;10':[],'0;11':[],
                            '1;2':[],'1;3':[],'1;4':[],'1;5':[],'1;6':[],'1;7':[],'1;8':[],'1;9':[],'1;10':[],'1;11':[],
                                    '2;3':[],'2;4':[],'2;5':[],'2;6':[],'2;7':[],'2;8':[],'2;9':[],'2;10':[],'2;11':[],
                                            '3;4':[],'3;5':[],'3;6':[],'3;7':[],'3;8':[],'3;9':[],'3;10':[],'3;11':[],
                                                    '4;5':[],'4;6':[],'4;7':[],'4;8':[],'4;9':[],'4;10':[],'4;11':[],
                                                            '5;6':[],'5;7':[],'5;8':[],'5;9':[],'5;10':[],'5;11':[],
                                                                    '6;7':[],'6;8':[],'6;9':[],'6;10':[],'6;11':[],
                                                                            '7;8':[],'7;9':[],'7;10':[],'7;11':[],
                                                                                    '8;9':[],'8;10':[],'8;11':[],
                                                                                            '9;10':[],'9;11':[],
                                                                                                    '10;11':[]}
    
    coherencias_d=coherencias_u

    #DEFINIMOS LOS OPERADORES A LOS QUE QUEREMOS QUE EL SOLVER TOME VALOR MEDIO. LOS PROYECTORES NOS DAN LAS POBLACIONES
    ops_nomb=['pr(gg0)','pr(gg1)','pr(eg0+ge0)','pr(eg0-ge0)','pr(gg2)','pr(eg1+ge1)','pr(eg1-ge1)','pr(ee0)','pr(eg2+ge2)','pr(eg2-ge2)',
          'pr(ee1)','1/2 <sz1+sz2>','<sx1>','<sx2>'] #NOMBRES PARA EL LEGEND DEL PLOT
    ops = [pr(gg0),pr(gg1),pr(eg0+ge0),pr(eg0-ge0),pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0),pr(eg2+ge2),pr(eg2-ge2),pr(ee1),
           0.5*(sz1+sz2),sx1,sx2]
    
    expectStartTime=time.process_time()
    ops_expect_u=np.empty((len(ops),len(sol_u.states)))
    for i in range(len(sol_u.states)): 
        for j in range(len(ops)):
            ops_expect_u[j][i]=expect(ops[j],sol_u.states[i])

    ops_expect_d=np.empty((len(ops),len(sol_d.states)))
    for i in range(len(sol_d.states)): 
        for j in range(len(ops)):
            ops_expect_d[j][i]=expect(ops[j],sol_d.states[i])
    expectRunTime=time.process_time()-expectStartTime

    # #CALCULAMOS COSAS INTERESANTES PARA EL SISTEMA
    # pasajeStartTime=time.process_time()
    # estados=np.empty_like(sol.states)
    # for j in range(len(sol.states)):
    #     estados[j]=sol.states[j]


    #CALCULAMOS LAS COHERENCIAS Y LAS METEMOS EL EL DATAFRAME
    coherenciasStartTime = time.process_time()

    for j in range(12): 
        for l in range(j+1,12):
            c_help=np.zeros(len(sol_u.states),dtype='complex')
            for i in range(len(sol_u.states)):
                c_help[i]=(sol_u.states[i][j]*sol_u.states[i][l])[0]
            coherencias_u[str(j)+';'+str(l)]=c_help
    
    for j in range(12): 
        for l in range(j+1,12):
            c_help=np.zeros(len(sol_d.states),dtype='complex')
            for i in range(len(sol_d.states)):
                c_help[i]=sol_d.states[i][j][l]
            coherencias_d[str(j)+';'+str(l)]=c_help
    coherenciasRunTime = time.process_time()-coherenciasStartTime
    print(f"coherenciasRunTime: {coherenciasRunTime}")
 
    # pasajeRunTime=time.process_time() - pasajeStartTime
    entropiaStartTime = time.process_time()
    
    SvN_u=entropy_vn(eigenvals_t_u)
    SvN_d=entropy_vn(eigenvals_t_d)
    Slin_u=entropy_linear(sol_u.states)
    Slin_d=entropy_linear(sol_d.states)

    atoms_states_u=np.empty_like(sol_u.states)
    for j in range(len(sol_u.states)):
        atoms_states_u[j]=sol_u.states[j].ptrace([0,1])

    atoms_states_d=np.empty_like(sol_d.states)
    for j in range(len(sol_d.states)):
        atoms_states_d[j]=sol_d.states[j].ptrace([0,1])    
    # data['Atom States']=atoms_states
    SvN_at_u=entropy_vn_atom(atoms_states_u)
    Slin_at_u=entropy_linear(atoms_states_u)
    conc_at_u=concurrence(atoms_states_u)

    SvN_at_d=entropy_vn_atom(atoms_states_d)
    Slin_at_d=entropy_linear(atoms_states_d)
    conc_at_d=concurrence(atoms_states_d)

    entropiaRunTime=time.process_time() - entropiaStartTime

    print("-----Tiempos de computo----")
    print(f"expectRunTime: {expectRunTime}",f"pasajeRunTime: no existe",f"entropiaRunTime: {entropiaRunTime}",sep='\n') #,f"coherenciasRunTime: {coherenciasRunTime}"
    return ops_expect_u,ops_expect_d,coherencias_u,coherencias_d,fg_u,fg_d,SvN_u,SvN_d,Slin_u,Slin_d,SvN_at_u,SvN_at_d,Slin_at_u,Slin_at_d,conc_at_u,conc_at_d

psi0=(eg0+ge0).unit()
steps=3000
w_0=1
J=0
g=0.001*w_0
p=0.005*g
d=0
gamma=0.1*g
kappa=[0,0.1*g,0.2*g,0.3*g,0.4*g,0.5*g,0.6*g,0.7*g,0.8*g,0.9*g,g,1.1*g,1.2*g,1.3*g,1.4*g,1.5*g,1.6*g,1.7*g,1.8*g,1.9*g,2*g]

param=kappa
ops_expect_u=np.zeros(len(param),steps)
ops_expect_d=np.zeros(len(param),steps)
coherencias_u=np.zeros(len(param),steps)
coherencias_d=np.zeros(len(param),steps)
fg_u=np.zeros(len(param),steps)
fg_d=np.zeros(len(param),steps)
SvN_u=np.zeros(len(param),steps)
SvN_d=np.zeros(len(param),steps)
Slin_u=np.zeros(len(param),steps)
Slin_d=np.zeros(len(param),steps)
SvN_at_u=np.zeros(len(param),steps)
SvN_at_d=np.zeros(len(param),steps)
Slin_at_u=np.zeros(len(param),steps)
Slin_at_d=np.zeros(len(param),steps)
conc_at_u=np.zeros(len(param),steps)
conc_at_d=np.zeros(len(param),steps)
for i,k in enumerate(param):
    ops_expect_u[i],ops_expect_d[i],coherencias_u[i],coherencias_d[i],fg_u[i],fg_d[i],SvN_u[i],SvN_d[i],Slin_u[i],Slin_d[i],SvN_at_u[i],SvN_at_d[i],Slin_at_u[i],Slin_at_d[i],conc_at_u[i],conc_at_d[i]=simu_unit_y_disip(w_0,g,k,J,d,gamma,p,psi0,steps=steps)


