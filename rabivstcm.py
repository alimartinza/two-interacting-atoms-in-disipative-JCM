import jcm_lib as jcm
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

N_c=10

# M=np.zeros((4*N_c,4*N_c))
# M[0,3*N_c]=1
# M[1,3*N_c+1]=1
# M[2,N_c]=1/np.sqrt(2)
# M[2,2*N_c]=1/np.sqrt(2)
# M[3,N_c]=1/np.sqrt(2)
# M[3,2*N_c]=-1/np.sqrt(2)

# for ii in range(1,N_c-1):
#     M[4*ii,3*N_c+1+ii]=1
# for ii in range(1,N_c-1):
#     M[4*ii+1,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
# for ii in range(1,N_c-1):
#     M[4*ii+2,ii-1]=1

ee=basis([2,2],[0,0])
eg=basis([2,2],[0,1])
ge=basis([2,2],[1,0])
gg=basis([2,2],[1,1])

def ggn(n):
    return tensor(gg,basis(N_c,n))

def egn(n):
    'SIMETRICO'
    return tensor((eg+ge).unit(),basis(N_c,n))

def een(n):
    return tensor(ee,basis(N_c,n))

def gen(n):
    'ANTISIMETRICO'
    return tensor((eg-ge).unit(),basis(N_c,n))

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



def simu_tcm_rabi(w_q:float,w_r:float,g:float,k:float,J:float,x:float,gamma:float,psi0,t_final:int=50000,steps:int=3000,return_all:bool=False):
    """Returns: 
    data_tcm: dataframe de pandas con los datos de la simulacion unitaria
    data_rabi: dataframe de pandas con los datos de la simulacion disipativa
    -Las keys de los dfs son:
    ,t,pr(gg0),pr(gg1),pr(eg0+ge0),pr(eg0-ge0),pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0),pr(eg2+ge2),pr(eg2-ge2),pr(ee1),1/2 <sz1+sz2>,<sx1>,<sx2>,0;1,0;2,0;3,0;4,0;5,0;6,0;7,0;8,0;9,0;10,0;11,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,2;3,2;4,2;5,2;6,2;7,2;8,2;9,2;10,2;11,3;4,3;5,3;6,3;7,3;8,3;9,3;10,3;11,4;5,4;6,4;7,4;8,4;9,4;10,4;11,5;6,5;7,5;8,5;9,5;10,5;11,6;7,6;8,6;9,6;10,6;11,7;8,7;9,7;10,7;11,8;9,8;10,8;11,9;10,9;11,10;11,FG,S von Neuman tot,S lineal tot,S vN atom,S lin atom,Concu atom,Eigenvalue 0,Eigenvalue 1,Eigenvalue 2,Eigenvalue 3,Eigenvalue 4,Eigenvalue 5,Eigenvalue 6,Eigenvalue 7,Eigenvalue 8,Eigenvalue 9,Eigenvalue 10,Eigenvalue 11
        """
    #DEFINIMOS CUAL MODELO VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO
    p=0.05*gamma

    def pr(estado):
        return estado.unit()*estado.unit().dag()

    '''---Hamiltoniano---'''

    H_tcm=w_r*n+x*n2 + w_q/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    H_rabi=w_r*n+x*n2 + w_q/2*(sz1+sz2) + g*(sx1+sx2)*(a+a.dag()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
    '''---Simulacion numerica---'''
    t_0=time.time()
    if gamma!=0: l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*sp1,np.sqrt(p)*sp2]
    else: l_ops=[]
    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION

    sol_tcm=mesolve(H_tcm,psi0,t,c_ops=l_ops)
    t_1=time.time()
    print(t_1-t_0,'s en simular jcm')

    sol_rabi=mesolve(H_rabi,psi0,t,c_ops=l_ops)
    t_2=time.time()
    print(t_2-t_1,'s en simular rabi')

    fg_tcm,arg,eigenvals_t_tcm = jcm.fases(sol_tcm)
    fg_rabi,arg,eigenvals_t_rabi = jcm.fases(sol_rabi)
    t_3=time.time()
    print(t_3-t_2,'s en simular las FGs')
    

    if return_all==False:
        atoms_states_tcm=np.empty_like(sol_tcm.states)
        for j in range(len(sol_tcm.states)):
            atoms_states_tcm[j]=sol_tcm.states[j].ptrace([0,1])

        atoms_states_rabi=np.empty_like(sol_rabi.states)
        for j in range(len(sol_rabi.states)):
            atoms_states_rabi[j]=sol_rabi.states[j].ptrace([0,1])
        concu_rabi=concurrence(atoms_states_rabi)
        concu_tcm=concurrence(atoms_states_tcm)
        return fg_tcm,fg_rabi,concu_tcm,concu_rabi
    else:
        data_tcm=pd.DataFrame()
        data_rabi=pd.DataFrame()
        data_tcm['t']=t
        data_rabi['t']=t
        #Hacemos un array de las coherencias y las completamos con el for
        coherencias_tcm={'0;1':[],'0;2':[],'0;3':[],'0;4':[],'0;5':[],'0;6':[],'0;7':[],'0;8':[],'0;9':[],'0;10':[],'0;11':[],
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
        
        coherencias_rabi=coherencias_tcm

        #DEFINIMOS LOS OPERADORES A LOS QUE QUEREMOS QUE EL SOLVER TOME VALOR MEDIO. LOS PROYECTORES NOS DAN LAS POBLACIONES
        ops_nomb=['pr(gg0)','pr(gg1)','pr(eg0)','pr(ge0)','pr(gg2)','pr(eg1)','pr(ge1)','pr(ee0)','pr(eg2)','pr(ge2)',
            'pr(ee1)','pr(ee2)','1/2 <Sz>','<sx1>','<sx2>','<n>'] #NOMBRES PARA EL LEGEND DEL PLOT
        ops = [pr(ggn(0)),pr(ggn(1)),pr(egn(0)),pr(gen(0)),pr(ggn(2)),pr(egn(1)),pr(gen(1)),pr(een(0)),pr(egn(2)),pr(gen(2)),pr(een(1)),pr(een(2)),
            0.5*(sz1+sz2),sx1,sx2,n]
        
        expectStartTime=time.process_time()
        ops_expect_tcm=np.empty((len(ops),len(sol_tcm.states)))
        for i in range(len(sol_tcm.states)): 
            for j in range(len(ops)):
                ops_expect_tcm[j][i]=expect(ops[j],sol_tcm.states[i])

        ops_expect_rabi=np.empty((len(ops),len(sol_rabi.states)))
        for i in range(len(sol_rabi.states)): 
            for j in range(len(ops)):
                ops_expect_rabi[j][i]=expect(ops[j],sol_rabi.states[i])

        for nombres,valores_rabie_expectacion_tcm in zip(ops_nomb,ops_expect_tcm):
            data_tcm[nombres]=valores_rabie_expectacion_tcm
        for key in coherencias_tcm.keys():
            data_tcm[key]=np.zeros(len(sol_tcm.states))
        for nombres,valores_rabie_expectacion_rabi in zip(ops_nomb,ops_expect_rabi):
            data_rabi[nombres]=valores_rabie_expectacion_rabi
        for key in coherencias_rabi.keys():
            data_rabi[key]=np.zeros(len(sol_rabi.states))
        #CALCULAMOS LAS COHERENCIAS Y LAS METEMOS EL EL DATAFRAME
        # coherenciasStartTime = time.process_time()

        # for j in range(12): 
        #     for l in range(j+1,12):
        #         c_help_tcm=np.zeros(len(sol_tcm.states),dtype='complex')
        #         for i in range(len(sol_tcm.states)):
        #             c_help_tcm[i]=(sol_tcm.states[i][j]*sol_tcm.states[i][l])[0]
        #         data_tcm[str(j)+';'+str(l)]=c_help_tcm

        # for j in range(12): 
        #     for l in range(j+1,12):
        #         c_help_rabi=np.zeros(len(sol_rabi.states),dtype='complex')
        #         for i in range(len(sol_rabi.states)):
        #             c_help_rabi[i]=sol_rabi.states[i][j][l]
        #         data_rabi[str(j)+';'+str(l)]=c_help_rabi

        # coherenciasRunTime = time.process_time()-coherenciasStartTime
        # print(f"coherenciasRunTime: {coherenciasRunTime}")
        data_tcm['fg']=fg_tcm
        data_rabi['fg']=fg_rabi

        expectRunTime=time.process_time()-expectStartTime

        #CALCULAMOS COSAS INTERESANTES PARA EL SISTEMA

        atoms_states_tcm=np.empty_like(sol_tcm.states)
        for j in range(len(sol_tcm.states)):
            atoms_states_tcm[j]=sol_tcm.states[j].ptrace([0,1])

        atoms_states_rabi=np.empty_like(sol_rabi.states)
        for j in range(len(sol_rabi.states)):
            atoms_states_rabi[j]=sol_rabi.states[j].ptrace([0,1])  

        data_tcm['conc_at_tcm']=jcm.concurrence(atoms_states_tcm)
        data_rabi['conc_at_rabi']=jcm.concurrence(atoms_states_rabi)

        return data_tcm,data_rabi


steps=6000
g_t=10

w_q=1
g=0.001*w_q
w_r=1+2*g

gamma=0       #.1*g

x=0         #1*g va en orden ascendiente
   #1.1001*g#.5*g

k=0*g        #0*g va en orden descendiente para ser consistente con la flecha dibujada mas abajo en el plot
J=0*g
psi0=egn(1)

tcm,rabi=simu_tcm_rabi(w_q,w_r,g,k,J,x,gamma,psi0,g_t/g,steps,True)

fig_pob=plt.figure(figsize=(12,9),tight_layout=True)
ax1=fig_pob.add_subplot(221)
ax1.plot(g*tcm['t'],tcm['pr(gg0)'],color='black',label='P|gg0>')
ax1.plot(g*tcm['t'],tcm['pr(gg1)'],color='red',label='P|gg0>')
ax1.plot(g*tcm['t'],tcm['pr(gg2)'],color='blue',label='P|gg0>')

ax1.plot(g*rabi['t'],rabi['pr(gg0)'],color='black',linestyle=' ',marker='o',markevery=int(steps/200))
ax1.plot(g*rabi['t'],rabi['pr(gg1)'],color='red',linestyle=' ',marker='o',markevery=int(steps/200))
ax1.plot(g*rabi['t'],rabi['pr(gg2)'],color='blue',linestyle=' ',marker='o',markevery=int(steps/200))
ax1.set_xlabel('gt')
ax1.set_ylabel(r'$P_{|ggn\rangle}$')
ax1.legend()

ax2=fig_pob.add_subplot(222)
ax2.plot(g*tcm['t'],tcm['pr(eg0)'],color='black',label='P|eg0>')
ax2.plot(g*tcm['t'],tcm['pr(eg1)'],color='red',label='P|eg0>')
ax2.plot(g*tcm['t'],tcm['pr(eg2)'],color='blue',label='P|eg0>')

ax2.plot(g*rabi['t'],rabi['pr(eg0)'],color='black',linestyle=' ',marker='o',markevery=int(steps/200))
ax2.plot(g*rabi['t'],rabi['pr(eg1)'],color='red',linestyle=' ',marker='o',markevery=int(steps/200))
ax2.plot(g*rabi['t'],rabi['pr(eg2)'],color='blue',linestyle=' ',marker='o',markevery=int(steps/200))
ax2.set_xlabel('gt')
ax2.legend()

ax3=fig_pob.add_subplot(223)
ax3.plot(g*tcm['t'],tcm['pr(ee0)'],color='black',label='P|ee0>')
ax3.plot(g*tcm['t'],tcm['pr(ee1)'],color='red',label='P|ee0>')
ax3.plot(g*tcm['t'],tcm['pr(ee2)'],color='blue',label='P|ee0>')

ax3.plot(g*rabi['t'],rabi['pr(ee0)'],color='black',linestyle=' ',marker='o',markevery=int(steps/200))
ax3.plot(g*rabi['t'],rabi['pr(ee1)'],color='red',linestyle=' ',marker='o',markevery=int(steps/200))
ax3.plot(g*rabi['t'],rabi['pr(ee2)'],color='blue',linestyle=' ',marker='o',markevery=int(steps/200))
ax3.set_xlabel('gt')
ax3.legend()

ax4=fig_pob.add_subplot(224)
ax4.plot(g*tcm['t'],tcm['<n>'],color='black',label='<n>')
ax4.plot(g*rabi['t'],rabi['<n>'],color='black',linestyle=' ',marker='o',markevery=int(steps/200))
ax4.set_ylabel(r'$\langle \hat{n} \rangle$')
ax4.set_xlabel('gt')
ax4.legend()

fig_fg=plt.figure(figsize=(8,6))
ax=fig_fg.add_subplot()
ax.plot(g*tcm['t'],tcm['fg']/np.pi,color='black',label='TCM')
ax.plot(g*rabi['t'],rabi['fg']/np.pi,color='red',label='RABI')
ax.set_xlabel('gt')
ax.set_ylabel('$\phi_G$')


plt.show()