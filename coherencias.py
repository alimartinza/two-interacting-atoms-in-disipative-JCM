from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

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

script_path=os.path.dirname(__file__)

folder_names=["8_30_22 disipativo lineal","8_31_3 disipativo bs","8_31_8 unitario lineal","8_31_14 unitario bs"]
condiciones_iniciales=["ee0","gg1","eg0"]
# psi0_folder=['ee0','gg1','eg0','gg2','eg0-','eg1-','eg1+ge0','eg1-ge0']
w_0=1

def evo(w_0:float,g:float,k:float,J:float,d:float,x:float,gamma:float,p:float,psi0,t_final:int,steps:int,disipation:bool=True,acoplamiento:str='lineal'):
    
    def f():
        if acoplamiento=='lineal':
            return 1
        elif acoplamiento=='bs':
            return sqrtN
        
    '''---Hamiltoniano---'''

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    '''---Simulacion numerica---'''
   
    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    if disipation:
        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
    elif not disipation:
        l_ops=[]
    print("------Comenzando nuevas condiciones iniciales-----")
    sol=mesolve(H,psi0,t,c_ops=l_ops,progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD

    #Hacemos un array de las coherencias y las completamos con el for
    coherencias={'0,1':[],'0,2':[],'0,3':[],'0,4':[],'0,5':[],'0,6':[],'0,7':[],'0,8':[],'0,9':[],'0,10':[],'0,11':[],
                            '1,2':[],'1,3':[],'1,4':[],'1,5':[],'1,6':[],'1,7':[],'1,8':[],'1,9':[],'1,10':[],'1,11':[],
                                    '2,3':[],'2,4':[],'2,5':[],'2,6':[],'2,7':[],'2,8':[],'2,9':[],'2,10':[],'2,11':[],
                                            '3,4':[],'3,5':[],'3,6':[],'3,7':[],'3,8':[],'3,9':[],'3,10':[],'3,11':[],
                                                    '4,5':[],'4,6':[],'4,7':[],'4,8':[],'4,9':[],'4,10':[],'4,11':[],
                                                            '5,6':[],'5,7':[],'5,8':[],'5,9':[],'5,10':[],'5,11':[],
                                                                    '6,7':[],'6,8':[],'6,9':[],'6,10':[],'6,11':[],
                                                                            '7,8':[],'7,9':[],'7,10':[],'7,11':[],
                                                                                    '8,9':[],'8,10':[],'8,11':[],
                                                                                            '9,10':[],'9,11':[],
                                                                                                    '10,11':[]}
    data=pd.DataFrame()
    for key in coherencias.keys():
        data[key]=np.zeros(len(sol.states))

    coherenciasStartTime = time.process_time()
    if not disipation:
        for i in range(len(sol.states)):
            for j in range(12): 
                for l in range(j+1,12):
                    data[str(j)+','+str(l)][i]=sol.states[i][j]*sol.states[i][l]      
    else:
        for j in range(12): 
            for l in range(j+1,12):
                c_help=np.zeros(len(sol.states),dtype='complex')
                for i in range(len(sol.states)):
                    c_help[i]=sol.states[i][j][l]
                    data[str(j)+','+str(l)]=c_help



    coherenciasRunTime = time.process_time()-coherenciasStartTime
    print(f"coherenciasRunTime: {coherenciasRunTime}")

    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    d_str=str(d).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')
    parameters_name=f"g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}"
    csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str} coherencias.csv'
    #HACEMOS UN SUBPLOT PARA CADA ESPACIO DE N EXITACIONES + UNO PARA EL VALOR MEDIO DE LAS MATRICES DE PAULI
    '''--------------SAVE TO CSV OR TO QU FILE-------------------'''
    data.to_csv(csvname)



for disipation in [True,False]:
    for acoplamiento in ['lineal','bs']:
        yr, mes, dia, hr, minute = map(int, time.strftime("%Y %m %d %H %M").split())
        mesydiayhora=str(mes)+'_'+str(dia)+'_'+str(hr)
        script_path=os.path.dirname(__file__)
        if disipation:
            relative_path="datos"+"\\"+mesydiayhora+" disipativo "+acoplamiento
        elif not disipation:
            relative_path="datos"+"\\"+mesydiayhora+" unitario "+acoplamiento
        else:
            print("Error! disipation tiene que ser True o False!")
            exit()

        path=os.path.join(script_path, relative_path)

        if os.path.exists(path):
            os.chdir(path)
        else: 
            os.makedirs(path)
            os.chdir(path)

        J=0
        t_final=100000
        steps=100000
        psi0=[(ge0+eg0+gg1).unit(),ee0,gg1,eg0,gg2]
        psi0_folder=['W1','ee0','gg1','eg0','gg2']

        '''------GUARDAR DATAFRAME COMO CSV-------'''
        for psi0,psi0_folder in zip(psi0,psi0_folder):
            folder_path=path+'\\'+psi0_folder
            if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            os.chdir(folder_path)
            g=[0.001*w_0]
            for g in g:
                p=0.005*g
                k=0.1*g
                x=[0,1/4*g,0.5*g]
                for x in x:
                    d=[0,0.5*g,2*g]
                    for d in d:
                        gamma=[0.1*g,2*g]
                        for gamma in gamma:
                            evo(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps,disipation=disipation,acoplamiento=acoplamiento)#,plot_show=True,save_plot=False)
