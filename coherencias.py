from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

script_path=os.path.dirname(__file__)

folder_names=["8_30_22 disipativo lineal","8_31_3 disipativo bs","8_31_8 unitario lineal","8_31_14 unitario bs"]
condiciones_iniciales=["ee0","gg1","eg0"]
# psi0_folder=['ee0','gg1','eg0','gg2','eg0-','eg1-','eg1+ge0','eg1-ge0']
w0=1
J=0
g=0.001*w0
k=0.1*g
p=0.005*g
t_final=100000
steps=100000
t=np.linspace(0,t_final,steps)
save_plot=False
plot_show=True

for ci in condiciones_iniciales:
    for folder_names in folder_names:

        relative_path="datos"+"\\"+folder_names+"\\"+ci
        path=os.path.join(script_path, relative_path)
        if os.path.exists(path):
            os.chdir(path)
        else: 
            print("Dir %s does not exist", path)


        x=[0,1/4*g,1/2*g]
        d=[0,0.5*g,2*g]
        gamma=[0.1*g,2*g]


        for m,gamma_m in enumerate(gamma):
            g_str=str(g).replace('.','_')
            k_str=str(k).replace('.','_')
            J_str=str(J).replace('.','_')
            d_str=str(d).replace('.','_')
            x_str=str(x).replace('.','_')
            gamma_m_str=str(gamma_m).replace('.','_')
            p_str=str(p).replace('.','_')
            
            param_name=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_m_str} p={p_str}'

            sol_states=fileio.qload(param_name+'sol states')
            # atom_states=fileio.qload(param_name+'atom states')
            # eigen_states=fileio.qload(param_name+'eigen states')
            
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

            coherenciasStartTime = time.process_time()
            for folname in folder_names:
                if folder_names.split(' ')[1]=='unitario': 
                    for i in range(len(sol_states)):
                        for j in range(12): 
                            for l in range(j+1,12):
                                coherencias[str(j)+','+str(l)].append(sol_states[i][j]*sol_states[i][l])        

                elif folder_names.split(' ')[1]=='disipativo':
                    for j in range(12): 
                        for l in range(j+1,12):
                            c_help=np.zeros(len(sol_states),dtype='complex')
                            for i in range(len(sol_states)):
                                c_help[i]=sol_states[i][j][l]
                                coherencias[str(j)+','+str(l)].append(c_help[i])
            coherenciasRunTime = time.process_time()-coherenciasStartTime
