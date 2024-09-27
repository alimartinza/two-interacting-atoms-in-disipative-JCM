from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
import tkinter as tk
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

#Espacio N=0 [9]

# N=1 [3,6,10] N=2 [0,4,7,11] N=3 [1,5,8]

w_0=1
# g=0.01*w_0 #atom-cavity coupling
# k=0 #atom-atom photon exchange rate
# J=0 #spin-spin coupling por lo que estuve viendo el efecto de k-J es enorme en la seleccion del estado estacionario
# d=2*g #atom frequency
# x=1/8*g #kerr medium

# #gamma/g>1 weak coupling es decir que el acople atom-field es weak en comparacion al entorno, gamma/g<1 strong coupling
# gamma=2*g
# p=0.005*g


def evolucion(w_0:float,g:float,k:float,J:float,d:float,x:float,gamma:float,p:float,t_final:int,steps:int,disipation:bool=True,acoplamiento:str='lineal'):
    #DEFINIMOS CUAL MODELO VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO

    def f():
        if acoplamiento=='lineal':
            return 1
        elif acoplamiento=='bs':
            return sqrtN

    #Espacio N=0 [9]

    # N=1 [3,6,10] N=2 [0,4,7,11] N=3 [1,5,8]

    #DEFINIMOS LA FUNCION PR QUE DADO UN ESTADO NOS DA SU PROYECTOR 
    
    def pr(ket_):
        return ket_.unit()*ket_.unit().dag()

    '''---Hamiltoniano---'''

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

    '''---Simulacion numerica---'''

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    #DEFINIMOS LOS DISIPADORES SI HAY DISIPACION, Y SI NO, ENTONCES ESTA VACIO
    if disipation:
        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
    elif not disipation:
        l_ops=[]
    print("------Comenzando nuevas condiciones iniciales-----")
    sol=mesolve(H,eg0,t,c_ops=l_ops,progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD

    """
    ###################################################################
    ############# DEFINO fase ####################################
    ###################################################################
    """

    def fases(sol):
        """params:
        -sol: solucion numerica de la evolucion temporal
        RETURNS
        -fg_pan: Array de longitud len(t) donde con la FG de Pancho acumulada tiempo a tiempo
        -arg: no se
        -eigenvals: array de len(t)x12, entonces el elemento eigenvals[k] me da los 12 autovalores a tiempo t_k."""
        
        len_t=len(sol.states)
        if sol.states[0].type == 'ket' or sol.states[0].type == 'bra':
            rho0 = ket2dm(sol.states[0])
        else:
            rho0 = sol.states[0]
        eval0,evec0=rho0.eigenstates()
        eigenvals_t = np.array([eval0])
        max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
        psi0 = evec0[max_eigenvalue_idx]
        psi_old = psi0
        Psi = []
        norma = []
        pan = 0
        Pan = []
        argumento = np.zeros(len_t)
        signo = 0
        for i in range(len_t):
            if sol.states[i].type == 'ket' or sol.states[i].type == 'bra':
                rho = ket2dm(sol.states[i])
            else:
                rho = sol.states[i]
            
            eigenval,eigenvec = rho.eigenstates()
            eigenvals_t=np.concatenate((eigenvals_t,[eigenval]),axis=0)

            psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])
    
            # norma.append(psi.overlap(psi0))

            pan += np.angle(psi.overlap(psi_old))
            Pan.append(pan - np.angle(psi.overlap(psi0)))
            psi_old = psi

            # Almaceno el argumento para cada tiempo
            argumento[i] = np.angle(psi0.dag() * psi)

        eigenvals_t=np.delete(eigenvals_t,0,axis=0)
        Pan = np.array(Pan)
        return np.unwrap(Pan), argumento, np.array(eigenvals_t)

    def gp_pan(sol):
        """params:
        -sol: solucion numerica de la evolucion temporal"""
        len_t=len(sol.states)
        if sol.states[0].type == 'ket' or sol.states[0].type == 'bra':
            rho0 = ket2dm(sol.states[0])
        else:
            rho0 = sol.states[0]
        # Diagonalizar la matriz
        eval0,evec0=rho0.eigenstates() 
        max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
        psi0 = evec0[max_eigenvalue_idx]
        psi_old = psi0 
        pan = 0
        GP  = []
        EV  = []
        for i in range(len(t)):
            if sol.states[i].type == 'ket' or sol.states[i].type == 'bra':
                rho = ket2dm(sol.states[i])
            else:
                rho = sol.states[i]            
            eigenval,eigenvec = rho.eigenstates()
            psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])
            # eig, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])
            # overlap_max = psi.overlap(psi_old)
            pan += np.angle(overlap_max)
            GP.append(pan - np.angle(psi.overlap(psi0)))
            EV.append(psi)
            psi_old = psi
        GP = np.unwrap(np.array(GP))

        return GP,EV


    pan,arg,eigenvals_t = fases(sol)
    sol_sim=mesolve(H,(eg0+ge0).unit(),t,c_ops=l_ops,progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD
    sol_asim=mesolve(H,(eg0-ge0).unit(),t,c_ops=l_ops,progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD
    pan_sim,arg,eigenvals_t_sim=fases(sol_sim)
    pan_asim,arg,eigenvals_t_asim=fases(sol_asim)
    sol_w=mesolve(H,(ge0+eg0+2*gg1).unit(),t,c_ops=l_ops,progress_bar=True)
    pan_w,arg,eigenvals_t_sim=fases(sol_w)

    """#########################################
    ####    GRAFICOS AUTOVALORES NUMERICOS  ####
    # ##########################################"""
    
    cmap=mpl.colormaps["plasma"]
    colors=cmap(np.linspace(0,1,12))
    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()
    for i,evals in enumerate(eigenvals_t.transpose()):
        ax_eval.scatter(g*t,evals,color=colors[i],label=f"$\lambda_{i}$")
    ax_eval.set_xlabel(r"$gt$")
    ax_eval.set_ylabel("Autovalores")
    plt.show()

    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()
    for i,evals in enumerate(eigenvals_t_sim.transpose()):
        ax_eval.scatter(g*t,evals,color=colors[i],label=f"$\lambda_{i}$")
    ax_eval.set_xlabel(r"$gt$")
    ax_eval.set_ylabel("Autovalores sim")
    plt.show()

    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()
    for i,evals in enumerate(eigenvals_t_asim.transpose()):
        ax_eval.scatter(g*t,evals,color=colors[i],label=f"$\lambda_{i}$")
    ax_eval.set_xlabel(r"$gt$")
    ax_eval.set_ylabel("Autovalores asim")
    plt.show()

    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()
    for i,evals in enumerate(eigenvals_t_asim.transpose()):
        ax_eval.scatter(g*t,evals,color=colors[i],label=f"$\lambda_{i}$")
    ax_eval.set_xlabel(r"$gt$")
    ax_eval.set_ylabel("Autovalores w")
    plt.show()

    fig_fg=plt.figure()
    fig_fg.suptitle("Fase Geometrica")
    ax_fg=fig_fg.add_subplot()
    ax_fg.plot(g*t,pan,color='black',label=r"$\psi_0=|eg0>$")
    ax_fg.plot(g*t,pan_sim,color='red',label=r"$\psi_0=|eg0>+|ge0>$")
    ax_fg.plot(g*t,pan_asim,color='blue',label=r"$\psi_0=|eg0>-|ge0>$")
    ax_fg.plot(g*t,pan_w,color='grey',label=r"$\psi_0=|w>$")
    ax_fg.legend()
    ax_fg.grid()
    plt.show()

    #GUARDAMOS EL DATAFRAME EN CSV. 
    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    d_str=str(d).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')
    # parameters_name=f"g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}"
    csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'

    '''--------------SAVE TO CSV OR TO QU FILE-------------------'''
    # data.to_csv(csvname)
    #EN ESTA VERSION NO GUARDAMOS LOS ESTADOS DIAGONALIZADOS PORQUE OCUPAN ESPACIO
    #  Y TIEMPO Y QUIZAS COMBIENE HACERLO SOLO ESPECIALMENTE PARA LA SIMULACION QUE QUEREMOS ANALIZAR
    # fileio.qsave(eigenvecs,parameters_name+'eigen states')

disipation=False
acoplamiento='bs'
g=0.001*w_0
p=0.005*g
k=0.1*g
x=0.5*g
d=0# 0.5*g
gamma=0.1*g
J=0
t_final=25000
steps=2000
psi0=(eg0+ge0).unit() #(ge0+eg0+2*gg1).unit()
evolucion(w_0,g,k,J,d,x,gamma,p,t_final,steps,disipation=disipation,acoplamiento=acoplamiento)

# for disipation in [True,False]:
#     for acoplamiento in ['lineal','bs']:
#         yr, mes, dia, hr, minute = map(int, time.strftime("%Y %m %d %H %M").split())
#         mesydiayhora=str(mes)+'_'+str(dia)+'_'+str(hr)
#         script_path=os.path.dirname(__file__)
#         if disipation:
#             relative_path="datos"+"\\"+mesydiayhora+" disipativo "+acoplamiento
#         elif not disipation:
#             relative_path="datos"+"\\"+mesydiayhora+" unitario "+acoplamiento
#         else:
#             print("Error! disipation tiene que ser True o False!")
#             exit()

#         path=os.path.join(script_path, relative_path)

#         if os.path.exists(path):
#             os.chdir(path)
#         else: 
#             os.makedirs(path)
#             os.chdir(path)

#         J=0
#         t_final=100000
#         steps=100000
#         psi0=[(ge0+eg0+2*gg1).unit()]#ee0,gg1,eg0,gg2,(eg0-ge0)/np.sqrt(2),(eg1-ge1)/np.sqrt(2),(eg1+ge0)/np.sqrt(2),(eg1-ge0)/np.sqrt(2)]
#         psi0_folder=['W2']#'ee0','gg1','eg0','gg2','eg0-','eg1-','eg1+ge0','eg1-ge0']

#         '''------GUARDAR DATAFRAME COMO CSV-------'''
#         for psi0,psi0_folder in zip(psi0,psi0_folder):
#             folder_path=path+'\\'+psi0_folder
#             if not os.path.exists(folder_path):
#                     os.makedirs(folder_path)
#             os.chdir(folder_path)
#             g=[0.001*w_0]
#             for g in g:
#                 p=0.005*g
#                 k=0.1*g
#                 x=[0,1/4*g,0.5*g]
#                 for x in x:
#                     d=[0,0.5*g,2*g]
#                     for d in d:
#                         gamma=[0.1*g,2*g]
#                         for gamma in gamma:
#                             evolucion(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps,disipation=disipation,acoplamiento=acoplamiento)

