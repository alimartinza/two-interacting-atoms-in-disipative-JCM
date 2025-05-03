from qutip import *
import numpy as np
import matplotlib.pyplot as plt
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

g0=tensor(gr,basis(3,0))
e0=tensor(e,basis(3,0))
g1=tensor(gr,basis(3,1))

w_0=1
# g=0.01*w_0 #atom-cavity coupling
# k=0 #atom-atom photon exchange rate
# J=0 #spin-spin coupling por lo que estuve viendo el efecto de k-J es enorme en la seleccion del estado estacionario
# d=2*g #atom frequency
# x=1/8*g #kerr medium

# #gamma/g>1 weak coupling es decir que el acople atom-field es weak en comparacion al entorno, gamma/g<1 strong coupling
# gamma=2*g
# p=0.005*g


def evolucion(w_0:float,g:float,k:float,J:float,d:float,x:float,gamma:float,p:float,psi0,t_final:int,steps:int,disipation:bool=True,acoplamiento:str='lineal'):
    #DEFINIMOS FUNCIONES PARA MEDIDAS QUE NOS GUSTARIA ANALIZAR

    def entropy_vn(rho):
        """
        Von-Neumann entropy of density matrix

        Parameters
        ----------
        rho : qobj or list of qobjs
            Density matrix.
        base : {e,2}
            Base of logarithm.
        sparse : {False,True}
            Use sparse eigensolver.

        Returns
        -------
        entropy : list of floats
            Von-Neumann entropy of `rho`.

        Examples
        --------
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_vn(rho,2)
        1.0

        """

        s=np.zeros(len(rho))
        eigenvals=np.empty([len(rho),12])
        eigenstatesvector=np.empty([len(rho),12,12,1],dtype="complex")
        for i in range(len(rho)):

            if rho[i].type == 'ket' or rho[i].type == 'bra':
                rho[i] = ket2dm(rho[i])
            rho[i]=rho[i].tidyup()
            eigenvals[i],eigenvecs = rho[i].eigenstates()
            nzvals = eigenvals[i][eigenvals[i] > 0]
            s[i] = float(np.real(-sum(nzvals * np.log(nzvals))))
            for j,vec in enumerate(eigenvecs):
                eigenstatesvector[i][j]=vec.full()

        return eigenvals,eigenvecs,s
    
    def entropy_vn_atom(rho):
        """
        Von-Neumann entropy of density matrix

        Parameters
        ----------
        rho : qobj or list of qobjs
            Density matrix.
        base : {e,2}
            Base of logarithm.
        sparse : {False,True}
            Use sparse eigensolver.

        Returns
        -------
        entropy : list of floats
            Von-Neumann entropy of `rho`.

        Examples
        --------
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_vn(rho,2)
        1.0

        """

        s=np.zeros(len(rho))
        for i in range(len(rho)):

            if rho[i].type == 'ket' or rho[i].type == 'bra':
                rho[i] = ket2dm(rho[i])
            rho[i]=rho[i].tidyup()
            eigenvals= rho[i].eigenenergies()
            nzvals = eigenvals[eigenvals > 0]
            s[i] = float(np.real(-sum(nzvals * np.log(nzvals))))

        return s

    def entropy_linear(rho):
        """
        Linear entropy of a density matrix.

        Parameters
        ----------
        rho : qobj or list of qobjs
            sensity matrix or ket/bra vector.

        Returns
        -------
        entropy : list of floats
            Linear entropy of rho.

        Examples
        --------
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_linear(rho)
        0.5

        """
        s=np.zeros(len(rho))
        for i in range(len(rho)):
            if rho[i].type == 'ket' or rho[i].type == 'bra':
                rho[i] = ket2dm(rho[i])
            
            s[i] = float(np.real(1.0 - (rho[i] ** 2).tr()))
        return s

    def concurrence(rho):
        """
        Calculate the concurrence entanglement measure for a two-qubit state.

        Parameters
        ----------
        state : qobj or list of qobjs
            Ket, bra, or density matrix for a two-qubit state.

        Returns
        -------
        concur : float or list of floats
            Concurrence

        References
        ----------

        .. [1] http://en.wikipedia.org/wiki/Concurrence_(quantum_computing)

        """
        c=np.zeros(len(rho))
        for i in range(len(rho)):
            if rho[i].isket and rho[i].dims != [[2, 2], [1, 1]]:
                raise Exception("Ket must be tensor product of two qubits.")

            elif rho[i].isbra and rho[i].dims != [[1, 1], [2, 2]]:
                raise Exception("Bra must be tensor product of two qubits.")

            elif rho[i].isoper and rho[i].dims != [[2, 2], [2, 2]]:
                raise Exception("Density matrix must be tensor product of two qubits.")

            if rho[i].isket or rho[i].isbra:
                rho[i] = ket2dm(rho[i])
        

            sysy = tensor(sigmay(), sigmay())

            rho_tilde = (rho[i] * sysy) * (rho[i].conj() * sysy)

            evals = rho_tilde.eigenenergies()

            # abs to avoid problems with sqrt for very small negative numbers
            evals = abs(np.sort(np.real(evals)))

            lsum = np.sqrt(evals[3]) - np.sqrt(evals[2]) - np.sqrt(evals[1]) - np.sqrt(evals[0])
            c[i]=max(0, lsum)
        return c
    
    #DEFINIMOS CUAL MODELO VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO

    
    def f():
        if acoplamiento=='lineal':
            return 1
        elif acoplamiento=='bs':
            return sqrtN

    
    #Espacio N=0 [9]

    # N=1 [3,4,10] N=2 [0,5,6,11] N=3 [1,7,8]

    #DEFINIMOS LA FUNCION PR QUE DADO UN ESTADO NOS DA SU PROYECTOR 

    def pr(estado):
        return estado.unit()*estado.unit().dag()

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
    
    ops_nomb=['pr(gg0)','pr(gg1)','pr(eg0+ge0)','pr(ge0-eg0)','pr(gg2)','pr(eg1+ge1)','pr(eg1-ge1)','pr(ee0)','pr(eg2)','pr(ge2)',
          'pr(ee1)','1/2 <sz1+sz2>','<sx1>','<sx2>'] #NOMBRES PARA EL LEGEND DEL PLOT
    ops = [pr(gg0),pr(gg1),pr(eg0+ge0),pr(ge0-eg0),pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0),pr(eg2),pr(ge2),pr(ee1),
           0.5*(sz1+sz2),sx1,sx2]
    expectStartTime=time.process_time()
    ops_expect=np.empty((len(ops),len(sol.states)))
    for i in range(len(sol.states)): 
        for j in range(len(ops)):
            ops_expect[j][i]=expect(ops[j],sol.states[i])
    expectRunTime=time.process_time()-expectStartTime
    """----Coherencias ---
    coherenciasStartTime = time.process_time()
    if not disipation:
        for i in range(len(sol.states)):
            for j in range(12): 
                for l in range(j+1,12):
                    coherencias[str(j)+','+str(l)].append(sol.states[i][j]*sol.states[i][l])        
    else:
        for j in range(12): 
            for l in range(j+1,12):
                c_help=np.zeros(len(sol.states),dtype='complex')
                for i in range(len(sol.states)):
                    c_help[i]=sol.states[i][j][l]
                    coherencias[str(j)+','+str(l)].append(c_help[i])
    coherenciasRunTime = time.process_time()-coherenciasStartTime
    """
    #CALCULAMOS COSAS INTERESANTES PARA EL SISTEMA
    pasajeStartTime=time.process_time()
    estados=np.empty_like(sol.states)
    for j in range(len(sol.states)):
        estados[j]=sol.states[j]

    data=pd.DataFrame()

    for nombres,valores_de_expectacion in zip(ops_nomb,ops_expect):
        data[nombres]=valores_de_expectacion
    pasajeRunTime=time.process_time() - pasajeStartTime
    entropiaStartTime = time.process_time()
    eigenvals,eigenvecs,data['S von Neuman tot']=entropy_vn(estados)
    data['S lineal tot']=entropy_linear(estados)
    atoms_states=np.empty_like(sol.states)
    for j in range(len(sol.states)):
        atoms_states[j]=sol.states[j].ptrace([0,1])
    
    # data['Atom States']=atoms_states
    data['S vN atom']=entropy_vn_atom(atoms_states)
    data['S lin atom']=entropy_linear(atoms_states)
    data['Concu atom']=concurrence(atoms_states)
    entropiaRunTime=time.process_time() - entropiaStartTime
    for i in range(12):
        data['Eigenvalue '+str(i)]=eigenvals[:,i]

    print("-----Tiempos de computo----")
    print(f"expectRunTime: {expectRunTime}",f"pasajeRunTime: {pasajeRunTime}",f"entropiaRunTime: {entropiaRunTime}",sep='\n') #,f"coherenciasRunTime: {coherenciasRunTime}"
    #PLOT PARA LA DINAMICA (POBLACIONES Y COHERENCIAS) DEL SIST. TRAZANDO SOBRE LOS FOTONES
    
    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    d_str=str(d).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')
    parameters_name=f"g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}"
    csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'
    #HACEMOS UN SUBPLOT PARA CADA ESPACIO DE N EXITACIONES + UNO PARA EL VALOR MEDIO DE LAS MATRICES DE PAULI
    '''--------------SAVE TO CSV OR TO QU FILE-------------------'''
    data.to_csv(csvname)
    fileio.qsave(sol.states,parameters_name+'sol states')
    fileio.qsave(atoms_states,parameters_name+'atom states')
    fileio.qsave(eigenvecs,parameters_name+'eigen states')


disipation=True
acoplamiento='lineal'
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
psi0=[ee0,gg1,eg0,gg2,(eg0-ge0)/np.sqrt(2),(eg1-ge1)/np.sqrt(2),(eg1+ge0)/np.sqrt(2),(eg1-ge0)/np.sqrt(2)]
psi0_folder=['ee0','gg1','eg0','gg2','eg0-','eg1-','eg1+ge0','eg1-ge0']

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
        x=[0]#,1/4*g,0.5*g]
        for x in x:
            d=[0]#,0.5*g,2*g]
            for d in d:
                gamma=[0.1*g]#,2*g]
                for gamma in gamma:
                    evolucion(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps,disipation=disipation,acoplamiento=acoplamiento)#,plot_show=True,save_plot=False)


'''----PARA PLOTS---'''
# for psi0,psi0_folder in zip(psi0,psi0_folder):
#     folders=['0','1','2','3','pauli','entropia','entropia_spin-spin']
#     for folder in folders:
#         folder_path=path+'\\'+psi0_folder+'\\'+folder
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#     psi0_path=path+'\\'+psi0_folder
#     os.chdir(psi0_path)
#     g=[0.001*w_0]
#     for g in g:
#         p=0.005*g
#         k=0.1*g
#         x=[0,1/4*g,0.5*g]
#         for x in x:
#             d=[0,0.5*g,2*g]
#             for d in d:
#                 gamma=[0.1*g,2*g]
#                 for gamma in gamma:
#                     main(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps,disipation=disipation)#,plot_show=True,save_plot=False)