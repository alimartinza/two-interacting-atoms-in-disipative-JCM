from qutip import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap 
import time
import os

#DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
n=tensor(qeye(2),qeye(2),num(3))
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
ge0=tensor(gr,e,basis(3,0)) #4

eg1=tensor(e,gr,basis(3,1)) #5
ge1=tensor(gr,e,basis(3,1)) #6

eg2=tensor(e,gr,basis(3,2)) #7
ge2=tensor(gr,e,basis(3,2)) #8

gg0=tensor(gr,gr,basis(3,0)) #9
gg1=tensor(gr,gr,basis(3,1)) #10
gg2=tensor(gr,gr,basis(3,2)) #11

w_0=1
# g=0.01*w_0 #atom-cavity coupling
# k=0 #atom-atom photon exchange rate
# J=0 #spin-spin coupling por lo que estuve viendo el efecto de k-J es enorme en la seleccion del estado estacionario
# d=2*g #atom frequency
# x=1/8*g #kerr medium

# #gamma/g>1 weak coupling es decir que el acople atom-field es weak en comparacion al entorno, gamma/g<1 strong coupling
# gamma=2*g
# p=0.005*g


def main(w_0:float,g:float,k:float,J:float,d:float,x:float,gamma:float,p:float,psi0,t_final:int,steps:int,disipation=True,plot_show=False,save_plot=True):
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
        for i in range(len(rho)):

            if rho[i].type == 'ket' or rho[i].type == 'bra':
                rho[i] = ket2dm(rho[i])
            if not rho[i].check_herm():
                print(f'La matriz en el paso {i} no es hermitica, la truncamos')
                # rho[i]=rho[i].trunc.neg()
            vals = np.array(rho[i].eigenenergies())
            nzvals = vals[vals > 0]
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

    medio= 'kerr' #int(input('Escribir lineal: h(n)=1,kerr: h(n)=1+x/w_0 *n'))
    acoplamiento = 'lineal' #int(input('Escribir lineal: f(n)=1,2:bs (Buck-Sukumar): f(n)=np.sqrt(n)'))

    modelo={'medio':medio,
            'acoplamiento':acoplamiento}

    def h(n):
        if modelo['medio']=='lineal':
            return 1
        elif modelo['medio']=='kerr':
            return 1+x/w_0*n
    def F(n):
        return n*(h(n)-1)
    def f(n):
        if modelo['acoplamiento']=='lineal':
            return 1
        elif modelo['acoplamiento']=='bs':
            return np.sqrt(n)

    
    #Espacio N=0 [9]

    # N=1 [3,4,10] N=2 [0,5,6,11] N=3 [1,7,8]

    #DEFINIMOS LA FUNCION PR QUE DADO UN ESTADO NOS DA SU PROYECTOR 

    def pr(estado):
        return estado.unit()*estado.unit().dag()


    proy=[[pr(gg0)],[pr(gg1),pr(eg0+ge0),pr(ge0-eg0)],[pr(gg2),pr(eg1+ge1),pr(eg1-ge1),pr(ee0)],[pr(eg2),pr(ge2),pr(ee1)]] #VECTOR CON LAS POBLACIONES QUE QUEREMOS MIRAR
    nomb=[['pr(gg0)'],['pr(gg1)','pr(eg0+ge0)','pr(ge0-eg0)'],['pr(gg2)','pr(eg1+ge1)','pr(eg1-ge1)','pr(ee0)'],['pr(eg2)','pr(ge2)','pr(ee1)']] #NOMBRES PARA EL LEGEND DEL PLOT

    # PARA HACER LA EVOLUCION NO UNITARIA, SI CONSIDERAMOS QUE TENEMOS UNA ECUACION MAESTRA DE TIPO LINBLAD,
    # ENTONCES LO UNICO QUE TENEMOS QUE HACER ES USAR EL SOLVER mesolve(H, psi0,t,[operadores de linblad],[expect operators])

    #FIJAMOS LOS PARAMETROS QUE VAMOS A USAR A LOS VALORES QUE QUERRAMOS

    '''---Hamiltoniano---'''
    #ESCRIBIMOS EL HAMILTONIANO DE NUESTRO MODELO UTILIZANDO TODO LO QUE DEFINIMOS ANTES, y tambien vamos a visualizarlo
    #usando una funcion de QuTip que nos permite representar la matriz con un histograma. El orden de la base es 
    #cualquier cosa por eso no tiene mucho sentido mirarla si no le ponemos bien los labels.

    H=w_0*n*(h(n)-1)+d/2*(sz1+sz2)+g*((sm1+sm2)*f(n)*a.dag()+(sp1+sp2)*a*f(n)) +2*k*(sm1*sp2+sp1*sm2)+J*sz1*sz2

    # '''---Steady states---'''

    # rho_ss = steadystate(H, [np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)])
    # hinton(rho_ss)

    # plt.show()

    '''---Simulacion numerica---'''

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 


    l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
    sol=mesolve(H,psi0,t,c_ops=l_ops,e_ops=proy[0]+proy[1]+proy[2]+proy[3]+[0.5*(sz1+sz2),sz1,sz2,sx1,sx2],progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD

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

    for j in range(12): 
        for l in range(j+1,12):
            c_help=np.zeros(len(sol.states),dtype='complex_')
            for i in range(len(sol.states)):
                c_help[i]=sol.states[i][j][l]
                coherencias[str(j)+','+str(l)].append(c_help[i])

    #CALCULAMOS COSAS INTERESANTES PARA EL SISTEMA
    estados=np.empty_like(sol.states)
    for j in range(len(sol.states)):
        estados[j]=sol.states[j]
    S_vn_tot=entropy_vn(estados)
    S_lin_tot=entropy_linear(estados)

    def plot_ReIm_coherencias(n:int,n_ax:int,xlabel=None,ylabel=None):
        '''
        Parametros
        - n: numero del vector de la base del cual se quieren graficar las coherencias
        -n_ax: en que ax queres graficar todas las coherencias
        
        Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
        colors = plt.cm.jet(np.linspace(0,1,12))
        i=0
        for key in coherencias.keys():
            if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                    ax[n_ax].plot(g*t,np.real(coherencias[key]),linestyle='dashed',label=f'Re[C({key})]',color=colors[i])
                    ax[n_ax].plot(g*t,np.imag(coherencias[key]),linestyle='dashdot',label=f'Im[C({key})]',color=colors[i])
                    i+=1
        ax[n_ax].legend()
        ax[n_ax].set_xlabel(xlabel)
        ax[n_ax].set_ylabel(ylabel)

    def plot_coherencias(n:int,n_ax:int,xlabel='t',ylabel='Abs(Coh)'):
        '''
        Parametros
        - n: numero del vector de la base del cual se quieren graficar las coherencias
        -n_ax: en que ax queres graficar todas las coherencias
        
        Pensado para usarlo semimanualmente, usar un plt.plots() e ir poniendo esta funcion en cada lugar donde queremos graficar las coherencias'''
        colors = plt.cm.jet(np.linspace(0,1,12))
        i=0
        if n==1:
            for key in ['0,1','1,2','1,3','1,4','1,5','1,6','1,7','1,8','1,9','1,10','1,11']:
                ax[n_ax].plot(g*t,np.abs(coherencias[key]),linestyle='dashed',color=colors[i]) #,label=f'C({key})'
                i+=1
        else:
            for key in coherencias.keys():
                if key.split(',')[0].startswith(str(n)) or key.split(',')[1].startswith(str(n)):
                        ax[n_ax].plot(g*t,np.abs(coherencias[key]),linestyle='dashed',color=colors[i]) #,label=f'C({key})'
                        i+=1
        ax[n_ax].legend()
        # ax[n_ax].set_xlabel(xlabel)
        ax[n_ax].set_ylabel(ylabel)


    '''---------------PLOTS-----------------------'''
    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    d_str=str(d).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')
    figname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}'
    #HACEMOS UN SUBPLOT PARA CADA ESPACIO DE N EXITACIONES + UNO PARA EL VALOR MEDIO DE LAS MATRICES DE PAULI

    '''--- N=0 ---'''
    fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
    ax=[ax]
    fig.suptitle('N=0')
    ax[0].plot(g*t,sol.expect[0],label=nomb[0][0],color='black')
    plot_coherencias(9,0) #N=0
    # plt.savefig(f'0\{figname}',dpi=100)

    if save_plot==True:
        plt.savefig(f'0\{figname}',dpi=100)
    else:
        None
    if plot_show==True:
        plt.show()
    else: 
        None    
    plt.close()
    '''--- N=1 ---'''
    fig,ax=plt.subplots(3,1,figsize=(16, 9),sharex=True) 
    fig.suptitle('N=1')
    ax[0].plot(g*t,sol.expect[1],label=nomb[1][0],color='black')
    ax[0].plot(g*t,sol.expect[2],label=nomb[1][1],color='blue')
    ax[0].plot(g*t,sol.expect[3],label=nomb[1][2],color='red')
    plot_coherencias(3,0) #N=1
    ax[1].plot(g*t,sol.expect[1],label=nomb[1][0],color='black')
    ax[1].plot(g*t,sol.expect[2],label=nomb[1][1],color='blue')
    ax[1].plot(g*t,sol.expect[3],label=nomb[1][2],color='red')
    plot_coherencias(4,1) #N=1
    ax[2].plot(g*t,sol.expect[1],label=nomb[1][0],color='black')
    ax[2].plot(g*t,sol.expect[2],label=nomb[1][1],color='blue')
    ax[2].plot(g*t,sol.expect[3],label=nomb[1][2],color='red')
    ax[2].set_xlabel('t')
    plot_coherencias(10,2) #N=1
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'1\{figname}',dpi=100)
    else: 
        None
    plt.close()
    '''--- N=2 ---'''
    fig,ax=plt.subplots(2,2,figsize=(16, 9),tight_layout=True,sharex=True) 
    ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
    fig.suptitle('N=2')
    ax[0].plot(g*t,sol.expect[4],label=nomb[2][0],color='black')
    ax[0].plot(g*t,sol.expect[5],label=nomb[2][1],color='blue')
    ax[0].plot(g*t,sol.expect[6],label=nomb[2][2],color='red')
    ax[0].plot(g*t,sol.expect[7],label=nomb[2][3],color='green')
    plot_coherencias(0,0) #N=2

    ax[1].plot(g*t,sol.expect[4],label=nomb[2][0],color='black')
    ax[1].plot(g*t,sol.expect[5],label=nomb[2][1],color='blue')
    ax[1].plot(g*t,sol.expect[6],label=nomb[2][2],color='red')
    ax[1].plot(g*t,sol.expect[7],label=nomb[2][3],color='green')
    plot_coherencias(5,1) #N=2

    ax[2].plot(g*t,sol.expect[4],label=nomb[2][0],color='black')
    ax[2].plot(g*t,sol.expect[5],label=nomb[2][1],color='blue')
    ax[2].plot(g*t,sol.expect[6],label=nomb[2][2],color='red')
    ax[2].plot(g*t,sol.expect[7],label=nomb[2][3],color='green')
    ax[2].set_xlabel('t')
    plot_coherencias(6,2) #N=2

    ax[3].plot(g*t,sol.expect[4],label=nomb[2][0],color='black')
    ax[3].plot(g*t,sol.expect[5],label=nomb[2][1],color='blue')
    ax[3].plot(g*t,sol.expect[6],label=nomb[2][2],color='red')
    ax[3].plot(g*t,sol.expect[7],label=nomb[2][3],color='green')
    ax[3].set_xlabel('t')
    plot_coherencias(11,3) #N=2
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'2\{figname}',dpi=100)
    else: 
        None
    plt.close()
    '''--- N=3 ---'''

    fig,ax=plt.subplots(1,1,figsize=(16, 9)) 
    ax=[ax]
    fig.suptitle('N=3')
    ax[0].plot(g*t,sol.expect[8],label=nomb[3][0],color='black')
    ax[0].plot(g*t,sol.expect[9],label=nomb[3][1],color='blue')
    ax[0].plot(g*t,sol.expect[10],label=nomb[3][2],color='red')
    plot_coherencias(1,0) #N=3
    plot_coherencias(7,0) #N=3
    plot_coherencias(8,0) #N=3
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'3\{figname}',dpi=100)
    else: 
        None
    plt.close()
    '''--- VM Pauli ---'''
    fig,ax=plt.subplots(1,1,figsize=(16, 9))
    fig.suptitle('V.M. Pauli')
    plt.plot(g*t,sol.expect[11],label='2*D=<sz1+sz2>',color='black')
    plt.plot(g*t,sol.expect[12],label='<sz1>',color='blue')
    plt.plot(g*t,sol.expect[13],label='<sz2>',color='red')
    plt.legend()
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'pauli\{figname}',dpi=100)
    else: 
        None
    plt.close()

    '''--- Entropias ---'''
    #PLOT PARA LAS ENTROPIAS
    fig,ax=plt.subplots(2,1,figsize=(16, 9),sharex=True)
    fig.suptitle("Entropia en A-A-F")
    ax[0].plot(g*t,S_vn_tot,color='black')
    # ax[0].set_xlabel('t')
    ax[0].set_ylabel('S_vn')

    ax[1].plot(g*t,S_lin_tot,color='red')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('S_lin')
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'entropia\{figname}',dpi=100)
    else: 
        None
    plt.close()
    #PLOT PARA LA DISTRIBUCION DE WIGNER. QUIZAS HACER UNA SIMULACION ESTARIA COPADO

    '''---Trazamos sobre el campo---'''
    #Y TOMANDO TRAZA PARCIAL SOBRE EL CAMPO, MIRAMOS EL ENTRELAZAMIENTO ENTRE ATOMOS

    atoms_states=np.empty_like(sol.states)
    for j in range(len(sol.states)):
        atoms_states[j]=sol.states[j].ptrace([0,1])
        
    S_vn_a=entropy_vn(atoms_states)
    S_lin_a=entropy_linear(atoms_states)
    concu_a=concurrence(atoms_states)
    #PLOT PARA LA DINAMICA (POBLACIONES Y COHERENCIAS) DEL SIST. TRAZANDO SOBRE LOS FOTONES

    #PLOT PARA LAS ENTROPIAS DEL SISTEMA TRAZANDO SOBRE LOS FOTONES
    fig,ax=plt.subplots(3,1,figsize=(16, 9),sharex=True)
    fig.suptitle("Sist. A-A sin foton")
    ax[0].plot(g*t,S_vn_a,color='black')
    # ax[0].set_xlabel('t')
    ax[0].set_ylabel('S_vn')

    ax[1].plot(g*t,S_lin_a,color='red')
    # ax[1].set_xlabel('t')
    ax[1].set_ylabel('S_lin')

    ax[2].plot(g*t,concu_a,color='blue')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('Concurrence')
    if plot_show==True:
        plt.show()
    else: 
        None
    if save_plot==True:
        plt.savefig(f'entropia_spin-spin\{figname}',dpi=100)
    else: 
        None
    plt.close()
yr, mes, dia, hr, minute = map(int, time.strftime("%Y %m %d %H %M").split())
mesydiayhora=str(mes)+'_'+str(dia)+'_'+str(hr)

script_path=os.path.dirname(__file__)
relative_path="graficos"+"\\"+mesydiayhora
path=os.path.join(script_path, relative_path)

if os.path.exists(path):
    os.chdir(path)
else: 
    os.makedirs(path)
    os.chdir(path)

J=0
t_final=100000
steps=25000
psi0=[eg0,(eg0-ge0)/np.sqrt(2),(eg1-ge1)/np.sqrt(2),(eg1+ge0)/np.sqrt(2),(eg1-ge0)/np.sqrt(2)]
psi0_folder=['eg0','eg0-','eg1-','eg1+ge0','eg1-ge0']
for psi0,psi0_folder in zip(psi0,psi0_folder):
    folders=['0','1','2','3','pauli','entropia','entropia_spin-spin']
    for folder in folders:
        folder_path=path+'\\'+psi0_folder+'\\'+folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    psi0_path=path+'\\'+psi0_folder
    os.chdir(psi0_path)
    g=[0.001*w_0]
    for g in g:
        p=0.0005*g
        k=0.1*g
        x=[0,1/4*g,0.5*g]
        for x in x:
            d=[0,0.1*g,2*g]
            for d in d:
                gamma=[0.1*g,2*g]
                for gamma in gamma:
                    main(w_0,g,k,J,d,x,gamma,p,psi0,t_final,steps)#,plot_show=True,save_plot=False)