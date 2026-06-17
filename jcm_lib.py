from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import matplotlib as mpl


# from mpl_toolkits.mplot3d import axes3d

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

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE,figsize=(16,9))  # fontsize of the figure title
# plt.rc('figure')
plt.rc('figure.subplot',left=0.064, right=0.95, top=0.94, bottom=0.064,hspace=0.02)

script_path = os.path.dirname(__file__)  #DEFINIMOS EL PATH AL FILE GENERICAMENTE PARA QUE FUNCIONE DESDE CUALQUIER COMPU

# folder_names=["8_30_22 disipativo lineal","8_31_3 disipativo bs","8_31_8 unitario lineal","8_31_14 unitario bs"] #PONEMOS LOS NOMBRES DE LAS CARPETAS QUE QUEREMOS VISITAR
# condiciones_iniciales=["ee0"]#,"gg1","eg0"] #CONDICIONES INICIALES QUE QUEREMOS GRAFICAR

#DEFINIMOS LOS PARAMETROS QUE NO VAMOS A QUERER MODIFICAR EN LOS GRAFICOS
# w0=1
# J=0
# g=0.001*w0
# k=0.1*g
# p=0.005*g
# t_final=25000
# steps=2000
# t=np.linspace(0,t_final,steps)
save_plot=False
plot_show=True

def triconcurrence(sol,alpha:float):
    'Implementacion numerica de medida de entrelazamiento tripartita genuina planteada por unos chinos basada en una medida genuinda de entrelazamiento bipartita'
    'A(|psi_ijk>)=sqrt(Q(Q-E^a_{i|jk})(Q-E^a_{j|ik})(Q-E^a_{k|ij})) donde E_{i|jk} es el entrelazamiento entre la particion i;jk, a es una potencia y a\in(0,1] y Q=(E^a_{i|jk}+E^a_{i|jk}+E^a_{i|jk})/2'
    if alpha<=0 or alpha>1:
        print('alpha tiene que ser entre (0,1]')
        exit()

    for i in range(len(sol.states)):
        if sol.states[i].isket: 
            pass
        else:
            print('No todos los estados de la evolucion son puros, corroborar porque.')
            break
    states12=np.empty_like(sol.states)
    states02=np.empty_like(sol.states)
    states01=np.empty_like(sol.states)
    for j in range(len(sol.states)):
        states12[j]=sol.states[j].ptrace([1,2])
        states02[j]=sol.states[j].ptrace([0,2])
        states01[j]=sol.states[j].ptrace([0,1])
    E1=concurrence(states12)      #E_i|jk
    E2=concurrence(states02)      #E_j|ik
    E3=concurrence(states01)      #E_k|ij
    Q=(E1^alpha+E2^alpha+E3^alpha)/2
    A=np.sqrt(Q*(Q-E1)*(Q-E2)*(Q-E3))
    return A

def fases(sol,open_system:bool):
    """params:
    -sol: solucion numerica de la evolucion temporal. Puede ser un Solver o un ndarray con las soluciones.
    -N_c 
    RETURNS
    if open_system is True
    -fg_pan: Array de longitud len(t) donde con la FG de Pancho acumulada tiempo a tiempo
    -arg: no deberia funcionar bien esto me parece, pero seria el primer termino de la fg
    -ordered_eigenvals
    -ordered_eigenvecs
    if open_system is False
    -fg_pan
    -arg"""
    # try: 
    if open_system:
        len_t=len(sol.states)
        if sol.states[0].type == 'ket' or sol.states[0].type == 'bra':
            rho0 = ket2dm(sol.states[0])
        else:
            rho0 = sol.states[0]
            
        eval0,evec0=rho0.eigenstates(sort='high')


        eigenvals_t = [[eval] for eval in eval0]
        eigenvecs_t = [[evec] for evec in evec0]

        norma = []
        
        signo = 0
        for t_i in range(1,len_t):
            if sol.states[t_i].type == 'ket' or sol.states[t_i].type == 'bra':
                rho = ket2dm(sol.states[t_i])
            else:
                rho = sol.states[t_i]
            
            eigenvals_rho,eigenvecs_rho = rho.eigenstates(sort='high')
            index_check_array=-1*np.ones(len(eval0))
            # print('i=',i)
            # print('len eigenvecsrho[i]=',len(eigenvecs_rho))
            for i_1 in range(len(eigenvecs_rho)):
                    
                psi, overlap,index = max(((autoestado, abs(autoestado.overlap(eigenvecs_t[i_1][t_i-1])),autoestado_index) for autoestado_index,autoestado in enumerate(eigenvecs_rho)),key=lambda x: x[1])
                
                psi_prob=expect(rho,psi)
                eigenvecs_t[i_1].append(psi)
                eigenvals_t[i_1].append(psi_prob)
                if index in index_check_array:
                    # Conflicto: dos autovectores se emparejan con el mismo del paso
                    # anterior. NO cortamos el loop (eso dejaba listas de longitud
                    # inconsistente -> IndexError). Seguimos: el append de psi ya se
                    # hizo arriba, y el autovector dominante (eigenvecs_t[0], el unico
                    # usado para la fase) nunca esta en conflicto porque su gap es grande.
                    pass
                else:
                    index_check_array[i_1]=index  
                #     else: raise Warning(f'tenemos un conflicto, dos vectores en el paso {i} parten del mismo vector en el paso anterior.')
                # else:
                #     index_check_array[i_1]=index
            
                # except:
                #     print('i=',i)
                #     print('i_1=',i_1)
                #     print('len eigenvecs_t=',len(eigenvecs_t))
                #     print('len eigenvecs_t[i_1]=',len(eigenvecs_t[i_1]))
                #     print(f'eigenvals_rho=',eigenvals_rho)
                #     print(f'eigenvecs_rho=',eigenvecs_rho)
                #     print(f'eigenvecs_t[i_1][i-1]=',eigenvecs_t[i_1][i-1])

        pan = 0
        Pan = []
        argumento = np.zeros(len_t)
        psi0=evec0[0]
        psi_old=psi0
        # print('type rho0',rho0.type)
        for t_j in range(len(eigenvecs_t[0])):
            # print('i',i)
            psi=eigenvecs_t[0][t_j]
            pan += np.angle(psi.overlap(psi_old))
            Pan.append(pan - np.angle(psi.overlap(psi0)))
            psi_old = eigenvecs_t[0][t_j]
            # Almaceno el argumento para cada tiempo
            argumento[t_j] = np.angle(psi0.dag().overlap(psi))

        Pan = np.array(Pan)
        # print(len(eigenvals_t))
        # print(len(eigenvecs_t))
        # print(len(eigenvecs_t[0]))
        return np.unwrap(Pan), argumento, eigenvals_t, eigenvecs_t
    
    else:
        len_t=len(sol.states)
        if sol.states[0].type != 'ket' and sol.states[0].type != 'bra':
            print()
            raise Exception(f'tu condicion inicial es de tipo {sol.states[0].type} pero deberia ser \'ket\' o \'bra\'.')
        
        psi0=sol.states[0]
        psi_old = psi0
        norma = []
        pan = 0
        Pan = []
        argumento = np.zeros(len_t)
        signo = 0
        for i in range(len_t):
            if sol.states[i].type != 'ket' and sol.states[i].type != 'bra':
                raise Exception('dijiste que el sistema es cerrado pero tu evolucion es mixta.')
            psi=sol.states[i]
            pan += np.angle(psi.overlap(psi_old))
            Pan.append(pan - np.angle(psi.overlap(psi0)))
            psi_old = sol.states[i]
            # Almaceno el argumento para cada tiempo
            argumento[i] = np.angle(psi0.dag() * psi)
    
    Pan = np.array(Pan)
    
    return np.unwrap(Pan), argumento

def fases_viejo(sol):
    """params:
    -sol: solucion numerica de la evolucion temporal. Puede ser un Solver o un ndarray con las soluciones.
    RETURNS
    -fg_pan: Array de longitud len(t) donde con la FG de Pancho acumulada tiempo a tiempo
    -arg: no se
    -eigenvals: array de len(t)x12, entonces el elemento eigenvals[k] me da los 12 autovalores a tiempo t_k."""
    try: 

        len_t=len(sol.states)
        if sol.states[0].type == 'ket' or sol.states[0].type == 'bra':
            rho0 = ket2dm(sol.states[0])
        else:
            rho0 = sol.states[0]
        eval0,evec0=rho0.eigenstates(sort='high')
        eigenvals_t = np.array([eval0])
        max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
        psi0 = evec0[max_eigenvalue_idx]
        psi_old = psi0
        Psi = [psi0]
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
            
            eigenval,eigenvec = rho.eigenstates(sort='high')
            eigenvals_t=np.concatenate((eigenvals_t,[eigenval]),axis=0)

            psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])

            # norma.append(psi.overlap(psi0))

            pan += np.angle(psi.overlap(psi_old))
            Pan.append(pan - np.angle(psi.overlap(psi0)))
            psi_old = psi
            Psi.append(psi)
            # Almaceno el argumento para cada tiempo
            argumento[i] = np.angle(psi0.dag() * psi)

    except:

        if type(sol) is np.ndarray: #ya sabemos que no es un solver, asi que probamos a ver si es un ndarray. Si no, sigue al else y tira un error.
            len_t=len(sol)
            if sol[0].type == 'ket' or sol[0].type == 'bra':
                rho0 = ket2dm(sol[0])
            else:
                rho0 = sol[0]
            eval0,evec0=rho0.eigenstates()
            eigenvals_t = np.array([eval0])
            max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
            psi0 = evec0[max_eigenvalue_idx]
            psi_old = psi0
            Psi = [psi0]
            norma = []
            pan = 0
            Pan = []
            argumento = np.zeros(len_t)
            signo = 0
            for i in range(len_t):
                if sol[i].type == 'ket' or sol[i].type == 'bra':
                    rho = ket2dm(sol[i])
                else:
                    rho = sol[i]
                
                eigenval,eigenvec = rho.eigenstates()
                eigenvals_t=np.concatenate((eigenvals_t,[eigenval]),axis=0)

                psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])

                # norma.append(psi.overlap(psi0))

                pan += np.angle(psi.overlap(psi_old))
                Pan.append(pan - np.angle(psi.overlap(psi0)))
                psi_old = psi
                Psi.append(psi)
                # Almaceno el argumento para cada tiempo
                argumento[i] = np.angle(psi0.dag() * psi)

    
        else: 
            raise ValueError('jcm_lib.fases() no toma como argumento estados de este tipo. Solo puede ser un Solver de qutip.mesolve() o un ndarray.')
        
    eigenvals_t=np.delete(eigenvals_t,0,axis=0)
    Pan = np.array(Pan)

    return np.unwrap(Pan), argumento, np.array(eigenvals_t),Psi

def fases_nuevo(result, times):
    # Autoestado calculado diagonalizando la matriz   --->   autoestado_2
    rho0 = result.states[0]
    eigenval, eigenvec = rho0.eigenstates()    # Diagonalizar la matriz
    max_eigenvalue_idx = eigenval.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
    psi0 = eigenvec[max_eigenvalue_idx]
    psi_old = eigenvec[max_eigenvalue_idx]
    Psi = []
    norma = []
    pan = 0
    Pan = []
    argumento = np.zeros(len(times))
    autoval, autoval_2, autoval_3, autoval_4 = [], [], [], []
    signo = 0
    for i in range(len(times)):
        # Autoestado numérico
        rho = result.states[i]
        eigenval, eigenvec = rho.eigenstates()    # diagonalizo la matriz

        psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])
        # eigenvec_list = list(eigenvec)
        # psi_prueba, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])

        # index = np.array([0, 1, 2, 3])
        # autoval.append(eigenval[eigenvec_list.index(psi_prueba)])

        # index = np.delete(index, int(eigenvec_list.index(psi_prueba)))

        # autoval_2.append(eigenval[index[0]])       ## solo para probar
        # autoval_3.append(eigenval[index[1]])       ## despues borrar
        # autoval_4.append(eigenval[index[2]])

        # norma.append(psi.overlap(psi0))

        pan += np.angle(psi.overlap(psi_old))
        Pan.append(pan - np.angle(psi.overlap(psi0)))
        psi_old = psi

        # Almaceno el argumento para cada tiempo
        argumento[i] = np.angle(psi0.dag() * psi)


    Pan = np.array(Pan)
    return Pan #, argumento, autoval, autoval_2, autoval_3, autoval_4


def fases_manual(result, times):
    # Autoestado calculado diagonalizando la matriz   --->   autoestado_2
    rho0 = result.states[0]
    eigenval, eigenvec = rho0.eigenstates()    # Diagonalizar la matriz
    max_eigenvalue_idx = eigenval.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
    psi0 = eigenvec[max_eigenvalue_idx]
    psi_old = eigenvec[max_eigenvalue_idx]
    Psi = []
    norma = []
    pan = 0
    Pan = []
    argumento = np.zeros(len(times))
    autoval, autoval_2, autoval_3, autoval_4 = [], [], [], []
    signo = 0
    for i in range(len(times)):
        # Autoestado numérico
        rho = result.states[i]
        eigenval, eigenvec = rho.eigenstates()    # diagonalizo la matriz

        psi, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])
        eigenvec_list = list(eigenvec)
        psi_prueba, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])

        index = np.array([0, 1, 2, 3])
        autoval.append(eigenval[eigenvec_list.index(psi_prueba)])

        index = np.delete(index, int(eigenvec_list.index(psi_prueba)))

        autoval_2.append(eigenval[index[0]])       ## solo para probar
        autoval_3.append(eigenval[index[1]])       ## despues borrar
        autoval_4.append(eigenval[index[2]])

        norma.append(psi.overlap(psi0))

        pan += np.angle(psi.overlap(psi_old))
        Pan.append(pan - np.angle(psi.overlap(psi0)))
        psi_old = psi

        # Almaceno el argumento para cada tiempo
        argumento[i] = np.angle(psi0.dag() * psi)


    Pan = np.array(Pan)
    return np.unwrap(Pan), argumento, autoval, autoval_2, autoval_3, autoval_4


def fases_mixta(sol):
    """params:
    -sol: solucion numerica de la evolucion temporal. Puede ser un Solver o un ndarray con las soluciones.
    RETURNS
    -fg_pan: Array de longitud len(t) donde con la FG de Pancho acumulada tiempo a tiempo
    -arg: no se
    -eigenvals: array de len(t)x12, entonces el elemento eigenvals[k] me da los 12 autovalores a tiempo t_k."""
    try:

        len_t=len(sol.states)
        if sol.states[0].type == 'ket' or sol.states[0].type == 'bra':
            rho0 = ket2dm(sol.states[0])
        else:
            rho0 = sol.states[0]
        eval0,evec0=rho0.eigenstates(sort='high')
        eigenvals_t = np.array([eval0])
        # max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
        psi0 = evec0
        len0=len(psi0)
        psi_old = psi0
        eval_old=eval0
        # Psi = [psi0]
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
            
            eigenval,eigenvec = rho.eigenstates(sort='high')
            eigenvals_t=np.concatenate((eigenvals_t,[eigenval]),axis=0)

            psi= [max(((autoestado, autovalor,abs(autoestado.overlap(evec_old))) for autoestado,autovalor in zip(eigenvec,eigenval)), key=lambda x: x[2]) for evec_old in psi_old]
            psi=[psi[i][0] for i in range(len(psi_old))]
            eigenval=[psi[i][1] for i in range(len(psi_old))]
            # norma.append(psi.overlap(psi0))

            pan += np.angle(np.sum(np.sqrt(eigenval[i]*eval_old[i])*psi[i].overlap(psi_old[i]) for i in range(len0)))
            Pan.append(pan - np.angle(np.sum(np.sqrt(eval0[i]*eigenval[i])*psi[i].overlap(psi0[i]) for i in range(len0))))
            psi_old = psi
            eval_old=eigenval
            # Psi.append(psi)
            # Almaceno el argumento para cada tiempo
            argumento[i] = np.angle(np.sum(psi0[i].dag() * psi[i] for i in range(len0)))
    except:
        if type(sol) is np.ndarray: #ya sabemos que no es un solver, asi que probamos a ver si es un ndarray. Si no, sigue al else y tira un error.

            len_t=len(sol)
            if sol[0].type == 'ket' or sol[0].type == 'bra':
                rho0 = ket2dm(sol[0])
            else:
                rho0 = sol[0]
            eval0,evec0=rho0.eigenstates(sort='high')
            eigenvals_t = np.array([eval0])
            # max_eigenvalue_idx = eval0.argmax()    # encuentro el autovector correspondiente al autovalor más grande en el tiempo 0
            psi0 = evec0
            len0=len(psi0)
            psi_old = psi0
            eval_old=eval0
            # Psi = [psi0]
            norma = []
            pan = 0
            Pan = []
            argumento = np.zeros(len_t)
            signo = 0
            for i in range(len_t):
                if sol[i].type == 'ket' or sol[i].type == 'bra':
                    rho = ket2dm(sol[i])
                else:
                    rho = sol[i]
                
                eigenval,eigenvec = rho.eigenstates(sort='high')
                eigenvals_t=np.concatenate((eigenvals_t,[eigenval]),axis=0)

                psi= [max(((autoestado, autovalor,abs(autoestado.overlap(evec_old))) for autoestado,autovalor in zip(eigenvec,eigenval)), key=lambda x: x[2]) for evec_old in psi_old]
                psi=[psi[i][0] for i in range(len(psi_old))]
                eigenval=[psi[i][1] for i in range(len(psi_old))]
                # norma.append(psi.overlap(psi0))

                pan += np.angle(np.sum(np.sqrt(eigenval[i]*eval_old[i])*psi[i].overlap(psi_old[i]) for i in range(len0)))
                Pan.append(pan - np.angle(np.sum(np.sqrt(eval0[i]*eigenval[i])*psi[i].overlap(psi0[i]) for i in range(len0))))
                psi_old = psi
                eval_old=eigenval
                # Psi.append(psi)
                # Almaceno el argumento para cada tiempo
                argumento[i] = np.angle(np.sum(psi0[i].dag() * psi[i] for i in range(len0)))    

    eigenvals_t=np.delete(eigenvals_t,0,axis=0)
    Pan = np.array(Pan)

    return np.unwrap(Pan), argumento, np.array(eigenvals_t)