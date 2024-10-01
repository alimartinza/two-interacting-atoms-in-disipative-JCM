from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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

#Espacio N=0 [9]

# N=1 [3,6,10] N=2 [0,4,7,11] N=3 [1,5,8]

w_0=1

#HAGO UNA FUNCION DONDE PONGO TODO LO QUE HACE EL CODIGO. LA IDEA ERA HACERLO ASI PARA PODER HACER ITERACIONES SOBRE LOS PARAMETROS, PERO EN ESTE
#CASO NO ES NECESARIO... PERO QUEDO ASI

def evolucion(psi0:list,psi0_names:list,w_0:float,g:float,k:float,J:float,d:float,x:float,gamma:float,p:float,t_final:int,steps:int,disipation:bool=True,acoplamiento:str='lineal'):
    if len(psi0) != len(psi0_names):
        print(f"ERROR: La lista de condiciones iniciales y sus nombres tienen que tener la misma longitud pero tienen tamanios {len(psi0)} y {len(psi0_names)}")
        exit()
    #DEFINIMOS CUAL MODELO PARA EL ACOPLAMIENTO QUE VAMOS A USAR, Y LAS FUNCIONES QUE DEPENDEN DEL NUMERO DE OCUPACION DEL CAMPO FOTONICO

    def f():
        if acoplamiento=='lineal':
            return 1
        elif acoplamiento=='bs':
            return sqrtN
        else:
            print(f"ERROR: el acoplamiento tiene que ser lineal o bs pero es {acoplamiento}")
            exit()

    #Espacio N=0 [9]

    # N=1 [3,6,10] N=2 [0,4,7,11] N=3 [1,5,8]

    #DEFINIMOS LA FUNCION PR QUE DADO UN ESTADO NOS DA SU PROYECTOR 
    
    def pr(ket_):
        return ket_.unit()*ket_.unit().dag()

    '''---Hamiltoniano---'''

    H=x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*f()*a.dag()+(sp1+sp2)*a*f()) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2

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

    g_str=str(g).replace('.','_')
    k_str=str(k).replace('.','_')
    J_str=str(J).replace('.','_')
    d_str=str(d).replace('.','_')
    x_str=str(x).replace('.','_')
    gamma_str=str(gamma).replace('.','_')
    p_str=str(p).replace('.','_')
    parameters_name=f"g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}"
    csvname=f'g={g_str} k={k_str} J={J_str} d={d_str} x={x_str} gamma={gamma_str} p={p_str}.csv'


    '''---Simulacion numerica---'''

    t=np.linspace(0,t_final,steps) #TIEMPO DE LA SIMULACION 
    #DEFINIMOS LOS DISIPADORES SI HAY DISIPACION, Y SI NO, ENTONCES ESTA VACIO
    if disipation:
        l_ops=[np.sqrt(gamma)*a,np.sqrt(p)*(sp1+sp2)]
    elif not disipation:
        l_ops=[]
    
    #SIMULACIONES PARA LAS CONDICIONES INICIALES QUE COMPARAMOS. LA ULTIMA ES UN ESTADO QUE ES DENTRO DEL BLOQUE DE MISMA CANTIDAD DE EXITACIONES,
    #DE LOS MAS ENTRELAZADO POSIBLE
    cmap=mpl.colormaps["plasma"]
    colors_evals=cmap(np.linspace(0,1,12))
    colors=cmap(np.linspace(0,1,len(psi0)))

    fig_autoval=plt.figure()
    ax_eval=fig_autoval.add_subplot()

    fig_fg=plt.figure()
    if disipation == True:
        fig_fg.suptitle(f"Fase Geometrica no-uni {acoplamiento}")
    elif disipation == False:
        fig_fg.suptitle(f"Fase Geometrica uni {acoplamiento}")

    ax_fg=fig_fg.add_subplot()
    ax_fg.set_title(parameters_name)
    for i,psi in enumerate(psi0):

        sol=mesolve(H,psi,t,c_ops=l_ops,progress_bar=True) #SOLVER QUE HACE LA RESOLUCION NUMERICA PARA LINBLAD
        fg_pan,arg,eigenvals_t = fases(sol)
        
        for j,evals in enumerate(eigenvals_t.transpose()):
            ax_eval.scatter(g*t,evals,color=colors_evals[j],label=f"$\lambda_{i}$")
        ax_eval.set_xlabel(r"$gt$")
        ax_eval.set_ylabel("Autovalores")


        ax_fg.plot(g*t,fg_pan,color=colors[i],label=rf"$\psi_0=|{psi0_names[i]}>$")
    
    ax_fg.legend()
    ax_fg.grid()
    plt.show()

    #ESTO ULTIMO NO HACE NADA, ESTA HECHO PARA GUARDAR LOS DATOS O LOS GRAFICOS DIRECTO EN LA COMPU. PERO AHORA ESTA COMENTADO ASI QUE NO HACE NADA

    #GUARDAMOS EL DATAFRAME EN CSV. 
    
    '''--------------SAVE TO CSV OR TO QU FILE-------------------'''
    # data.to_csv(csvname)
    #EN ESTA VERSION NO GUARDAMOS LOS ESTADOS DIAGONALIZADOS PORQUE OCUPAN ESPACIO
    #  Y TIEMPO Y QUIZAS COMBIENE HACERLO SOLO ESPECIALMENTE PARA LA SIMULACION QUE QUEREMOS ANALIZAR
    # fileio.qsave(eigenvecs,parameters_name+'eigen states')


#ACA ES DONDE CORRE EL CODIGO DIGAMOS. LOS PARAMETROS HAY QUE CAMBIARLOS DESDE ACA.

disipation=False #PUEDE SER False o True. False para unitario, True para decoherencia
acoplamiento='bs' #por ahora no vi que haya cambios significativos, pero puede ser "lineal" o "bs". Solo es un acoplamiento que depende del numero de fotones de la cavidad, pero no hace mucho la verdad
g=0.001*w_0 #acoplamiento atomo-cavidad
p=0.005*g #pumping rate para disipador sigma+ (no hace falta poner en 0 si pones sin disipacion, solo se activa si disipation=True)
k=0.1*g #interaccion sigma+sigma- entre atomos
x=0.5*g #medio kerr
d=0# 0.5*g #detuning
gamma=0.1*g #rate de perdida de fotones (igual que el p)
J=0 #interaccion tipo ising entre atomos 
t_final=25000
steps=2000
psi0=[eg0,(eg0+ge0).unit(),(eg0-ge0).unit(),(eg1-ge1).unit()]
psi0_names=['eg0','eg0+ge0','eg0-ge0','eg1-ge1']
evolucion(psi0,psi0_names,w_0,g,k,J,d,x,gamma,p,t_final,steps,disipation=disipation,acoplamiento=acoplamiento)



#Lo de abajo no importa, es lo de las iteraciones que decia al principio para hacer un barrido de parametros y que guarde automaticamente los graficos


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

