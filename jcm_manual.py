import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import solve_ivp,cumulative_trapezoid
from jcm_lib import fases

# print(tensor(basis(2,0),basis(2,0)))

#EN ESTE CODIGO VEMOS LA COMPARACION ENTRE EL METODO INTEGRAL Y LA FASE DE PANCHARATNAM. 
#ES MEJOR EL DE PANCHANATRAM PERO CREO QUE LAS ECS DIFS SON MAS RAPIDAS DE SIMULAR, ENTONCES QUIERO UTILIZAR
#EL METODO DE PANCHANATRAM PERO USANDO ECS DIFS COMO SOLVER.

w_0=1
g=0.05*w_0
delta=0.001*g
x=0
omega=np.sqrt(4*g**2+(delta-x)**2)
T=2*np.pi/omega
t_final=10*T
steps=30000
t=np.linspace(0,t_final,steps)

gamma=0.1*g
p=0.01*g
gamma_z=0


#Definimos ecs difs del problema a lo calculo numerico

y0=np.array([0,1,0,0,0,0,0,0,0]) #rho00,rho11,im_rho12,re_rho12,rho22,rho33,im_rho34,re_rho34,rho44 con orden g0,e0,g1,e1,g2,.... 

dot_rho=np.array([[0,p,0,0,gamma,0,0,0,0],
                  [0,-p,-2*g,0,0,gamma,0,0,0],
                  [0,g,-(p+gamma+2*gamma_z)/2,-(delta-x),-g,0,np.sqrt(2)*gamma,0,0],
                  [0,0,(delta-x),-(p+gamma+2*gamma_z)/2,0,0,0,np.sqrt(2)*gamma,0],
                  [0,0,2*g,0,-gamma,p,0,0,2*gamma],
                  [0,0,0,0,0,-p-gamma,-2*np.sqrt(2)*g,0,0],
                  [0,0,0,0,0,-np.sqrt(2)*g,-p/2-3*gamma/2-2*gamma_z,-(delta-3*x),np.sqrt(2)*g],
                  [0,0,0,0,0,0,(delta-3*x),-p/2-3*gamma/2-2*gamma_z,0],
                  [0,0,0,0,0,0,2*np.sqrt(2)*g,0,-2*gamma]])

def f(t,y):
    return dot_rho@y

sol=solve_ivp(f,(t[0],t[-1]),y0,t_eval=t,rtol=1e-3,atol=1e-6) #Default values are 1e-3 for rtol and 1e-6 for atol

# print(sol.t)
# print(sol.y)
rho00=sol.y[0]
rho11=sol.y[1]
im_rho12=sol.y[2]
re_rho12=sol.y[3]
rho22=sol.y[4]

fig_pob=plt.figure(figsize=(8,6))
ax_pob=fig_pob.add_subplot()
ax_pob.plot(sol.t/T,rho00,color='black',label='g0')
ax_pob.plot(sol.t/T,rho11,color='red',label='e0')
ax_pob.plot(sol.t/T,np.sqrt(im_rho12**2+re_rho12**2),color='magenta',linestyle='dashed',label='e0-g1')
ax_pob.plot(sol.t/T,rho22,color='blue',label='g1')
ax_pob.legend()

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
    argumento = np.zeros(len(t))
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

def fases_anterior(delta, g, kappa, gamma, gamma_phi, t,rho_11,rho_12,rho_22):

    dotrho_12 = (-1j*(delta) - kappa/2 - gamma/2 - gamma_phi)*np.array(rho_12) - 1j*g*(np.array(rho_22) - np.array(rho_11))
    epsilon = 0.5*(np.array(rho_11) + np.array(rho_22) + np.sqrt((np.array(rho_11)-np.array(rho_22))**2 + 4*np.array(rho_12)*np.array(np.conj(rho_12))))

    # la función evaluada en el tiempo t = 0
    def funcion(x):
       x = int(x)
       return np.imag(np.conj(dotrho_12[x])*np.array(rho_12)[x])/((np.array(rho_22)[x]-epsilon[x])**2+np.array(rho_12)[x]*np.array(np.conj(rho_12))[x])

 # integral acumulada
    index = np.arange(0, len(t)+1)
    y = np.zeros(len(t))
    for i in range(len(t)):
      y[i] = funcion(i)

    phi_g = cumulative_trapezoid(y, t, initial=0)

    return phi_g

N_cav=2
psi0=tensor(basis(2,0),basis(2,0))
a_jcm=tensor(qeye(2),destroy(N_cav))
n_jcm=tensor(qeye(2),num(N_cav))
sm_jcm=tensor(sigmam(),qeye(N_cav))
sz_jcm=tensor(sigmaz(),qeye(N_cav))
H_jcm=delta/2*sz_jcm + x*n_jcm*n_jcm + g*(sm_jcm*a_jcm.dag()+sm_jcm.dag()*a_jcm)
sol_qutip=mesolve(H_jcm,psi0,t,c_ops=[np.sqrt(gamma)*a_jcm,np.sqrt(p)*sm_jcm,np.sqrt(gamma_z)*sz_jcm])
print(H_jcm*psi0)

fg_ant=fases_anterior(delta,g,gamma,0,0,t,rho11,re_rho12+1j*im_rho12,rho22)
fg_nue,argumento, eigenvalst,Psi=fases(sol_qutip)

fig_fg=plt.figure(figsize=(8,6))
ax_fg=fig_fg.add_subplot()
# ax_fg.plot(sol.t/T,nominador,color='red',linestyle='dashed',label='nom')
# ax_fg.plot(sol.t/T,denominador,color='blue',linestyle='dashed',label='denom')
ax_fg.plot(sol.t/T,fg_ant/np.pi,color='black',label='fg integral')
ax_fg.plot(sol.t/T,fg_nue/np.pi,color='red',label='fg panch')
# ax_fg.hlines(0,0,t[-1]/T)
ax_fg.legend()
plt.show()
