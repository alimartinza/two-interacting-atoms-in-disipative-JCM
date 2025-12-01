import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import integrate

#CODIGO DE PAU QUE CREO QUE USARON PARA EL PAPER DE TRANSMON, DONDE USAN METODO INTEGRAL Y METODO DE PANCHA.
# LO USE PARA COMPARAR CON LO QUE TENGO YO Y POR EJEMPLO EN RESONANCIA TAMBIEN APARECEN LOS REBOTES. 
# ME PARECE QUE ES MEJOR EL METODO DE PANCHANATRAM

###################### DEFINICION DE FUNCIONES################################

# Carga el estado inicial en funcion de la dimension del espacio del atomo N y del campo 
# N2, y las frecuencias del Hamiltoniano. 
def inicio(N, N2, wr, wq, g, EC, inicial = 0):
  # estado inicial
  if inicial == 0:
    psi0 = tensor(basis(N,0), basis(N2,1))    # qubit excitado |0 e>= |0 1>
  elif inicial == 1:
    psi0 = tensor(basis(N,0), basis(N2,2))    # qubit excitado  |0 2>
  elif inicial == 2:
    psi0 = tensor(basis(N,1), basis(N2,1))    # qubit excitado + 1 foton |11>
  # operadores
  a  = tensor(destroy(N), qeye(N2))    # a estados del campo
  b  = tensor(qeye(N), destroy(N2))    # b estados del atomo
  I = tensor(qeye(N), qeye(N2))        # I identidad
  # Hamiltoniano del sistema, si Ec=0, vamos a 2x2
  H = wr * a.dag() * a + wq*(b.dag() * b + 0.5*I) - EC/2*b.dag()*b.dag() * b * b + g * (b.dag() * a + b * a.dag())
  return psi0, a, b, H

# operadores de la ecuacion maestra
def op_colapso(kappa, a, b, gamma, gamma_phi):
  c_ops = []
  # cavity relaxation
  c_ops.append(np.sqrt(kappa) * a)    # afectan al foton o es (kappa)?
  c_ops.append(np.sqrt(gamma) * b)    # afectan al atomo o es (gamma)?
  # pure dephasing
  c_ops.append(np.sqrt(gamma_phi*2) * b.dag()*b) #dephasing
  return c_ops

# resuelve la ecuacion maestra a partir del Hamiltoniano y el estado inicial
def master_eq(H, psi0, t, c_ops1,c_ops2):
  return mesolve(H, psi0, t, c_ops1), mesolve(H, psi0, t,c_ops2)

#define la matriz densidad
def rho_ij(result):
  rho_00, rho_11, rho_22, rho_12, rho_01, rho_33, rho_23, rho_23, rho_21 = [], [], [], [], [], [], [], [], []
  for i in range(len(result.states)):
    rho_00.append(result.states[i][0,0]) # elemento |00><00|
    rho_11.append(result.states[i][1,1]) # elemento |00><00|
    rho_22.append(result.states[i][2,2]) # elemento |10><10|
    rho_23.append(result.states[i][2,3]) # elemento |10><02|
    rho_33.append(result.states[i][3,3]) # elemento |02><02|
    rho_12.append(result.states[i][1,2]) # elemento |01><10|
    rho_21.append(result.states[i][2,1]) # elemento |10><01|
    rho_01.append(result.states[i][0,1]) # elemento |00><01|
  return rho_00, rho_11, rho_22, rho_12, rho_01, rho_33, rho_23, rho_21

def rho_ij2(result):
  rho_00, rho_11, rho_33, rho_01, rho_03, rho_13= [], [], [], [], [], []
  for i in range(len(result.states)):     #|ab>
    rho_00.append(result.states[i][0,0])  # elemento |00><00|
    rho_11.append(result.states[i][1,1])  # elemento |01><01|
    rho_33.append(result.states[i][3,3])  # elemento |02><02|
    rho_01.append(result.states[i][0,1])  # elemento |00><01|
    rho_03.append(result.states[i][0,3])  # elemento |00><02|
    rho_13.append(result.states[i][1,3])  # elemento |01><02|
  return rho_00, rho_11, rho_33, rho_01, rho_03, rho_13

def rho_ij3(result):
  rho_22, rho_44, rho_66, rho_24, rho_26, rho_46 = [], [], [], [], [], []
  for i in range(len(result.states)):
    rho_22.append(result.states[i][2,2])  # elemento |10><10|
    rho_44.append(result.states[i][4,4])  # elemento |11><11|
    rho_66.append(result.states[i][6,6])  # elemento |20><20|
    rho_24.append(result.states[i][2,4])  # elemento |10><11|
    rho_26.append(result.states[i][2,6])  # elemento |10><20|
    rho_46.append(result.states[i][4,6])  # elemento |11><20|
  return rho_22, rho_44, rho_66, rho_24, rho_26, rho_46

def fases(result, times):
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
        eigenvec_list = list(eigenvec)
        psi_prueba, overlap_max = max(((autoestado, abs(autoestado.overlap(psi_old))) for autoestado in eigenvec), key=lambda x: x[1])

        index = np.array([0, 1, 2, 3])
        autoval.append(eigenval[eigenvec_list.index(psi_prueba)])

        index = np.delete(index, int(eigenvec_list.index(psi_prueba)))

        autoval_2.append(eigenval[index[0]])       ## solo para probar
        autoval_3.append(eigenval[index[1]])       ## despues borrar
        autoval_4.append(eigenval[index[2]])

        norma.append(psi.overlap(psi0))

        pan += np.angle(psi_old.overlap(psi))
        Pan.append(pan - np.angle(psi.overlap(psi0)))
        psi_old = psi

        # Almaceno el argumento para cada tiempo
        argumento[i] = np.angle(psi0.dag() * psi)


    Pan = np.array(Pan)
    return np.unwrap(Pan), argumento, autoval, autoval_2, autoval_3, autoval_4

def fases_anterior(wr, wq, g, kappa, gamma, gamma_phi, t, rdo):
    rho_00, rho_11, rho_22, rho_12, rho_01, rho_33, rho_23, rho_21 = rho_ij(rdo)

    dotrho_12 = (1j*(wr-wq) - kappa/2 - gamma/2 - gamma_phi)*np.array(rho_12) - 1j*g*(np.array(rho_22) - np.array(rho_11))
    epsilon = 0.5*(np.array(rho_11) + np.array(rho_22) + np.sqrt((np.array(rho_11)-np.array(rho_22))**2 + 4*np.array(rho_12)*np.array(rho_21)))

    # la función evaluada en el tiempo t = 0
    def funcion(x):
       x = int(x)
       return np.imag(np.conj(dotrho_12[x])*np.array(rho_12)[x])/((np.array(rho_22)[x]-epsilon[x])**2+np.array(rho_12)[x]*np.array(rho_21)[x])

 # integral acumulada
    index = np.arange(0, len(t)+1)
    y = np.zeros(len(t))
    for i in range(len(t)):
      y[i] = funcion(i)

    phi_g = integrate.cumulative_trapezoid(y, t, initial=0)

    return phi_g


# Parametros
# g  = 0.05 * 2 * np.pi             # coupling strength
# wr = 1.0  * 2 * np.pi             # frecuencia de la cavidad
# wq = 0.0*g + wr                     # frecuencia del qubit
# gamma_phi = 0               # pure dephasing rate
# gamma = 0                  # atom dissipation rate
# # p = gamma
# kappa = 0.1*g             # cavity dissipation rate 
# N = 2                            # estados de fock en la cavidad
# N2 = 2
# EJ_EC = np.linspace(20, 80,2)
# # EC = np.array([0.000001])
# # EC = np.append(EC, wq*((np.sqrt(8*EJ_EC)-1)**(-1)))

# Omega_0 = np.sqrt((wq-wr)**2+4*g**2)
# tau = 2*np.pi/Omega_0
# t = np.arange(0,5*tau,0.001)

# #{|mode, atom >} = {00, 10, 01, 20, 11, 02}
# #psi0 = tensor(basis(N,0), basis(N2,0))  #(1,0,0,0,0,0, ..)
# #psi1 = tensor(basis(N,0), basis(N2,1))  #(0,0,0,1,0,0, ..)
# #psi2 = tensor(basis(N,1), basis(N2,0))  #(0,1,0,0,0,0, ..)
# #psi3 = tensor(basis(N,0), basis(N2,2))  #(0,0,0,0,0,0,1,...)
# #psi4 = tensor(basis(N,1), basis(N2,1))  #(0,0,0,0,1,0, ..)
# #psi5 = tensor(basis(N,2), basis(N2,0))  #(0,0,1,0,0,0,...)
# #OBS: qutip ordena la base según (N2,N), no es el mismo orden que en las cuentas
# #mi_base = [psi0, psi1, psi2, psi3, psi4, psi5]
# #mi_base_name = ['00', '01', '10', '02', '11', '20']

# #############DINAMICA#####################3
# # for EC_i in EC:
# fig, axes = plt.subplots(1, figsize=(10,5))
# # fig.suptitle(r'$E_C/\omega_q = {}$'.format(round(0/wq,2)))
# psi0, a, b, H = inicio(N, N2, wr, wq, g, 0, inicial = 0)
# c_ops = op_colapso(kappa, a, b, gamma, gamma_phi)
# c_ops_unit = op_colapso(0, a, b, 0, 0)
# disip, unit2 = master_eq(H, psi0, t, c_ops, c_ops_unit)

# rho_00_u, rho_11_u, rho_22_u, rho_12_u, rho_01_u, rho_33_u, rho_23_u, rho_21_u = rho_ij(unit2)
# rho_00_d, rho_11_d, rho_22_d, rho_12_d, rho_01_d, rho_33_d, rho_23_d, rho_21_d = rho_ij(disip)

# axes.plot(t/tau, rho_00_u, '--', color = 'dimgray', label=r"$\rho_{00}$")
# axes.plot(t/tau, rho_11_u, '-', color = 'sandybrown', label=r"$\rho_{11}$")
# axes.plot(t/tau, rho_22_u, '-.', color = 'darkcyan', label=r"$\rho_{22}$")
# axes.plot(t/tau, np.abs(rho_12_u), ':', color = 'mediumvioletred', label=r"$|\rho_{12}|$")


# axes.legend(loc=0)
# axes.set_xlabel(r'$t/\tau$')
# axes.set_ylabel(r'$\rho_{ij}$')
# axes.grid(alpha = 0.5)
    
    
  #############fase#####################3  
# Parametros
N=2
N2=2
g  = 0.1 * 2 * np.pi             # coupling strength
wr = 1.0  * 2 * np.pi             # frecuencia de la cavidad
# wq = np.linspace(0.1, 2, 4)*g + wr                     # frecuencia del qubit
wq = np.array([0.*g + wr])

Omega_0 = np.sqrt((wq-wr)**2+4*g**2)
tau = 2*np.pi/Omega_0
#t = np.arange(0,5*tau,0.01)
kappa=0.1*g
gamma=0
gamma_phi=0

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, axes = plt.subplots(1, 1, figsize=(10,5))
plt.title(r'No Unitaria: $\kappa/g$ = {}'.format(kappa/g))
color = ['gray', 'cyan', 'violet', 'blue']
for i in range(len(wq)):
    wq_i = wq[i]
    Omega0 = Omega_0[i]
    tau = 2*np.pi/Omega0
    t = np.arange(0,5*tau,0.01)
    psi0, a, b, H = inicio(N, N2, wr, wq_i, g, 0.5, inicial = 0)
    c_ops = op_colapso(kappa, a, b, gamma, gamma_phi)
    c_ops_unit = op_colapso(0, a, b, 0, 0)
    disip, unit2 = master_eq(H, psi0, t, c_ops, c_ops_unit )

    rdo_nuevo, argumento, autoval, autoval_2, autoval_3, autoval_4 = fases(unit2, t*tau)
    rdo_nuevo_d, argumento, autoval, autoval_2, autoval_3, autoval_4 = fases(disip, t*tau)
    rdo_anterior = fases_anterior(wr, wq_i, g, 0, 0, 0, t, unit2)
    rdo_anterior_d = fases_anterior(wr, wq_i, g, kappa, gamma, gamma_phi, t, disip)
    axes.plot(t/tau, rdo_anterior/np.pi, '-', label = 'cálculo anterior u', color = color[0])
    axes.plot(t/tau, rdo_anterior_d/np.pi, '-', label = 'cálculo anterior d', color = color[1])
    axes.plot(t/tau, rdo_anterior_d/np.pi-rdo_anterior/np.pi, '--', label = 'd-u', color = color[1])

    axes.plot(t/tau, rdo_nuevo/np.pi, '-', label = 'cálculo nuevo u', color = color[2])
    axes.plot(t/tau, rdo_nuevo_d/np.pi, '-', label = 'cálculo nuevo d', color = color[3])
    axes.plot(t/tau, rdo_nuevo_d/np.pi-rdo_nuevo/np.pi, '--', label = 'd-u', color = color[3])
    axes.legend(loc=0)
    axes.set_xlabel(r'$t/\tau$')
    axes.set_ylabel(r'$\phi_g/\pi$')
    # Crear el gráfico de zoom
    axins = inset_axes(axes, width="30%", height="30%", loc="lower right", bbox_to_anchor=(-0.051, 0.1, 1, 1), bbox_transform=axes.transAxes)
    axins.plot(t / tau, rdo_anterior / np.pi, '-', label='cálculo anterior', color=color[0])
    axins.plot(t / tau, rdo_anterior_d / np.pi, '-', label='cálculo anterior', color=color[1])
    axins.plot(t / tau, rdo_nuevo / np.pi, '-', label='cálculo nuevo', color=color[2])
    axins.plot(t / tau, rdo_nuevo_d / np.pi, '-', label='cálculo nuevo', color=color[3])
    axins.set_xlim(0, 1)
    axins.set_ylim(-1,0.04)
    axins.set_xticks([0, 0.5, 1])  # Establecer las ubicaciones de las marcas en el eje x del zoom
    axins.set_yticks([-0.5, -1])  # Establecer las ubicaciones de las marcas en el eje y del zoom
#    axins.set_xticklabels('')
#    axins.set_yticklabels('')
plt.show()