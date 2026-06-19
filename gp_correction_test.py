#!/usr/bin/env python3
# =============================================================================
#  test_correccion_saltos.py
#
#  CASO DE CONTROL para probar si la heuristica de correccion de saltos
#  (jcm_lib.fases(corregir_saltos=True)) funciona como se espera.
#
#  Por que este caso: con Delta=0 (resonancia exacta) y psi0 = (eg1+ge1)/sqrt2
#  (el autoestado intermedio del bloque 3x3, el "dark state" en resonancia),
#  el usuario ya sabe que aparece inestabilidad numerica por redondeo en
#  fases() (saltos de phi_u(t) con signo invertido). Es el caso ideal para
#  verificar la heuristica: sabemos DE ANTEMANO que deberia haber algo que
#  corregir en la rama unitaria.
#
#  Que hace este script:
#    1) Corre la evolucion unitaria Y disipativa para este psi0 fijo.
#    2) Llama a fases() CON y SIN corregir_saltos, en ambas ramas, y guarda
#       las 4 curvas (fu_sin, fu_con, fd_sin, fd_con) para comparar.
#    3) Imprime el log COMPLETO de saltos detectados en cada rama (no solo
#       el conteo), para poder inspeccionar a mano que esta pasando.
#    4) Grafica todo, con paneles separados para la rama unitaria y la
#       disipativa, así se puede ver si la correccion hace lo esperado en
#       cada una por separado.
#
#  IMPORTANTE - motivacion del diagnostico extra en la rama disipativa:
#    en una corrida previa con N_theta=100 barriendo theta, n_corregidos_d
#    (correcciones en la rama DISIPATIVA) llego a valores como 1929 en un
#    solo theta, sobre un total de Ncyc*spc=60000 puntos. Esto es sospechoso:
#    no se espera que la heuristica, pensada para 1-2 saltos puntuales por
#    redondeo, dispare miles de "correcciones" en una sola curva. Este script
#    imprime el numero de saltos detectados/corregidos en fd para chequear
#    si esto tambien aparece en este caso de control mas simple, antes de
#    asumir que es exclusivo del barrido completo.
# =============================================================================
 
import sys, os
import numpy as np
# sys.path.insert(0, '/mnt/project')      # <-- AJUSTAR al path con jcm_lib.py
# os.chdir('/mnt/project')
import warnings; warnings.filterwarnings('ignore')
import matplotlib #; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qutip import (basis, tensor, qeye, destroy, sigmam, sigmap, sigmaz,
                   Qobj, mesolve)

from jcm_lib_completa import fases as fases_vieja

from jcm_lib import fases

# ----------------------------- PARAMETROS ------------------------------------
g       = 0.01
gamma   = 0.1   * g
p_at    = 0.05  * g
Delta   = 0.5*g        # resonancia exacta: aca el usuario espera inestabilidad
Ncyc    = 4
spc     = 10000
# -----------------------------------------------------------------------------
 
# ============================================================================
#  OPERADORES Y HAMILTONIANO (idéntico a code1_Neff2_realphase.py)
# ============================================================================
N_c = 3
e = basis(2,0); gr = basis(2,1)
a   = tensor(qeye(2),qeye(2),destroy(N_c))
sm1 = tensor(sigmam(),qeye(2),qeye(N_c)); sp1 = tensor(sigmap(),qeye(2),qeye(N_c))
sm2 = tensor(qeye(2),sigmam(),qeye(N_c)); sp2 = tensor(qeye(2),sigmap(),qeye(N_c))
sz1 = tensor(sigmaz(),qeye(2),qeye(N_c)); sz2 = tensor(qeye(2),sigmaz(),qeye(N_c))
n2  = tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
 
ee0 = tensor(e,e,basis(N_c,0))
eg1 = tensor(e,gr,basis(N_c,1)); ge1 = tensor(gr,e,basis(N_c,1))
gg2 = tensor(gr,gr,basis(N_c,2))
 
def H_tcm(d, x=0.0, k=0.0, J=0.0):
    return (x*n2 + d/2*(sz1+sz2)
            + g*((sm1+sm2)*a.dag() + (sp1+sp2)*a)
            + 2*k*(sm1*sp2 + sp1*sm2) + J*sz1*sz2)
 
def l_ops():
    return [np.sqrt(gamma)*a, np.sqrt(p_at)*sm1, np.sqrt(p_at)*sm2]
 
def energy_eigenbasis(d):
    """Devuelve los autovalores del bloque 3x3 del manifold N=1 (no del
    Hamiltoniano completo). Omega debe calcularse a partir de ESTE bloque,
    no de H_tcm(Delta).eigenstates(), que diagonaliza las 12 dimensiones
    completas y mezcla escalas de energia de distintos manifolds de
    excitacion, dando un Omega incorrecto (factor ~4.7x mas chico en el
    caso de control con Delta=0 verificado)."""
    Hblock = np.array([[d,            np.sqrt(2)*g, 0   ],
                       [np.sqrt(2)*g, 0,            2*g ],
                       [0,            2*g,         -d   ]])
    evals, evecs = np.linalg.eigh(Hblock)
    return evals

 
# ============================================================================
#  ESTADO INICIAL FIJO: psi0 = (eg1+ge1)/sqrt2, el caso pedido por el usuario.
# ============================================================================
psi0 = (eg1 + ge1).unit()
 
# ============================================================================
#  CORRIDA PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    evals = energy_eigenbasis(Delta)
    Omega = max(abs(evals[2] - evals[0]),abs(evals[2] - evals[1]),abs(evals[1] - evals[0]))
    T = 2*np.pi/Omega
    t = np.linspace(0, Ncyc*T, Ncyc*spc)
 
    print(f"Omega = {Omega:.6g}  (g={g}, Delta={Delta})")
    print(f"T = {T:.6g}, Ncyc={Ncyc}, spc={spc}, puntos totales={Ncyc*spc}")
    print()
 
    print("Corriendo mesolve unitario...")
    su = mesolve(H_tcm(Delta), psi0, t, c_ops=[])
    print("Corriendo mesolve disipativo...")
    sd = mesolve(H_tcm(Delta), psi0, t, c_ops=l_ops())
    print("Listo.\n")
 
    # --- rama unitaria: SIN y CON correccion ---
    print("="*70)
    print("RAMA UNITARIA (fu)")
    print("="*70)
    fu, arg_u = fases(su, open_system=False)
    fu_vieja, arg_u_vieja = fases_vieja(su, open_system=False)
    print()
 
    # --- rama disipativa: SIN y CON correccion ---
    print("="*70)
    print("RAMA DISIPATIVA (fd)")
    print("="*70)
    fd, arg_d, evalsd_t, evecsd_t = fases(sd, open_system=True)
    fd_vieja, arg_d_vieja, evalsd_t, evecsd_t = fases_vieja(sd,open_system=True)
    # np.savetxt('orbits/data/evalsd_t_sin.txt',evalsd_t)
    np.savetxt('orbits/data/fg_u.txt',fu)
    np.savetxt('orbits/data/arg_global_u.txt',arg_u)
    np.savetxt('orbits/data/fg_d.txt',fd)
    np.savetxt('orbits/data/arg_global_d.txt',arg_d)
    np.savetxt('orbits/data/fg_u_vieja.txt',fu_vieja)
    np.savetxt('orbits/data/arg_global_u_vieja.txt',arg_u_vieja)
    np.savetxt('orbits/data/fg_d_vieja.txt',fd_vieja)
    np.savetxt('orbits/data/arg_global_vieja.txt',arg_d_vieja)
    print()

    # ========================================================================
    #  FIGURAS
    # ========================================================================
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
 
    axs[0].plot(t/T, fu/np.pi, label=r'$\phi_u$', color='tab:blue', alpha=0.6)
    axs[0].plot(t/T, fd/np.pi, label=r'$\phi_d$', color='tab:orange', alpha=0.6)
    axs[0].set_ylabel(r'$\phi_u(t)/\pi$')
    axs[0].legend(fontsize=8)
    axs[0].set_title('Fases nueva')
 
    axs[1].plot(t/T, fu_vieja/np.pi, label=r'$\phi_u$', color='tab:blue', alpha=0.6)
    axs[1].plot(t/T, fd_vieja/np.pi, label=r'$\phi_d$', color='tab:orange', alpha=0.6)
    axs[1].set_ylabel(r'$\phi_d(t)/\pi$')
    axs[1].legend(fontsize=8)
    axs[1].set_title('Fases vieja')
 
    axs[2].plot(t/T, (fd-fu)/np.pi, label=r'$\delta\phi$ nueva', color='tab:green', alpha=0.6)
    axs[2].plot(t/T, (fd_vieja-fu_vieja)/np.pi, label=r'$\delta\phi$ vieja', color='black', alpha=0.6)
    axs[2].axhline(0, color='k', lw=0.4)
    axs[2].set_ylabel(r'$\delta\phi(t)/\pi$')
    axs[2].set_xlabel(r'$t/T$')
    axs[2].legend(fontsize=8)
    axs[2].set_title(r'$\delta\phi = \phi_d - \phi_u$')
 
    plt.tight_layout()
    # plt.show()
    plt.savefig('orbits/imgs/test_fg.png', dpi=150, bbox_inches='tight')
    print("\nGuardado: test_correccion_saltos.png")
    
    # mask_zoom = t <= 2*T
    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 11), sharex=True)
    # axs2[0].plot(t[mask_zoom]/T, fu_sin[mask_zoom]/np.pi, color='tab:blue', alpha=0.6, label='sin corregir')
    # axs2[0].plot(t[mask_zoom]/T, fu_con[mask_zoom]/np.pi, color='tab:blue', ls='--', label='con corregir')
    # axs2[0].set_ylabel(r'$\phi_u(t)/\pi$'); axs2[0].legend(fontsize=8); axs2[0].set_title('Rama unitaria (zoom 2 ciclos)')
    # axs2[1].plot(t[mask_zoom]/T, fd_sin[mask_zoom]/np.pi, color='tab:orange', alpha=0.6, label='sin corregir')
    # axs2[1].plot(t[mask_zoom]/T, fd_con[mask_zoom]/np.pi, color='tab:orange', ls='--', label='con corregir')
    # axs2[1].set_ylabel(r'$\phi_d(t)/\pi$'); axs2[1].legend(fontsize=8); axs2[1].set_title('Rama disipativa (zoom 2 ciclos)')
    # axs2[2].plot(t[mask_zoom]/T, delta_sin[mask_zoom]/np.pi, color='tab:green', alpha=0.6, label='sin corregir')
    # axs2[2].plot(t[mask_zoom]/T, delta_con[mask_zoom]/np.pi, color='tab:green', ls='--', label='con corregir')
    # axs2[2].axhline(0, color='k', lw=0.4)
    # axs2[2].set_ylabel(r'$\delta\phi(t)/\pi$'); axs2[2].set_xlabel(r'$t/T$'); axs2[2].legend(fontsize=8)
    # axs2[2].set_title(r'$\delta\phi$ (zoom 2 ciclos)')
    axs2.scatter(t/T,arg_u,color='black',label='u')
    axs2.scatter(t/T,arg_u_vieja,color='green',label='u vieja')
    axs2.scatter(t/T,arg_d,color='red',label='d')
    axs2.scatter(t/T,arg_d_vieja,color='blue',label='d vieja')
    axs2.set_xlabel('t/T')
    axs2.set_xlabel(r'Arg[$\langle\psi(t)|\psi_0\rangle$]')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('orbits/imgs/test_arg.png', dpi=150, bbox_inches='tight')
    print("Guardado: test_correccion_saltos_zoom.png")