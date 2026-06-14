#!/usr/bin/env python3
# =============================================================================
#  CODIGO 1 — Robustez de la fase geometrica sobre la familia N_eff = 2
#             con AMPLITUDES REALES (sin barrido de fases relativas).
#
#  Idea:
#    - La familia de estados iniciales con N_eff = 2 (sobre la base de autoestados
#      de energia) es la interseccion de un plano (sum p_k = 1) con una esfera
#      (sum p_k^2 = 1/2). Es un circulo, parametrizado por UN solo angulo theta.
#    - Para cada theta construimos el estado inicial con amplitudes reales
#      c_k = sqrt(p_k(theta)) sobre los autoestados de energia, lo evolucionamos
#      con y sin disipacion, y medimos si la fase geometrica es robusta.
#
#  Criterio de robustez (metodologia validada):
#    delta_phi(t) = phi_disipativa(t) - phi_unitaria(t).
#    Robusto  => la LINEA DE BASE de delta_phi(t) (los valles entre picos) se
#                mantiene ~0 a todo tiempo. Los picos transitorios que suben y
#                vuelven a 0 son el desfasaje temporal de los saltos y no cuentan.
#    Metrica  => max sobre ciclos de |valle|, donde valle = min |delta_phi| dentro
#                de cada ciclo. Robusto => metrica ~ 0; no robusto => crece.
# =============================================================================

import sys, os
import numpy as np
# sys.path.insert(0, '/mnt/project')      # <-- AJUSTAR al path con jcm_lib.py
# os.chdir('/mnt/project')
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qutip import (basis, tensor, qeye, destroy, sigmam, sigmap, sigmaz,
                   Qobj, mesolve)
from jcm_lib import fases

# ----------------------------- PARAMETROS ------------------------------------
g       = 0.01
gamma   = 0.1   * g        # tasa de perdida de cavidad
p_at    = 0.005 * g        # tasa de decaimiento atomico (igual en ambos atomos)
Delta   = 0.0              # resonancia (poner !=0 para explorar fuera de resonancia)
N_theta = 60               # numero de puntos en el barrido de theta
Ncyc    = 10               # numero de oscilaciones (ciclos de Rabi)
spc     = 1500             # pasos temporales por ciclo (resolucion)
# -----------------------------------------------------------------------------

# ============================================================================
#  1) OPERADORES Y HAMILTONIANO DEL TCM (dos atomos + cavidad), manifold N=2.
# ============================================================================
N_c = 3
e = basis(2,0); gr = basis(2,1)
a   = tensor(qeye(2),qeye(2),destroy(N_c))
sm1 = tensor(sigmam(),qeye(2),qeye(N_c)); sp1 = tensor(sigmap(),qeye(2),qeye(N_c))
sm2 = tensor(qeye(2),sigmam(),qeye(N_c)); sp2 = tensor(qeye(2),sigmap(),qeye(N_c))
sz1 = tensor(sigmaz(),qeye(2),qeye(N_c)); sz2 = tensor(qeye(2),sigmaz(),qeye(N_c))
n2  = tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))

# Estados base del bloque N=2:  phi1=|ee,0>, phi2=(|eg,1>+|ge,1>)/sqrt2, phi3=|gg,2>
ee0 = tensor(e,e,basis(N_c,0))
eg1 = tensor(e,gr,basis(N_c,1)); ge1 = tensor(gr,e,basis(N_c,1))
gg2 = tensor(gr,gr,basis(N_c,2))
phi1, phi2, phi3 = ee0, (eg1+ge1).unit(), gg2

def H_tcm(d, x=0.0, k=0.0, J=0.0):
    """Hamiltoniano de Tavis-Cummings no lineal (Kerr x, dipolo k, Ising J)."""
    return (x*n2 + d/2*(sz1+sz2)
            + g*((sm1+sm2)*a.dag() + (sp1+sp2)*a)
            + 2*k*(sm1*sp2 + sp1*sm2) + J*sz1*sz2)

def l_ops():
    """Operadores de colapso: perdida de cavidad + decaimiento de cada atomo."""
    return [np.sqrt(gamma)*a, np.sqrt(p_at)*sm1, np.sqrt(p_at)*sm2]

# ============================================================================
#  2) BASE DE AUTOESTADOS DE ENERGIA del bloque 3x3 (depende de Delta).
#     En resonancia (Delta=0): el del medio es el DARK STATE (energia 0).
# ============================================================================
def energy_eigenbasis(d):
    """Devuelve [|E0>,|E1>,|E2>] (autoestados de energia, ordenados por energia)
    y sus energias. Construidos en el espacio completo a partir del bloque 3x3."""
    Hblock = np.array([[d,            np.sqrt(2)*g, 0   ],
                       [np.sqrt(2)*g, 0,            2*g ],
                       [0,            2*g,         -d   ]])
    evals, evecs = np.linalg.eigh(Hblock)            # columnas = autovectores
    def lift(k):  # combina los phi_i con los coeficientes del autovector k
        return (evecs[0,k]*phi1 + evecs[1,k]*phi2 + evecs[2,k]*phi3).unit()
    return [lift(0), lift(1), lift(2)], evals

# ============================================================================
#  3) PARAMETRIZACION DE LA FAMILIA N_eff = 2 (circulo plano-esfera).
#     Centro = equiparticion (1/3,1/3,1/3); radio r = sqrt(1/2 - 1/3).
#     theta recorre el circulo; algunos theta dan p_k<0 (fuera del simplex fisico)
#     y se descartan.
# ============================================================================
# base ortonormal del plano sum p_k = 1
e1_hat = np.array([1, -1,  0]) / np.sqrt(2)
e2_hat = np.array([1,  1, -2]) / np.sqrt(6)
origin = np.array([1/3, 1/3, 1/3])
r_Neff2 = np.sqrt(1/2 - 1/3)        # radio que da N_eff = 2

def populations(theta):
    """Poblaciones (p1,p2,p3) sobre los autoestados de energia para un theta."""
    return origin + r_Neff2*(np.cos(theta)*e1_hat + np.sin(theta)*e2_hat)

def initial_state(theta, eigstates):
    """Estado inicial = sum_k sqrt(p_k) |E_k>  (amplitudes REALES positivas).
    Devuelve None si las poblaciones se salen del simplex fisico."""
    p = populations(theta)
    if np.any(p < -1e-9):
        return None
    p = np.clip(p, 0, None)
    c = np.sqrt(p)                                  # amplitudes reales positivas
    psi = c[0]*eigstates[0] + c[1]*eigstates[1] + c[2]*eigstates[2]
    return psi.unit()

# ============================================================================
#  4) CALCULO DE delta_phi(t) Y METRICA DE VALLE.
# ============================================================================
def delta_phi_curve(H, psi0, c_ops, T, Ncyc, spc):
    """Evoluciona psi0 con y sin disipacion y devuelve (t, delta_phi(t))."""
    t  = np.linspace(0, Ncyc*T, Ncyc*spc)
    su = mesolve(H, psi0, t, c_ops=[])              # unitaria
    sd = mesolve(H, psi0, t, c_ops=c_ops)           # disipativa
    fu, _       = fases(su, open_system=False)      # fase geometrica unitaria
    fd, _, _, _ = fases(sd, open_system=True)       # fase geometrica disipativa
    return t, (fd - fu)                             # SIN wrapping

def valley_metric(d, Ncyc, spc):
    """max sobre ciclos de |valle|. valle = valor de d en el punto de |d| minimo
    dentro de cada ciclo (la linea de base, ya que los picos son breves)."""
    valleys = []
    for c in range(Ncyc):
        seg = d[c*spc:(c+1)*spc]
        if len(seg) == 0:
            continue
        valleys.append(seg[np.argmin(np.abs(seg))])
    valleys = np.array(valleys)
    return np.max(np.abs(valleys)), valleys

# ============================================================================
#  5) BARRIDO PRINCIPAL sobre theta.
# ============================================================================
if __name__ == "__main__":
    eigstates, evals = energy_eigenbasis(Delta)
    Omega = abs(evals[2] - evals[0])                # gap mayor => frecuencia de ciclo
    T = 2*np.pi/Omega

    theta_arr = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    metric    = np.full(N_theta, np.nan)
    darkw     = np.full(N_theta, np.nan)            # peso en el dark state (diagnostico)

    for i, th in enumerate(theta_arr):
        psi0 = initial_state(th, eigstates)
        if psi0 is None:                            # theta fuera del simplex fisico
            continue
        _, d = delta_phi_curve(H_tcm(Delta), psi0, l_ops(), T, Ncyc, spc)
        m, _ = valley_metric(d, Ncyc, spc)
        metric[i] = m
        # peso sobre el autoestado del medio (dark en resonancia), para diagnostico
        darkw[i] = populations(th)[1]
        print(f"  theta={th/np.pi:5.2f}pi  N_eff=2  dark_weight={darkw[i]:.3f}  metric={m:8.4f}")

    # --------------------------- guardar + graficar --------------------------
    np.savez('orbits/data/code1_Neff2_realphase.npz',
             theta=theta_arr, metric=metric, darkw=darkw,
             g=g, gamma=gamma, p_at=p_at, Delta=Delta, Ncyc=Ncyc, spc=spc)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(theta_arr, metric, 'o-', color='#228833')
    ax.set_xlabel(r'$\theta$ (parametriza la familia $N_{\rm eff}=2$)')
    ax.set_ylabel(r'robustez: $\max_{\rm cyc}|\delta\phi_{\rm base}|$')
    ax.set_title(rf'Robustez sobre la familia $N_{{\rm eff}}=2$ (fase real), '
                 rf'$\Delta={Delta/g}g$, $N_{{\rm cyc}}={Ncyc}$')
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('orbits/imgs/code1_Neff2_realphase.png', dpi=150, bbox_inches='tight')
    print("\nGuardado: code1_Neff2_realphase.npz / .png")
