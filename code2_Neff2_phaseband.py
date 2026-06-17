#!/usr/bin/env python3
# =============================================================================
#  CODIGO 2 — Robustez sobre la familia N_eff = 2 INCLUYENDO fases relativas.
#
#  Motivacion:
#    En el Codigo 1 fijamos amplitudes reales c_k = sqrt(p_k). Pero un estado con
#    las mismas poblaciones p_k puede tener fases relativas arbitrarias:
#        |psi> = sum_k sqrt(p_k) e^{i alpha_k} |E_k>.
#    Una fase global no importa => quedan dos fases libres (phi1, phi2) = (alpha_2-alpha_1,
#    alpha_3-alpha_1). La pregunta es: ¿las fases relativas pueden INDUCIR (o destruir)
#    robustez que el barrido de fase real no ve?
#
#  Estrategia (mezcla de "banda min-max" + "optimizacion"):
#    Para cada theta (poblaciones fijas, N_eff=2), en vez de barrer (phi1,phi2) en
#    una grilla densa (caro e incompleto), OPTIMIZAMOS la metrica de robustez sobre
#    (phi1,phi2) buscando su MINIMO y su MAXIMO. Eso da la BANDA [M_min, M_max](theta):
#        - M_min(theta) ~ 0  => existe un estado robusto con esas poblaciones.
#        - banda angosta     => las fases no importan (Codigo 1 era suficiente).
#        - banda ancha       => las fases SI importan; el cuadro de fase real era incompleto.
#
#  Cuidados (ver funcion check_metric_smoothness, correr ANTES):
#    - La optimizacion continua (Nelder-Mead) asume que la metrica varia suave con
#      (phi1,phi2). La metrica de valle tiene un argmin adentro que puede dar escalones.
#      Si no es suave, la optimizacion no es confiable: hay que suavizar la metrica.
#    - Nelder-Mead cae en minimos locales => usamos MULTI-START (varios puntos iniciales).
# =============================================================================

import sys, os
import numpy as np
sys.path.insert(0, '/mnt/project')      # <-- AJUSTAR
os.chdir('/mnt/project')
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qutip import (basis, tensor, qeye, destroy, sigmam, sigmap, sigmaz,
                   Qobj, mesolve)
from jcm_lib import fases

# ----------------------------- PARAMETROS ------------------------------------
g       = 0.01
gamma   = 0.1   * g
p_at    = 0.005 * g
Delta   = 0.0
N_theta = 12              # theta GRUESO: solo queremos saber si las fases cambian el cuadro
Ncyc    = 10
spc     = 1500
N_start = 5               # numero de puntos iniciales para el multi-start del optimizador
# -----------------------------------------------------------------------------

# ============================================================================
#  1) OPERADORES, HAMILTONIANO, BASE DE ENERGIA  (identico al Codigo 1)
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
phi1, phi2, phi3 = ee0, (eg1+ge1).unit(), gg2

def H_tcm(d, x=0.0, k=0.0, J=0.0):
    return (x*n2 + d/2*(sz1+sz2)
            + g*((sm1+sm2)*a.dag() + (sp1+sp2)*a)
            + 2*k*(sm1*sp2 + sp1*sm2) + J*sz1*sz2)
def l_ops():
    return [np.sqrt(gamma)*a, np.sqrt(p_at)*sm1, np.sqrt(p_at)*sm2]

def energy_eigenbasis(d):
    Hblock = np.array([[d,            np.sqrt(2)*g, 0   ],
                       [np.sqrt(2)*g, 0,            2*g ],
                       [0,            2*g,         -d   ]])
    evals, evecs = np.linalg.eigh(Hblock)
    def lift(k):
        return (evecs[0,k]*phi1 + evecs[1,k]*phi2 + evecs[2,k]*phi3).unit()
    return [lift(0), lift(1), lift(2)], evals

# ============================================================================
#  2) FAMILIA N_eff=2 CON FASES:  ahora el estado depende de (theta, phi1, phi2)
# ============================================================================
e1_hat = np.array([1, -1,  0]) / np.sqrt(2)
e2_hat = np.array([1,  1, -2]) / np.sqrt(6)
origin = np.array([1/3, 1/3, 1/3])
r_Neff2 = np.sqrt(1/2 - 1/3)

def populations(theta):
    return origin + r_Neff2*(np.cos(theta)*e1_hat + np.sin(theta)*e2_hat)

def initial_state(theta, phi1_, phi2_, eigstates):
    """Estado con poblaciones p_k(theta) y fases relativas (phi1_, phi2_):
       |psi> = sqrt(p1)|E0> + sqrt(p2)e^{i phi1_}|E1> + sqrt(p3)e^{i phi2_}|E2>."""
    p = populations(theta)
    if np.any(p < -1e-9):
        return None
    p = np.clip(p, 0, None); c = np.sqrt(p)
    psi = (c[0]*eigstates[0]
           + c[1]*np.exp(1j*phi1_)*eigstates[1]
           + c[2]*np.exp(1j*phi2_)*eigstates[2])
    return psi.unit()

# ============================================================================
#  3) METRICA (identica al Codigo 1)
# ============================================================================
def delta_phi_curve(H, psi0, c_ops, T, Ncyc, spc):
    t  = np.linspace(0, Ncyc*T, Ncyc*spc)
    su = mesolve(H, psi0, t, c_ops=[])
    sd = mesolve(H, psi0, t, c_ops=c_ops)
    fu, _       = fases(su, open_system=False)
    fd, _, _, _ = fases(sd, open_system=True)
    return t, (fd - fu)

def valley_metric(d, Ncyc, spc):
    valleys = []
    for c in range(Ncyc):
        seg = d[c*spc:(c+1)*spc]
        if len(seg) == 0: continue
        valleys.append(seg[np.argmin(np.abs(seg))])
    return np.max(np.abs(np.array(valleys)))

def robustness(theta, phi1_, phi2_, eigstates, T):
    """Metrica de robustez para un estado (theta, phi1_, phi2_). Mayor => menos robusto."""
    psi0 = initial_state(theta, phi1_, phi2_, eigstates)
    if psi0 is None:
        return np.nan
    _, d = delta_phi_curve(H_tcm(Delta), psi0, l_ops(), T, Ncyc, spc)
    return valley_metric(d, Ncyc, spc)

# ============================================================================
#  4) CHEQUEO DE SUAVIDAD  (CORRER ESTO PRIMERO, ANTES DE OPTIMIZAR)
#     Evalua la metrica en una grilla gruesa de (phi1,phi2) para un theta fijo
#     y reporta cuan "escalonada" es. Si es muy escalonada, la optimizacion
#     continua no es confiable y conviene suavizar la metrica.
# ============================================================================
def check_metric_smoothness(theta_test, eigstates, T, ngrid=9):
    """Devuelve la grilla de metrica y una medida de rugosidad (variacion entre
    vecinos relativa al rango total). Imprime un diagnostico."""
    ph = np.linspace(0, 2*np.pi, ngrid, endpoint=False)
    M = np.full((ngrid, ngrid), np.nan)
    for i, p1 in enumerate(ph):
        for j, p2 in enumerate(ph):
            M[i,j] = robustness(theta_test, p1, p2, eigstates, T)
    finite = M[np.isfinite(M)]
    if len(finite) < 2:
        print("  [suavidad] insuficientes puntos validos."); return M, np.nan
    rng = np.ptp(finite)
    # rugosidad = media de |diferencia entre vecinos| / rango
    dn = []
    for i in range(ngrid):
        for j in range(ngrid):
            if not np.isfinite(M[i,j]): continue
            for (di,dj) in [(1,0),(0,1)]:
                ii,jj = (i+di)%ngrid, (j+dj)%ngrid
                if np.isfinite(M[ii,jj]):
                    dn.append(abs(M[i,j]-M[ii,jj]))
    rough = (np.mean(dn)/rng) if (rng>0 and dn) else np.nan
    print(f"  [suavidad] theta={theta_test:.2f}: rango metrica={rng:.4f}, "
          f"rugosidad={rough:.3f}  "
          f"({'OK, suave' if rough<0.25 else 'ESCALONADA: optimizacion dudosa'})")
    return M, rough

# ============================================================================
#  5) OPTIMIZACION MIN/MAX con MULTI-START (Nelder-Mead) sobre (phi1,phi2)
# ============================================================================
def optimize_band(theta, eigstates, T, find_max=False):
    """Encuentra min (o max) de la metrica sobre (phi1,phi2) usando multi-start.
    find_max=False -> minimiza la metrica (busca el estado mas robusto).
    find_max=True  -> maximiza (busca el menos robusto)."""
    sign = -1.0 if find_max else 1.0
    def obj(x):
        val = robustness(theta, x[0], x[1], eigstates, T)
        return sign*val if np.isfinite(val) else 1e6   # penaliza no-fisicos
    best = None
    # multi-start: puntos iniciales sembrados en la grilla de fases
    seeds = np.linspace(0, 2*np.pi, N_start, endpoint=False)
    for s1 in seeds:
        for s2 in seeds:
            res = minimize(obj, x0=[s1, s2], method='Nelder-Mead',
                           options={'xatol':1e-2, 'fatol':1e-4, 'maxiter':100})
            if best is None or res.fun < best.fun:
                best = res
    return sign*best.fun, best.x        # valor optimo, fases optimas

# ============================================================================
#  6) PROGRAMA PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    eigstates, evals = energy_eigenbasis(Delta)
    Omega = abs(evals[2] - evals[0]); T = 2*np.pi/Omega

    # ---- PASO 1: chequear suavidad de la metrica en un par de theta ----
    print("=== CHEQUEO DE SUAVIDAD DE LA METRICA (correr e inspeccionar) ===")
    for th_test in [np.pi/4, np.pi]:
        check_metric_smoothness(th_test, eigstates, T, ngrid=9)
    print("Si la metrica es ESCALONADA, NO confiar en la optimizacion de abajo.\n")

    # ---- PASO 2: banda min-max sobre theta ----
    print("=== BANDA min-max sobre la familia N_eff=2 ===")
    theta_arr = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    M_min = np.full(N_theta, np.nan); M_max = np.full(N_theta, np.nan)
    ph_min = np.full((N_theta,2), np.nan); ph_max = np.full((N_theta,2), np.nan)
    for i, th in enumerate(theta_arr):
        if initial_state(th, 0, 0, eigstates) is None:
            continue
        mn, xmn = optimize_band(th, eigstates, T, find_max=False)
        mx, xmx = optimize_band(th, eigstates, T, find_max=True)
        M_min[i], M_max[i] = mn, mx
        ph_min[i], ph_max[i] = xmn, xmx
        print(f"  theta={th:5.2f}: M_min={mn:7.4f} (phi={xmn[0]:.2f},{xmn[1]:.2f}) | "
              f"M_max={mx:7.4f}")

    np.savez('code2_Neff2_phaseband.npz',
             theta=theta_arr, M_min=M_min, M_max=M_max,
             ph_min=ph_min, ph_max=ph_max,
             g=g, gamma=gamma, p_at=p_at, Delta=Delta, Ncyc=Ncyc, spc=spc)

    # ---- figura: banda min-max ----
    fig, ax = plt.subplots(figsize=(9,5))
    ax.fill_between(theta_arr, M_min, M_max, alpha=0.3, color='#4477AA',
                    label='banda sobre fases relativas')
    ax.plot(theta_arr, M_min, 'o-', color='#228833', label=r'$M_{\min}$ (mas robusto)')
    ax.plot(theta_arr, M_max, 's-', color='#EE6677', label=r'$M_{\max}$ (menos robusto)')
    ax.set_xlabel(r'$\theta$ (familia $N_{\rm eff}=2$)')
    ax.set_ylabel(r'robustez: $\max_{\rm cyc}|\delta\phi_{\rm base}|$')
    ax.set_title(rf'Banda de robustez sobre fases relativas, $\Delta={Delta/g}g$')
    ax.axhline(0, color='k', lw=0.4); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('code2_Neff2_phaseband.png', dpi=150, bbox_inches='tight')
    print("\nGuardado: code2_Neff2_phaseband.npz / .png")
    print("INTERPRETACION:")
    print("  - M_min(theta) ~ 0  => existe estado robusto con esas poblaciones.")
    print("  - banda angosta     => las fases no importan (Codigo 1 alcanzaba).")
    print("  - banda ancha       => las fases relativas SI cambian la robustez.")