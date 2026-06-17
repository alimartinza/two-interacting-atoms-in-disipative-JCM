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
#
#  CORRECCION DE SALTOS CON SIGNO INVERTIDO (agregada despues, ver jcm_lib.fases):
#    En resonancia exacta (y en general cuando el estado inicial es geodesico,
#    es decir combinacion de solo 2 de los 3 autoestados de energia), phi_u(t)
#    deberia ser una escalera de saltos limpios de magnitud fija. Un error de
#    redondeo de np.angle/np.unwrap cerca del corte de rama puede invertir el
#    signo de algun salto puntual, haciendo que phi_u y phi_g se vean muy
#    distintas aunque deberian coincidir. fases(..., corregir_saltos=True,
#    Omega=Omega) detecta estos saltos via un umbral ABSOLUTO en |dphi/dt|
#    (anclado a Omega, no a un percentil de la curva, para no confundir
#    transiciones suaves -lejos de resonancia- con discontinuidades numericas)
#    y corrige el signo de los saltos minoritarios SOLO si hay una mayoria
#    clara de saltos similares que respalde el signo correcto.
#    ADVERTENCIA: esto es la correccion LOCAL/heuristica, mas barata (no
#    resimula nada). NO es la verificacion mas fuerte por continuidad en
#    theta+-epsilon (ver jcm_lib.verificar_signo_por_continuidad), que hay
#    que correr aparte sobre los theta que esta correccion marque como
#    corregidos, antes de confiar en el resultado para el paper. Revisar
#    siempre los mensajes "[correccion] ..." impresos en la consola.
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
from jcm_lib import fases, verificar_signo_por_continuidad

# ----------------------------- PARAMETROS ------------------------------------
g       = 0.01
gamma   = 0.1   * g        # tasa de perdida de cavidad
p_at    = 0.05 * g        # tasa de decaimiento atomico (igual en ambos atomos)
Delta   = 0*g              # resonancia (poner !=0 para explorar fuera de resonancia)
N_theta = 100               # numero de puntos en el barrido de theta
Ncyc    = 6            # numero de oscilaciones (ciclos de Rabi)
spc     = 10000             # pasos temporales por ciclo (resolucion)
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
def delta_phi_curve(H, psi0, c_ops, T, Ncyc, spc, Omega):
    """Evoluciona psi0 con y sin disipacion y devuelve (t, delta_phi(t)).

    Omega: gap espectral maximo del sistema (|E2-E0| del bloque 3x3), ya
    calculado en el bloque principal. Se pasa a fases(corregir_saltos=True,
    Omega=...) para que la correccion local de saltos abruptos con signo
    invertido use la escala fisica correcta del problema (ver discusion:
    el umbral de deteccion es absoluto, |dphi/dt| > margen*Omega, no un
    percentil relativo de la propia curva).

    IMPORTANTE: corregir_saltos=True aplica SOLO la heuristica local de
    "mayoria de signos" sobre la propia curva, sin resimular nada con
    theta+-epsilon. Es la correccion barata; no es la verificacion mas
    fuerte por continuidad (ver jcm_lib.verificar_signo_por_continuidad).
    Por eso devolvemos tambien log_u/log_d: para poder auditar a mano,
    theta por theta, que correccion se aplico (si alguna) antes de confiar
    en la curva ya corregida.
    """
    t  = np.linspace(0, Ncyc*T, Ncyc*spc)
    su = mesolve(H, psi0, t, c_ops=[])                            # unitaria
    sd = mesolve(H, psi0, t, c_ops=c_ops)                         # disipativa
    # fu, fd ya vienen con la correccion de saltos aplicada (si la heuristica
    # encontro evidencia suficiente; si no, quedan iguales a la version sin
    # corregir, y log_u/log_d lo van a indicar explicitamente).
    fu, _, log_u = fases(su, open_system=False, corregir_saltos=True, Omega=Omega)
    fd, _, evalsd_t, evecsd_t, log_d = fases(sd, open_system=True, corregir_saltos=True, Omega=Omega)
    return t, fd, fu, evecsd_t[0], su, log_u, log_d                          # SIN wrapping

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

def fidelity(state1,state2):
    if state1.isbra or state2.isbra:
        print('en fidelity por favor pone solo kets no bras')
        raise Exception 
    elif state1.isket and state2.isket:
        return np.power(np.abs(state1.dag()*state2),2)
    elif state1.isdm:
        return state2.dag()*state1*state2
    elif state2.isdm:
        return state1.dag()*state2*state1
    else:
        print('hay algo raro en fideilty...')
        raise Exception 

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
    # Resumen de auditoria de la correccion de saltos: cuantos saltos se
    # marcaron 'CORREGIDO' en cada curva (unitaria/disipativa), por theta.
    # NaN donde theta no es fisico (mismo criterio que metric/darkw).
    n_corregidos_u = np.full(N_theta, np.nan)
    n_corregidos_d = np.full(N_theta, np.nan)
    # Resultado de la verificacion FUERTE por continuidad (solo se evalua en
    # los theta donde n_corregidos_u>0, ver mas abajo). Codificacion:
    #   1.0  = verificacion CONSISTENTE (ambos vecinos dan el mismo signo;
    #          la correccion heuristica esta respaldada por evidencia mas fuerte)
    #   0.0  = INCONSISTENTE (los vecinos no coinciden; posible bifurcacion
    #          genuina, NO confiar en la correccion heuristica sin revision manual)
    #  -1.0  = AMBIGUO (algun vecino cayo fuera del simplex fisico, no se pudo evaluar)
    #   NaN  = no se corrio la verificacion (no hubo correccion heuristica que verificar)
    verif_continuidad = np.full(N_theta, np.nan)

    # 'initial_state' necesita (theta, eigstates); aca fijamos eigstates
    # (que depende solo de Delta, no de theta) para obtener una funcion de
    # un solo argumento theta->psi0, que es lo que pide verificar_signo_por_continuidad.
    initial_state_fn = lambda th: initial_state(th, eigstates)
    H_fija = H_tcm(Delta)   # mismo Hamiltoniano para theta+-epsilon (Delta no depende de theta)
    EPSILON_CONTINUIDAD = 1e-3 * (theta_arr[1] - theta_arr[0])  # ver advertencia en jcm_lib:
    # CALIBRAR: este valor es chico respecto al paso del barrido de theta,
    # pero no hay garantia universal de que sea suficiente para alejarse del
    # punto de ambiguedad numerica. Si la verificacion da resultados raros
    # (ej. siempre INCONSISTENTE), probar 2-3 valores de epsilon distintos
    # antes de confiar en una sola corrida (ver docstring en jcm_lib.py).

    fig_fg=plt.figure(figsize=(8,6))
    ax_fg=fig_fg.add_subplot(4,1,(1,3))
    ax_delta=fig_fg.add_subplot(4,1,4)
    colors=plt.colormaps['winter'](np.linspace(0,1,len(theta_arr)))
    fid=np.zeros((N_theta,Ncyc*spc))
    for i, th in enumerate(theta_arr):
        psi0 = initial_state(th, eigstates)
        if psi0 is None:                            # theta fuera del simplex fisico
            continue
        t, fd, fu, pointerstates, sol_u, log_u, log_d = delta_phi_curve(
            H_tcm(Delta), psi0, l_ops(), T, Ncyc, spc, Omega)
        for t_j in range(Ncyc*spc):
            fid[i][t_j]=fidelity(sol_u.states[t_j],pointerstates[t_j])

        n_corregidos_u[i] = sum(1 for e in log_u if e.get('accion') == 'CORREGIDO')
        n_corregidos_d[i] = sum(1 for e in log_d if e.get('accion') == 'CORREGIDO')
        if n_corregidos_u[i] > 0 or n_corregidos_d[i] > 0:
            print(f"    [correccion] theta={th/np.pi:5.2f}π: "
                  f"{int(n_corregidos_u[i])} salto(s) corregido(s) en fu, "
                  f"{int(n_corregidos_d[i])} en fd. REVISAR A MANO antes de confiar.")

        # --- VERIFICACION FUERTE: solo si la heuristica corrigio algo en fu ---
        # (la rama unitaria es donde el usuario reporto ver las inestabilidades;
        # si en el futuro tambien interesa verificar la rama disipativa, replicar
        # este bloque usando log_d y c_ops=l_ops() en vez de c_ops=[] -- pero ahi
        # el costo ya no es despreciable, porque mesolve con c_ops es mas caro).
        if n_corregidos_u[i] > 0:
            saltos_corregidos = [e for e in log_u if e.get('accion') == 'CORREGIDO']
            resultados_verif = []
            for entrada in saltos_corregidos:
                t_sospechoso = entrada['t']
                print(f"      [verificacion fuerte] theta={th/np.pi:.2f}π, "
                      f"salto en t~{t_sospechoso:.4g} ...")
                res = verificar_signo_por_continuidad(
                    H=H_fija, initial_state_fn=initial_state_fn, theta=th,
                    t_salto_sospechoso=t_sospechoso, T=T, Ncyc=Ncyc, spc=spc,
                    Omega=Omega, mesolve_fn=mesolve, epsilon=EPSILON_CONTINUIDAD,
                    verbose=True)
                resultados_verif.append(res['coincide'])
            # Resumimos en un solo codigo por theta (peor caso entre los saltos
            # verificados: si CUALQUIERA de ellos resulto inconsistente o ambiguo,
            # marcamos el theta como tal, para no esconder un problema parcial).
            if any(r is False for r in resultados_verif):
                verif_continuidad[i] = 0.0
                print(f"      [verificacion fuerte] theta={th/np.pi:.2f}π: "
                      f"AL MENOS UN salto resulto INCONSISTENTE. No confiar en la "
                      f"correccion heuristica aqui sin revision manual.")
            elif any(r is None for r in resultados_verif):
                verif_continuidad[i] = -1.0
                print(f"      [verificacion fuerte] theta={th/np.pi:.2f}π: "
                      f"resultado AMBIGUO (vecino fuera del simplex en algun salto).")
            else:
                verif_continuidad[i] = 1.0
                print(f"      [verificacion fuerte] theta={th/np.pi:.2f}π: "
                      f"CONSISTENTE en todos los saltos verificados.")

        ax_fg.plot(t/T,fu/np.pi,color=colors[i])
        ax_fg.plot(t/T,fd/np.pi,color=colors[i],linestyle='dashed')
        ax_delta.plot(t/T,(fd-fu)/np.pi,color=colors[i])
        m, _ = valley_metric((fd-fu)/np.pi, Ncyc, spc)
        metric[i] = m
        # peso sobre el autoestado del medio (dark en resonancia), para diagnostico
        darkw[i] = populations(th)[1]
        print(f"  theta={th/np.pi:5.2f}π  N_eff=2  dark_weight={darkw[i]:.3f}  metric={m:8.4f}")
    plt.savefig(f'orbits/imgs/code1_Neff2_fgcheck{Delta/g}g.png',dpi=150)
    print('Guardada imagen chequeo.')
    # --------------------------- guardar + graficar --------------------------
    np.savez(f'orbits/data/code1_Neff2_realphase{Delta/g}g.npz',
             theta=theta_arr, metric=metric, darkw=darkw,
             n_corregidos_u=n_corregidos_u, n_corregidos_d=n_corregidos_d,
             verif_continuidad=verif_continuidad,
             g=g, gamma=gamma, p_at=p_at, Delta=Delta, Ncyc=Ncyc, spc=spc)


    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(theta_arr, metric, 'o-', color='#228833')
    ax2=ax.twinx()
    ax2.set_ylabel(r'$F(|\langle psi_+ | \psi \rangle )[\tau]$')
    ax2.plot(theta_arr,fid[:,-1],'o-',color='red',label=fr'$\tau={Ncyc}T')
    ax2.plot(theta_arr,fid[:,int(Ncyc*spc/2)],'o-',color='orange',label=fr'$\tau={int(Ncyc/2)}T')
    ax.set_xlabel(r'$\theta$ (parametriza la familia $N_{\rm eff}=2$)')
    ax.set_ylabel(r'robustez: $\max_{\rm cyc}|\delta\phi_{\rm base}|$')
    ax.set_title(rf'Robustez sobre la familia $N_{{\rm eff}}=2$ (fase real), '
                 rf'$\Delta={Delta/g}g$, $N_{{\rm cyc}}={Ncyc}$')
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'orbits/imgs/code1_Neff2_realphase{Delta/g}g.png', dpi=150, bbox_inches='tight')
    print("\nGuardado: code1_Neff2_realphase.npz / .png")