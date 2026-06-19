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

def _detectar_saltos_absoluto(t, phi, Omega, margen=5.0):
    """
    Encuentra puntos donde |dphi/dt| excede un umbral ABSOLUTO ligado a la
    escala fisica del sistema (margen*Omega), en vez de un umbral relativo
    (percentil) que siempre encuentra "algo" incluso en curvas suaves.

    Por que un umbral absoluto funciona: una discontinuidad numerica (error
    de redondeo en np.angle/np.unwrap cerca del corte de rama) ocurre en UN
    paso de integracion dt_sim, asi que su derivada aparente es ~1/dt_sim,
    ORDENES DE MAGNITUD mayor que Omega (la escala de variacion fisica del
    sistema, ej. el gap espectral maximo |E2-E0|). Una transicion suave
    nunca excede ~Omega por construccion fisica, asi que la separacion de
    escalas permite un umbral fijo sin calibrar caso por caso.

    Devuelve: (t_saltos, mags_con_signo, bordes, umbral_usado)
    """
    if len(t) < 3:
        return [], np.array([]), [], margen * Omega
    dt = np.diff(t)
    deriv = np.diff(phi) / dt
    umbral = margen * Omega
    es_salto = np.abs(deriv) > umbral
    idx_salto = np.where(es_salto)[0]
    if len(idx_salto) == 0:
        return [], np.array([]), [], umbral
    grupos = np.split(idx_salto, np.where(np.diff(idx_salto) > 1)[0] + 1)
    t_saltos = [t[g].mean() for g in grupos]
    mags, bordes = [], []
    for g in grupos:
        i0 = max(g[0] - 1, 0)
        i1 = min(g[-1] + 2, len(phi) - 1)
        mags.append(phi[i1] - phi[i0])
        bordes.append((i0, i1))
    return t_saltos, np.array(mags), bordes, umbral


def _agrupar_por_magnitud(abs_mags, tol_rel=0.15):
    """Agrupa indices de saltos por similitud de magnitud (sin mirar signo),
    via clustering simple por umbral relativo. El grupo con mas miembros
    define la magnitud "tipica" del salto geodesico, inferida de los propios
    datos (no se asume de antemano que valga pi, 2pi, etc.)."""
    if len(abs_mags) == 0:
        return []
    orden = np.argsort(abs_mags)
    usados = np.zeros(len(abs_mags), dtype=bool)
    grupos = []
    for i in orden:
        if usados[i]:
            continue
        ref = abs_mags[i]
        miembros = [j for j in range(len(abs_mags))
                    if not usados[j] and abs(abs_mags[j] - ref) <= tol_rel * max(ref, 1e-12)]
        for j in miembros:
            usados[j] = True
        grupos.append(miembros)
    return grupos


def _corregir_signos_saltos(t, phi, Omega, margen=5.0, tol_rel=0.15, min_grupo=2, verbose=False):
    """
    Aplica deteccion + correccion local de saltos con signo invertido sobre
    UNA curva de fase ya calculada (phi = Pan, ANTES de np.unwrap).

    Logica de correccion: si la mayoria de los saltos abruptos detectados
    comparten una magnitud similar pero unos pocos tienen el signo opuesto,
    esos pocos se reinterpretan como error de redondeo de np.angle cerca del
    corte de rama, y se corrigen al signo mayoritario. Si no hay una mayoria
    clara (menos de 'min_grupo' saltos del mismo tipo), NO se corrige nada:
    mas vale no tocar que corregir sin evidencia suficiente.

    Esta funcion NO resimula nada ni necesita el Hamiltoniano: solo usa la
    curva ya calculada. Ver tambien verificar_signo_por_continuidad() en
    este mismo modulo para una verificacion mas fuerte (pero mas cara)
    cuando se dispone del Hamiltoniano y la parametrizacion del estado
    inicial.

    Devuelve: (phi_corregida, log_de_acciones)
    """
    t_s, mags, bordes, umbral = _detectar_saltos_absoluto(t, phi, Omega, margen=margen)
    phi_corr = phi.copy()
    log = []

    if len(mags) == 0:
        return phi_corr, log

    abs_mags = np.abs(mags)
    grupos_mag = _agrupar_por_magnitud(abs_mags, tol_rel=tol_rel)
    grupo_dom = max(grupos_mag, key=len) if grupos_mag else []

    if len(grupo_dom) < min_grupo:
        for k in range(len(mags)):
            log.append({'t': t_s[k], 'magnitud': mags[k], 'accion': 'sin evidencia suficiente, no se toca'})
        if verbose:
            import warnings as _w
            _w.warn(f"fases(): {len(mags)} salto(s) abrupto(s) detectado(s) pero sin mayoria clara; no se corrige nada.")
        return phi_corr, log

    signos_grupo = np.sign(mags[grupo_dom])
    signo_mayoritario = 1 if np.sum(signos_grupo > 0) >= np.sum(signos_grupo < 0) else -1
    s_dom = np.mean(abs_mags[grupo_dom]) * signo_mayoritario

    for k, (i0, i1) in enumerate(bordes):
        if k in grupo_dom and np.sign(mags[k]) != signo_mayoritario:
            delta = s_dom - mags[k]
            phi_corr[i1:] += delta
            log.append({'t': t_s[k], 'magnitud': mags[k], 'accion': 'CORREGIDO', 'magnitud_corregida': s_dom})
            if verbose:
                import warnings as _w
                _w.warn(f"fases(): salto con signo invertido detectado en t~{t_s[k]:.4g} "
                        f"(magnitud observada {mags[k]:.4g} -> corregida a {s_dom:.4g}).")
        else:
            log.append({'t': t_s[k], 'magnitud': mags[k], 'accion': 'sin cambios'})

    return phi_corr, log


def fases(sol, open_system: bool, corregir_saltos=False, Omega=None, margen_salto=5.0, tol_rel_salto=0.15):
    """params:
    -sol: solucion numerica de la evolucion temporal. Puede ser un Solver o un ndarray con las soluciones.
    -N_c 
    -corregir_saltos: si True, aplica una correccion LOCAL (no resimula nada)
        de saltos abruptos con signo invertido por error de redondeo de
        np.angle/np.unwrap cerca del corte de rama. Detecta los saltos via
        un umbral absoluto |dphi/dt| > margen_salto*omega (requiere pasar
        omega), y corrige solo si una mayoria clara de saltos similares
        respalda el signo correcto. Default False: preserva exactamente el
        comportamiento historico de esta funcion si no se pide explicitamente.
        ADVERTENCIA: esta correccion es conservadora pero local; para casos
        donde se sospeche un bug y se quiera la verificacion mas fuerte por
        continuidad en un parametro de control, usar
        verificar_signo_por_continuidad() por separado.
    -Omega: gap espectral maximo del sistema (escala fisica para el umbral
        de deteccion de saltos). Obligatorio si corregir_saltos=True.
    -margen_salto: margen multiplicativo del umbral (ver _detectar_saltos_absoluto).
    -tol_rel_salto: tolerancia relativa para agrupar magnitudes de salto similares.
    RETURNS
    if open_system is True
    -fg_pan: Array de longitud len(t) donde con la FG de Pancho acumulada tiempo a tiempo
    -arg: no deberia funcionar bien esto me parece, pero seria el primer termino de la fg
    -ordered_eigenvals
    -ordered_eigenvecs
    if open_system is False
    -fg_pan
    -arg
    Si corregir_saltos=True, se agrega un valor de retorno adicional al final:
    -log_saltos: lista de diccionarios con el diagnostico de cada salto
        detectado y la accion tomada (para poder auditar antes de confiar
        en la curva corregida)."""
    if corregir_saltos and Omega is None:
        raise ValueError("fases(): corregir_saltos=True requiere pasar Omega (escala fisica del sistema).")

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

                # Fijar el gauge U(1) del autovector dominante para que el overlap con
                # el paso anterior tenga parte real positiva.  Esto elimina los saltos
                # espurios de exactamente pi que LAPACK introduce al devolver -psi en
                # lugar de psi (o cualquier fase discreta) en cada diagonalizacion.
                # Solo se aplica al autovector dominante (i_1==0), unico que entra en
                # el calculo de Pan; los demas no se tocan para no introducir cambios
                # innecesarios en eigenvals_t/eigenvecs_t.
                if i_1 == 0:
                    raw_overlap = psi.overlap(eigenvecs_t[0][t_i-1])
                    if raw_overlap.real < 0:
                        psi = -psi

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
        # pan_local=[]
        argumento = np.zeros(len_t)
        psi0=evec0[0]
        psi_old=psi0
        # print('type rho0',rho0.type)
        for t_j in range(len(eigenvecs_t[0])):
            # print('i',i)
            psi=eigenvecs_t[0][t_j]
            angulo_local=np.angle(psi.overlap(psi_old))
            # pan_local.append(angulo_local)
            pan += angulo_local
            arg_global=np.angle(psi.overlap(psi0))
  
            Pan.append(pan - arg_global)
            psi_old = eigenvecs_t[0][t_j]
            # Almaceno el argumento para cada tiempo
            argumento[t_j] = arg_global

        Pan = np.array(Pan)
        # print(len(eigenvals_t))
        # print(len(eigenvecs_t))
        # print(len(eigenvecs_t[0]))

        if corregir_saltos:
            t_arr = getattr(sol, 'times', None)
            if t_arr is None:
                raise ValueError("fases(): corregir_saltos=True requiere que 'sol' tenga "
                                  "el atributo 'times' (resultado de mesolve); no disponible aqui.")
            Pan_corr, log_saltos = _corregir_signos_saltos(
                np.asarray(t_arr), Pan, Omega, margen=margen_salto, tol_rel=tol_rel_salto, verbose=True)
            return Pan_corr, argumento, eigenvals_t, eigenvecs_t, log_saltos

        return Pan, argumento, eigenvals_t, eigenvecs_t
    
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
            arg_global=np.angle(psi.overlap(psi0))
            Pan.append(pan - arg_global)
            psi_old = sol.states[i]
            # Almaceno el argumento para cada tiempo
            argumento[i] = arg_global
    
    Pan = np.array(Pan)

    if corregir_saltos:
        t_arr = getattr(sol, 'times', None)
        if t_arr is None:
            raise ValueError("fases(): corregir_saltos=True requiere que 'sol' tenga "
                              "el atributo 'times' (resultado de mesolve); no disponible aqui.")
        Pan_corr, log_saltos = _corregir_signos_saltos(
            np.asarray(t_arr), Pan, Omega, margen=margen_salto, tol_rel=tol_rel_salto, verbose=True)
        return Pan_corr, argumento, log_saltos

    return Pan, argumento


def verificar_signo_por_continuidad(H, initial_state_fn, theta, t_salto_sospechoso, T, Ncyc, spc, Omega,
                                     mesolve_fn, epsilon=None, margen_salto=5.0,
                                     tol_rel_salto=0.15, ventana_busqueda=None, verbose=True):
    """
    Verificacion MAS FUERTE (pero mas cara) que la correccion local de
    fases(corregir_saltos=True): en vez de inferir el signo correcto de un
    salto solo a partir de la propia curva en theta, resimula la evolucion
    UNITARIA (sin disipacion: c_ops vacios, mucho mas barata que el caso
    disipativo) en theta+epsilon y theta-epsilon, y compara el SIGNO
    OBSERVADO (sin corregir) del salto mas cercano a 't_salto_sospechoso' en
    cada una de las dos curvas vecinas.

    Por que se compara un salto especifico por tiempo, y no un conteo
    global: comparar el numero total de saltos "corregidos" en toda la
    ventana [0, Ncyc*T] resulto fragil (el numero TOTAL de cruces que entran
    en una ventana de tiempo fija puede diferir en uno simplemente por el
    desfasaje acumulado a lo largo de muchos ciclos, un efecto de borde de
    muestreo sin relacion con el bug de redondeo). Comparar el salto puntual
    mas cercano al sospechoso evita ese problema.

    Logica: un salto cuyo signo depende de un error de redondeo de
    np.angle/np.unwrap cerca del corte de rama deberia "resolverse" de forma
    estable apenas se perturba theta levemente (la perturbacion aleja la
    trayectoria del punto exacto de ambiguedad numerica). Si el salto mas
    cercano a t_salto_sospechoso tiene el MISMO signo en theta+epsilon y en
    theta-epsilon, eso es evidencia, independiente de la heuristica de
    mayoria de fases(corregir_saltos=True), de cual es el signo correcto en
    theta. Si en cambio theta+epsilon y theta-epsilon NO coinciden entre si,
    esto NO es un simple artefacto de redondeo aislado: podria ser una
    bifurcacion/degeneracion genuina de la trayectoria geodesica en funcion
    de theta, y se reporta como tal en vez de forzar una correccion.

    IMPORTANTE: esta funcion no inventa la fisica del problema; necesita que
    quien la llama le provea como construir el Hamiltoniano y el estado
    inicial para evaluarlos en theta+-epsilon. Por eso vive separada de
    fases(), que es agnostica a la parametrizacion fisica del problema.

    Params:
      H                   : Qobj, el Hamiltoniano (mismo para theta+-epsilon).
      initial_state_fn    : funcion theta -> Qobj (estado inicial).
      theta               : valor de theta donde se sospecha el bug (centro).
      t_salto_sospechoso  : tiempo aproximado del salto que se quiere
                             verificar (tipicamente, uno de los 't' devueltos
                             por el log de fases(corregir_saltos=True) en
                             theta, marcado como 'CORREGIDO').
      T, Ncyc, spc        : igual que en delta_phi_curve del script de barrido.
      Omega               : gap espectral maximo (escala fisica del umbral).
      mesolve_fn          : la funcion mesolve de qutip (se pasa explicitamente).
      epsilon             : paso de la perturbacion en theta. Si None, 1e-3.
      margen_salto, tol_rel_salto: pasados a la deteccion de saltos.
      ventana_busqueda    : maxima distancia temporal para considerar que un
                             salto detectado en el vecino "es el mismo" que
                             el sospechoso original. Si None, se usa T/4.
      verbose             : si True, imprime el resultado de la comparacion.

    ADVERTENCIA sobre epsilon: no hay un valor universalmente seguro, probar
    2-3 valores decrecientes y confirmar que el resultado no cambia antes de
    confiar en una sola corrida.

    Devuelve un diccionario con:
      'coincide'      : True/False/None. None si no se encontro un salto
                         cercano a t_salto_sospechoso en alguno de los dos
                         vecinos (dentro de ventana_busqueda) -> resultado
                         AMBIGUO, no concluyente.
      'signo_mas'     : signo (+1/-1) del salto encontrado en theta+epsilon,
                         o None si no se encontro ninguno cercano.
      'signo_menos'   : idem para theta-epsilon.
      'epsilon_usado' : el valor de epsilon efectivamente usado.
    """
    if epsilon is None:
        epsilon = 1e-3
    if ventana_busqueda is None:
        ventana_busqueda = T / 4.0

    t_arr = np.linspace(0, Ncyc * T, Ncyc * spc)

    def salto_mas_cercano(t_s_list, mags, t_objetivo, ventana):
        """De la lista de tiempos/magnitudes de saltos detectados, devuelve
        el signo del que esta mas cerca de t_objetivo, si cae dentro de
        'ventana'; si no hay ninguno dentro de la ventana, devuelve None."""
        if len(t_s_list) == 0:
            return None
        t_s_arr = np.asarray(t_s_list)
        idx_cercano = np.argmin(np.abs(t_s_arr - t_objetivo))
        if np.abs(t_s_arr[idx_cercano] - t_objetivo) > ventana:
            return None
        return np.sign(mags[idx_cercano])

    signos = {}
    for signo_eps, etiqueta in [(+1, 'mas'), (-1, 'menos')]:
        theta_pert = theta + signo_eps * epsilon
        psi0 = initial_state_fn(theta_pert)
        if psi0 is None:
            if verbose:
                print(f"  [verificar_signo_por_continuidad] theta{'+' if signo_eps > 0 else '-'}epsilon "
                      f"={theta_pert:.6g} cae fuera del simplex fisico; no se puede evaluar ahi.")
            signos[etiqueta] = None
            continue
        sol = mesolve_fn(H, psi0, t_arr, c_ops=[])
        # Usamos fases() SIN corregir_saltos aca: queremos el signo OBSERVADO
        # crudo de cada vecino, no su version ya corregida por la heuristica
        # de mayoria (que es justamente lo que estamos tratando de validar).
        fu_pert, _ = fases(sol, open_system=False)
        t_s_list, mags, _, _ = _detectar_saltos_absoluto(t_arr, fu_pert, Omega, margen=margen_salto)
        signos[etiqueta] = salto_mas_cercano(t_s_list, mags, t_salto_sospechoso, ventana_busqueda)

    signo_mas = signos.get('mas')
    signo_menos = signos.get('menos')

    if signo_mas is None or signo_menos is None:
        if verbose:
            print(f"  [verificar_signo_por_continuidad] no se encontro un salto cercano a "
                  f"t={t_salto_sospechoso:.4g} (dentro de +-{ventana_busqueda:.4g}) en "
                  f"alguno de los dos vecinos; resultado AMBIGUO, no se concluye nada.")
        return {'coincide': None, 'signo_mas': signo_mas, 'signo_menos': signo_menos,
                'epsilon_usado': epsilon}

    coincide = (signo_mas == signo_menos)
    if verbose:
        print(f"  [verificar_signo_por_continuidad] theta={theta:.6g}, t_sospechoso={t_salto_sospechoso:.4g}, "
              f"epsilon={epsilon:.3g}")
        print(f"    signo del salto en theta+epsilon: {signo_mas:+.0f}")
        print(f"    signo del salto en theta-epsilon: {signo_menos:+.0f}")
        print(f"    {'CONSISTENTE: ambos vecinos dan el mismo signo (evidencia de bug de redondeo, no de fisica).' if coincide else 'INCONSISTENTE: los vecinos dan signos distintos; posible bifurcacion/degeneracion genuina, no solo redondeo. Revisar a mano.'}")

    return {'coincide': coincide, 'signo_mas': signo_mas, 'signo_menos': signo_menos,
            'epsilon_usado': epsilon}