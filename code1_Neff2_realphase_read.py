#!/usr/bin/env python3
# =============================================================================
#  read_npz.py — Lector/inspector de los archivos .npz guardados por
#                 code1_Neff2_realphase.py (y compatibles, p.ej. code2_*).
#
#  COMO USAR ESTE SCRIPT (version para IDE, sin terminal):
#    Todo lo que tenes que tocar esta en el bloque "PARAMETROS DE ENTRADA"
#    mas abajo, justo despues de los imports. Ahi escribis:
#      - la lista de archivos .npz que querer leer (FILES)
#      - si querer generar un grafico comparativo (HACER_PLOT)
#      - si el grafico se guarda a un archivo o se muestra en pantalla (SAVE_PLOT_AS)
#      - si querer exportar los datos a .csv (EXPORTAR_CSV)
#    Despues le das "Run" al script desde el IDE, como a cualquier otro .py.
#    NO hace falta escribir nada en una terminal ni pasar argumentos.
#
#  Que hace el script (resumen):
#    1) Para cada .npz: lista las claves guardadas, muestra los parametros
#       fisicos escalares (g, gamma, p_at, Delta, Ncyc, spc) y resume los
#       arrays grandes (forma, dtype, min/max/mean, cantidad de NaNs).
#    2) Si HACER_PLOT = True: superpone metric(theta) (o M_min/M_max(theta)
#       si el archivo es del formato de code2) de TODOS los archivos
#       listados en FILES, en una sola figura. Util para comparar, por
#       ejemplo, distintos valores de Delta entre si. Si el archivo tiene
#       la columna 'fidelity' (agregada despues, = fid[:,-1] = fidelidad
#       final F(tau) entre estado unitario y pointer-state disipativo),
#       tambien se grafica, en un eje Y secundario (escala [0,1]).
#    3) Si EXPORTAR_CSV = True: escribe un .csv (mismo nombre que el .npz,
#       cambiando la extension) con las columnas theta/metric/darkw/fidelity
#       (o theta/M_min/M_max), para poder abrirlo en Excel/Sheets si se quiere.
# =============================================================================

import numpy as np   # numpy es quien sabe leer el formato .npz (np.load)

# =============================================================================
#  PARAMETROS DE ENTRADA — EDITAR ACA CADA VEZ QUE QUIERAS CORRER EL SCRIPT
# =============================================================================

# Lista de archivos .npz a leer. Podes poner uno solo o varios; si poner
# varios y pedis HACER_PLOT=True, se superponen todos en el mismo grafico.
# Importante: la ruta es relativa a la carpeta desde donde corres el script,
# o podes poner la ruta completa (absoluta) si preferis no tener dudas.
FILES = [
    "orbits/data/code1_Neff2_realphase.npz",
    "orbits/data/code1_Neff2_realphase1.0g.npz",
]

# Si es True, genera una figura superponiendo metric(theta) de todos los
# archivos de FILES. Si es False, el script solo imprime el reporte de texto
# (parametros + resumen de arrays) y no genera ninguna figura.
HACER_PLOT = True

# Que hacer con el grafico generado (solo aplica si HACER_PLOT = True):
#   - Si le poner un nombre de archivo (string), p.ej. "comparativa.png",
#     el grafico se GUARDA en disco con ese nombre y no se abre ninguna
#     ventana (esto es necesario si estas corriendo en un servidor remoto
#     o un entorno sin pantalla; en tu IDE local tambien funciona igual).
#   - Si le poner None, el script intenta ABRIR una ventana con el grafico
#     (plt.show()). Esto solo funciona si tu IDE/entorno soporta mostrar
#     ventanas graficas.
SAVE_PLOT_AS = None #"orbits/imgs/comparativa.png"   # o poner None para intentar mostrarlo en pantalla

# Si es True, ademas exporta cada .npz a un .csv (mismo nombre, extension
# distinta) con las columnas theta/metric/darkw (o theta/M_min/M_max).
EXPORTAR_CSV = False


# =============================================================================
#  A PARTIR DE ACA: CODIGO DEL SCRIPT. No hace falta tocar nada mas abajo
#  para el uso normal; las funciones estan comentadas para quien quiera
#  entender o modificar el detalle de como se procesa cada archivo.
# =============================================================================

# Estas listas son solo una referencia interna: nombres de columnas que
# ESPERAMOS encontrar adentro de los .npz, segun de que codigo vinieron.
# El script es tolerante: si una clave no esta, simplemente la ignora
# (no rompe ni tira error), por eso podes usar el mismo lector para
# archivos de code1_Neff2_realphase.py y de code2_Neff2_phaseband.py.
SCALAR_KEYS = ['g', 'gamma', 'p_at', 'Delta', 'Ncyc', 'spc']        # parametros fisicos, un numero cada uno
ARRAY_KEYS_CODE1 = ['theta', 'metric', 'darkw', 'fidelity']         # columnas tipicas de code1 (+ fidelity, agregada despues)
ARRAY_KEYS_CODE2 = ['theta', 'M_min', 'M_max', 'ph_min', 'ph_max']  # columnas tipicas de code2


def is_scalar_like(arr):
    """
    Cuando guardas un numero suelto con np.savez(g=0.01, ...), numpy lo
    guarda internamente como un array de "dimension 0" (no como un float
    de Python comun). Esta funcion detecta esos casos para que despues
    podamos convertirlos de vuelta a un numero normal con .item().

    arr.ndim == 0   -> es un array "escalar" (sin forma, un solo numero)
    arr.size == 1   -> tiene un solo elemento (por si vino como [0.01], por ejemplo)
    """
    return arr.ndim == 0 or arr.size == 1


def summarize_array(name, arr):
    """
    Imprime una linea de resumen para UN array (una de las columnas
    guardadas en el .npz, por ejemplo 'metric' o 'theta').

    Mostramos:
      - forma (shape): cuantos elementos tiene y en que dimensiones
      - dtype: el tipo de dato (float64, etc.)
      - min, max, mean: estadisticas basicas, IGNORANDO los NaN
      - n_nan: cuantos valores NaN tiene el array

    El n_nan es importante en tu caso particular: en code1_Neff2_realphase.py,
    cuando un theta cae fuera del simplex fisico (alguna poblacion p_k<0),
    initial_state() devuelve None y esa entrada de 'metric' queda como NaN.
    Entonces n_nan te dice cuantos thetas del barrido fueron descartados.
    """
    # Si el array es de texto (strings) no tiene sentido calcular min/max/mean
    if arr.dtype.kind in ('U', 'S', 'O'):
        print(f"    {name:10s} shape={arr.shape}  dtype={arr.dtype}  "
              f"(no es numerico, no calculo estadisticas)")
        return

    # np.isfinite(arr) da True donde el valor es un numero "normal"
    # (no NaN, no +-infinito). Con esto separamos los valores validos.
    finite = arr[np.isfinite(arr)] if arr.size else arr
    n_nan = int(np.sum(~np.isfinite(arr))) if arr.size else 0

    if finite.size:  # si quedo al menos un valor finito, mostramos estadisticas
        print(f"    {name:10s} shape={arr.shape}  dtype={arr.dtype}  "
              f"min={finite.min():.6g}  max={finite.max():.6g}  "
              f"mean={finite.mean():.6g}  n_nan={n_nan}")
    else:  # caso raro: todo el array es NaN/infinito
        print(f"    {name:10s} shape={arr.shape}  dtype={arr.dtype}  "
              f"(sin valores finitos)  n_nan={n_nan}")


def load_and_report(path):
    """
    Abre UN archivo .npz y muestra en pantalla:
      1) la lista de claves (nombres de columnas) que contiene
      2) los parametros fisicos escalares que reconoce (g, gamma, etc.)
      3) un resumen de cada array grande (theta, metric, darkw, ...)

    Devuelve el objeto 'data' (tipo NpzFile) para que despues lo podamos
    usar en el grafico o en la exportacion a csv, sin tener que volver
    a abrir el archivo del disco.

    Si el archivo no se puede abrir (no existe, esta corrupto, etc.),
    imprime el error y devuelve None en vez de hacer crashear el script.
    """
    print(f"\n{'=' * 78}\n{path}\n{'=' * 78}")

    try:
        # np.load no carga los arrays en memoria todavia (es "perezoso"),
        # los va leyendo a medida que los pedis con data['nombre_clave'].
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  [ERROR] no se pudo abrir el archivo: {e}")
        return None

    # data.keys() te da los nombres de TODO lo que se guardo con np.savez,
    # tanto los parametros escalares como los arrays grandes, todo junto.
    keys = list(data.keys())
    print(f"  claves guardadas en el archivo: {keys}")

    # --- mostramos los parametros fisicos (numeros sueltos) ---
    print("  --- parametros ---")
    scalar_found = False
    for k in SCALAR_KEYS:
        if k in data:  # si esa clave esta presente en este archivo en particular
            v = data[k]
            # Si es un array "escalar" (un solo numero), lo convertimos a
            # un float/int de Python normal con .item() para que se vea
            # mas limpio al imprimirlo (en vez de "array(0.01)").
            val = v.item() if hasattr(v, 'item') and is_scalar_like(v) else v
            print(f"    {k:8s} = {val}")
            scalar_found = True
    if not scalar_found:
        print("    (no se encontraron los parametros escalares esperados)")

    # --- mostramos resumen de los arrays grandes (theta, metric, etc.) ---
    print("  --- arrays ---")
    for k in keys:
        if k in SCALAR_KEYS:
            continue  # los escalares ya los mostramos arriba, no los repetimos
        arr = data[k]
        summarize_array(k, np.asarray(arr))

    return data


def make_plot(datasets, paths, save_path=None):
    """
    Recibe una LISTA de datasets ya cargados (uno por cada archivo de FILES)
    y los superpone en un solo grafico: metric(theta) si el archivo es del
    formato de code1, o la banda M_min/M_max(theta) si es del formato de code2.

    Parametros:
      datasets : lista de objetos 'data' (los que devuelve load_and_report)
      paths    : lista de nombres de archivo (para poner en la leyenda)
      save_path: si es un string, guarda el grafico en ese archivo.
                 Si es None, intenta abrir una ventana con plt.show().
    """
    import matplotlib
    # Si vamos a GUARDAR el grafico en un archivo, usamos el backend "Agg",
    # que no necesita pantalla. Si vamos a MOSTRARLO en una ventana, no lo
    # forzamos, para que matplotlib use el backend interactivo normal de tu
    # sistema/IDE.
    if save_path:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False  # para saber si al final hubo algo que graficar

    ax2 = ax.twinx()
    any_fidelity_plotted = False

    # Recorremos en paralelo los datasets cargados y sus nombres de archivo.
    # zip() empareja el elemento i de 'datasets' con el elemento i de 'paths'.
    for data, path in zip(datasets, paths):
        if data is None or 'theta' not in data:
            # data is None  -> ese archivo no se pudo abrir (ver load_and_report)
            # 'theta' not in data -> el archivo no tiene la columna que graficamos
            continue

        theta = np.asarray(data['theta'])

        # Armamos una etiqueta extra con el valor de Delta/g, si esta disponible,
        # para que la leyenda del grafico diga por ejemplo "(Δ=1g)".
        label_extra = ""
        if 'Delta' in data and 'g' in data:
            try:
                label_extra = f" (Δ={float(data['Delta']) / float(data['g']):.3g}g)"
            except Exception:
                pass  # si algo falla calculando esto, seguimos sin la etiqueta extra

        # Nos quedamos solo con el nombre del archivo (sin la carpeta) para
        # que la leyenda no quede gigante si la ruta es larga.
        short_name = path.split('/')[-1]

        if 'metric' in data:
            # Formato de code1_Neff2_realphase.py: una sola curva de robustez.
            ax.plot(theta, data['metric'], 'o-', ms=3,
                    label=f"{short_name}{label_extra}")
            any_plotted = True

        elif 'M_min' in data and 'M_max' in data:
            # Formato de code2_Neff2_phaseband.py: una BANDA entre el minimo
            # y el maximo de robustez sobre las fases relativas.
            ax.fill_between(theta, data['M_min'], data['M_max'], alpha=0.25)
            ax.plot(theta, data['M_min'], 'o-', ms=3,
                    label=f"{short_name} M_min{label_extra}")
            ax.plot(theta, data['M_max'], 's-', ms=3,
                    label=f"{short_name} M_max{label_extra}")
            any_plotted = True

        if 'fidelity' in data:
            # Curva de fidelidad final F(tau), en el eje Y secundario (ax2),
            # con linea punteada para distinguirla visualmente de 'metric'.
            ax2.plot(theta, data['fidelity'], '^--', ms=3, alpha=0.7,
                      label=f"{short_name} fidelity{label_extra}")
            any_fidelity_plotted = True

    if not any_plotted:
        # Si ningun archivo tenia las columnas esperadas, avisamos y salimos
        # sin generar una figura vacia.
        print("\n[plot] ningun archivo tenia 'metric' o 'M_min'/'M_max'; no se genera figura.")
        return

    ax.set_xlabel(r'$\theta$ (familia $N_{\rm eff}=2$)')
    ax.set_ylabel(r'robustez: $\max_{\rm cyc}|\delta\phi_{\rm base}|$')
    ax2.set_ylabel(r'fidelidad final $F(\tau=N_{\rm cyc}T)$')
    ax.axhline(0, color='k', lw=0.4)   # linea horizontal en y=0, de referencia
    ax.grid(alpha=0.3)

    # Combinamos las leyendas de ax (metric/M_min/M_max) y ax2 (fidelity)
    # en una sola caja de leyenda, en vez de tener dos leyendas superpuestas
    # que se tapan entre si.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[plot] guardado en: {save_path}")
    else:
        plt.show()  # abre una ventana interactiva (si el entorno lo permite)


def export_csv(data, path):
    """
    Escribe un archivo .csv con la columna 'theta' mas las columnas de
    interes que existan en 'data' (metric, darkw, M_min, M_max).

    El nombre del .csv se genera a partir del nombre del .npz, cambiando
    la extension. Por ejemplo: 'code1_Neff2_realphase.npz' -> 'code1_Neff2_realphase.csv'.
    """
    import csv  # libreria estandar de Python para escribir archivos .csv

    if data is None or 'theta' not in data:
        print(f"  [csv] {path}: no tiene 'theta', no exporto nada.")
        return

    theta = np.asarray(data['theta'])
    # Armamos un diccionario ordenado: nombre de columna -> array de datos.
    # Siempre incluimos 'theta'; las demas columnas solo si estan presentes.
    # 'fidelity' es la fidelidad final F(tau=Ncyc*T) entre el estado unitario
    # y el pointer-state disipativo, guardada en code1 como fid[:,-1].
    cols = {'theta': theta}
    for k in ('metric', 'darkw', 'fidelity', 'M_min', 'M_max'):
        if k in data:
            cols[k] = np.asarray(data[k])

    # rsplit('.', 1)[0] corta el nombre de archivo en el ULTIMO punto,
    # para reemplazar la extension .npz por .csv sin tocar el resto del nombre.
    out_path = path.rsplit('.', 1)[0] + '.csv'

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols.keys())          # primera fila: nombres de columna
        for row in zip(*cols.values()):        # despues, una fila por cada theta
            writer.writerow(row)

    print(f"  [csv] exportado: {out_path}")


def main():
    """
    Funcion principal: usa las variables que definiste en el bloque
    'PARAMETROS DE ENTRADA' (FILES, HACER_PLOT, SAVE_PLOT_AS, EXPORTAR_CSV)
    en vez de leer argumentos de la terminal.
    """
    datasets = []  # aca vamos a ir guardando los datos cargados de cada archivo

    # Paso 1: cargar y reportar cada archivo de la lista FILES.
    for path in FILES:
        data = load_and_report(path)
        datasets.append(data)
        if EXPORTAR_CSV:
            export_csv(data, path)

    # Paso 2: si se pidio, generar el grafico comparativo con todos los archivos.
    if HACER_PLOT:
        make_plot(datasets, FILES, save_path=SAVE_PLOT_AS)


# Este "if" es una convencion estandar de Python: el codigo adentro solo
# se ejecuta si corres ESTE archivo directamente (con "Run" en el IDE, o
# "python read_npz.py"). Si en cambio importaras este archivo desde otro
# script con "import read_npz", el bloque de abajo NO se ejecutaria solo.
if __name__ == "__main__":
    main()