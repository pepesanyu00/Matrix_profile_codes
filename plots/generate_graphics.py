import os
import sys
import argparse
import matplotlib.pyplot as plt


# Take the arguments
parser = argparse.ArgumentParser(description="Argumentos para la generación de las gráficas")
parser.add_argument('type', type=str, choices=['vlen','threads','no_hor'], help='Tipo de gráfica: vlen="vector length en eje X", threads:"número de threads en eje X", no_hor="comparativa con/sin reducción horizontal"')
parser.add_argument('--stats_dir', required=False, type=str, default='stats', help="directorio con las estadísticas. Estructura por dentro: stats1,stats2,stats3...")
parser.add_argument('--no_vect', action='store_true', default=False, help='Incluir líneas no vectorizadas en las gráficas.')
parser.add_argument('algorithm', type=str, choices=['scamp','scrimp'], help='algoritmo a generar las gráficas.')
parser.add_argument('--average', action='store_true', default=False, help='Agrupar y promediar todas las carpetas de estadísticas en unas únicas gráficas.')
args = parser.parse_args()
suffix = '_no_vect' if args.no_vect else ''


# Estructura de las carpetas de estadísticas:
# Secuenciales:
# arq-sec |
#         |-- scamp |
#         .         |-- human_act |
#         .         .             |-- 1th
#         .         .             .
# Vectorizadas:
# arq-vect |
#          |-- scamp |
#          .         |-- human_act |
#          .         .             |-- vlen-256 |
#          .         .             .            |-- 1threads
#          .         .             .            .

arqs = ['riscv', 'vlen']
riscv_vlen = [256,512,1024,2048,4096,8192,16384]
#riscv_vlen = [256,512,1024,2048]
arm_vlen = [256,512,1024,2048]
no_hor_vlen = [256,512,1024,2048]
threads = [1,2,4,8,16,32,64]
timeseries = ['power-MPIII-SVF_n180000',
    'seismology-MPIII-SVE_n180000',
    'e0103_n180000',
    'penguin_sample_TutorialMPweb',
    #'audio-MPIII-SVD',
    #'human_activity-MPIII-SVC'
]

ts_display_names = {
    'power-MPIII-SVF_n180000': 'power',
    'seismology-MPIII-SVE_n180000': 'seismology',
    'e0103_n180000': 'ecg',
    'penguin_sample_TutorialMPweb': 'penguin',
    #'audio-MPIII-SVD': 'audio',
    #'human_activity-MPIII-SVC': 'human activity'
}


# Función que imprime un diccionario de forma apropiada.
def print_dict_pretty(dictionary, indent=4, method='json'):
    """
    Print a dictionary in a readable format.
    
    Args:
        dictionary: The dictionary to print
        indent: Number of spaces for indentation (default: 4)
        method: Printing method ('json', 'pprint', or 'custom')
    """
    if not dictionary:
        print("{}")
        return
    
    if method == 'json':
        import json
        print(json.dumps(dictionary, indent=indent, sort_keys=True))
    
    elif method == 'pprint':
        import pprint
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(dictionary)
    
    elif method == 'custom':
        def _print_dict(d, level=0):
            if not isinstance(d, dict):
                print(d)
                return
                
            print("{")
            for i, (key, value) in enumerate(d.items()):
                end_char = "" if i == len(d) - 1 else ","
                
                if isinstance(value, dict):
                    print(f"{' ' * (level+indent)}{repr(key)}: ", end="")
                    _print_dict(value, level + indent)
                    print(f"{end_char}")
                else:
                    print(f"{' ' * (level+indent)}{repr(key)}: {repr(value)}{end_char}")
            print(f"{' ' * level}}}", end="")
        
        _print_dict(dictionary)
        print()
    
    else:
        print("Invalid method. Choose 'json', 'pprint', or 'custom'")


# Se inicia la búsqueda de datos en las carpetas
if not os.path.exists(args.stats_dir):
    print(f"El directorio {args.stats_dir} no existe, por favor, introduzca uno válido.")
    sys.exit(1)

subdirs = [d for d in os.listdir(args.stats_dir) if os.path.isdir(os.path.join(args.stats_dir, d))]
subdirs.sort()

if not subdirs:
    print(f"No se encontraron subdirectorios en {args.stats_dir}.")
    sys.exit(1)

if args.average:
    groups = [('average', subdirs)]
else:
    groups = [(d, [d]) for d in subdirs]

def collect_raw_data(target_dirs):
    """Recorre las carpetas de estadísticas y devuelve los tiempos medios (simSeconds) por arquitectura/modo."""
    arm_no_vect = {}
    arm_vect = {}
    arm_vect_no_hor = {}
    riscv_no_vect = {}
    riscv_vect = {}
    riscv_vect_no_hor = {}

    for stat_dir in target_dirs:
        stats_dir = os.path.join(args.stats_dir, stat_dir)
        if not os.path.exists(stats_dir):
            continue
        for arq_mode in os.listdir(stats_dir):
            if '-' not in arq_mode:
                continue
            arq = arq_mode.split('-')[0]
            is_no_hor = ('no_hor' in arq_mode or 'no_horizontal' in arq_mode)
            is_sec = ('-sec' in arq_mode)
            is_vect = ('-vect' in arq_mode and not is_no_hor)

            alg_dir = os.path.join(stats_dir, arq_mode, args.algorithm)
            if not os.path.exists(alg_dir):
                continue
            # Directorio con las TimeSeries
            for ts in os.listdir(alg_dir):
                ts_dir = os.path.join(alg_dir, ts)
                if not os.path.isdir(ts_dir):
                    continue
                if is_sec:
                    # Directorio con los threads
                    for thread in os.listdir(ts_dir):
                        thread_number = thread.split('_')[0]
                        # Archivo de estadísticas
                        stats_file = os.path.join(ts_dir, thread, 'stats.txt')
                        if not os.path.exists(stats_file):
                            continue
                        with open(stats_file, 'r') as file:
                            for line in file:
                                if line.startswith('simSeconds'):
                                    sim_seconds = float(line.split()[1])
                                    if (arq == 'arm'):
                                        arm_no_vect.setdefault(ts, {}).setdefault(thread_number, []).append(sim_seconds)
                                        break
                                    elif (arq == 'riscv'):
                                        riscv_no_vect.setdefault(ts, {}).setdefault(thread_number, []).append(sim_seconds)
                                        break
                elif is_no_hor:
                    for vlen in os.listdir(ts_dir):
                        if '-' not in vlen:
                            continue
                        vlen_number = vlen.split('-')[1]
                        vlen_dir = os.path.join(ts_dir, vlen)
                        if not os.path.isdir(vlen_dir):
                            continue
                        for thread in os.listdir(vlen_dir):
                            thread_number = thread.split('_')[0]
                            stats_file = os.path.join(vlen_dir, thread, 'stats.txt')
                            if not os.path.exists(stats_file):
                                continue
                            with open(stats_file, 'r') as file:
                                for line in file:
                                    if line.startswith('simSeconds'):
                                        sim_seconds = float(line.split()[1])
                                        if (arq == 'arm'):
                                            arm_vect_no_hor.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)
                                            break
                                        elif (arq == 'riscv'):
                                            riscv_vect_no_hor.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)
                                            break
                elif is_vect:
                    for vlen in os.listdir(ts_dir):
                        if '-' not in vlen:
                            continue
                        vlen_number = vlen.split('-')[1]
                        vlen_dir = os.path.join(ts_dir, vlen)
                        if not os.path.isdir(vlen_dir):
                            continue
                        for thread in os.listdir(vlen_dir):
                            thread_number = thread.split('_')[0]
                            stats_file = os.path.join(vlen_dir, thread, 'stats.txt')
                            if not os.path.exists(stats_file):
                                continue
                            with open(stats_file, 'r') as file:
                                for line in file:
                                    if line.startswith('simSeconds'):
                                        sim_seconds = float(line.split()[1])
                                        if (arq == 'arm'):
                                            arm_vect.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)
                                            break
                                        elif (arq == 'riscv'):
                                            riscv_vect.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)
                                            break

    # Calculamos la media de los tiempos (puede haber varias repeticiones por combinación)
    for data in (riscv_no_vect, arm_no_vect):
        for ts in data:
            for key in data[ts]:
                valores = data[ts][key]
                if valores:
                    data[ts][key] = [sum(valores) / len(valores)]

    for data in (riscv_vect, arm_vect, riscv_vect_no_hor, arm_vect_no_hor):
        for ts in data:
            for vlen_number in data[ts]:
                for thread_number in data[ts][vlen_number]:
                    valores = data[ts][vlen_number][thread_number]
                    if valores:
                        data[ts][vlen_number][thread_number] = [sum(valores) / len(valores)]

    return arm_no_vect, arm_vect, riscv_no_vect, riscv_vect, arm_vect_no_hor, riscv_vect_no_hor


def compute_no_hor_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect, arm_vect_no_hor, riscv_vect_no_hor):
    speedup_riscv = {}
    speedup_riscv_no_hor = {}
    speedup_arm = {}
    speedup_arm_no_hor = {}

    for ts in timeseries:
        riscv_no_vect_time = riscv_no_vect[ts]["1"][0] if ts in riscv_no_vect and "1" in riscv_no_vect[ts] else None
        arm_no_vect_time = arm_no_vect[ts]["1"][0] if ts in arm_no_vect and "1" in arm_no_vect[ts] else None

        for vlen in no_hor_vlen:
            vlen_str = str(vlen)
            if (riscv_no_vect_time is not None and
                    ts in riscv_vect and
                    vlen_str in riscv_vect[ts] and
                    "1" in riscv_vect[ts][vlen_str]):
                speedup_riscv.setdefault(ts, {})[vlen] = riscv_no_vect_time / riscv_vect[ts][vlen_str]["1"][0]

            if (riscv_no_vect_time is not None and
                    ts in riscv_vect_no_hor and
                    vlen_str in riscv_vect_no_hor[ts] and
                    "1" in riscv_vect_no_hor[ts][vlen_str]):
                speedup_riscv_no_hor.setdefault(ts, {})[vlen] = riscv_no_vect_time / riscv_vect_no_hor[ts][vlen_str]["1"][0]

            if (arm_no_vect_time is not None and
                    ts in arm_vect and
                    vlen_str in arm_vect[ts] and
                    "1" in arm_vect[ts][vlen_str]):
                speedup_arm.setdefault(ts, {})[vlen] = arm_no_vect_time / arm_vect[ts][vlen_str]["1"][0]

            if (arm_no_vect_time is not None and
                    ts in arm_vect_no_hor and
                    vlen_str in arm_vect_no_hor[ts] and
                    "1" in arm_vect_no_hor[ts][vlen_str]):
                speedup_arm_no_hor.setdefault(ts, {})[vlen] = arm_no_vect_time / arm_vect_no_hor[ts][vlen_str]["1"][0]

    return speedup_riscv, speedup_riscv_no_hor, speedup_arm, speedup_arm_no_hor


def compute_vlen_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect):
    riscv_no_vect_time = {}
    arm_no_vect_time = {}
    speedup_riscv = {}
    speedup_arm = {}
    for ts in timeseries:
        riscv_no_vect_time[ts] = riscv_no_vect[ts]["1"][0] if ts in riscv_no_vect and "1" in riscv_no_vect[ts] else None
        arm_no_vect_time[ts] = arm_no_vect[ts]["1"][0] if ts in arm_no_vect and "1" in arm_no_vect[ts] else None

        for idx, vlen in enumerate(riscv_vlen):
            vlen_str = str(vlen)
            if (riscv_no_vect_time[ts] is not None and
                    ts in riscv_vect and
                    vlen_str in riscv_vect[ts] and
                    "1" in riscv_vect[ts][vlen_str]):
                speedup_riscv.setdefault(ts, {})[vlen] = riscv_no_vect_time[ts] / riscv_vect[ts][vlen_str]["1"][0]

            if idx < len(arm_vlen) and vlen == arm_vlen[idx]:
                if (arm_no_vect_time[ts] is not None and
                        ts in arm_vect and
                        vlen_str in arm_vect[ts] and
                        "1" in arm_vect[ts][vlen_str]):
                    speedup_arm.setdefault(ts, {})[vlen] = arm_no_vect_time[ts] / arm_vect[ts][vlen_str]["1"][0]

    return speedup_riscv, speedup_arm


def compute_threads_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect):
    speedup_riscv_threads = {}
    speedup_riscv_16k = {}
    speedup_arm_threads = {}
    speedup_riscv_no_vect = {}
    speedup_arm_no_vect = {}

    vlen_fixed = "2048"
    vlen_16k = "16384"

    for ts in timeseries:
        riscv_no_vect_time = riscv_no_vect[ts]["1"][0] if ts in riscv_no_vect and "1" in riscv_no_vect[ts] else None
        arm_no_vect_time = arm_no_vect[ts]["1"][0] if ts in arm_no_vect and "1" in arm_no_vect[ts] else None

        if riscv_no_vect_time is not None:
            if ts in riscv_vect and vlen_fixed in riscv_vect[ts]:
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_vect[ts][vlen_fixed] and riscv_vect[ts][vlen_fixed][thread_str]:
                        speedup_riscv_threads.setdefault(ts, {})[thread] = riscv_no_vect_time / riscv_vect[ts][vlen_fixed][thread_str][0]

            if ts in riscv_vect and vlen_16k in riscv_vect[ts]:
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_vect[ts][vlen_16k] and riscv_vect[ts][vlen_16k][thread_str]:
                        speedup_riscv_16k.setdefault(ts, {})[thread] = riscv_no_vect_time / riscv_vect[ts][vlen_16k][thread_str][0]

            if ts in riscv_no_vect:
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_no_vect[ts] and riscv_no_vect[ts][thread_str]:
                        speedup_riscv_no_vect.setdefault(ts, {})[thread] = riscv_no_vect_time / riscv_no_vect[ts][thread_str][0]

        if arm_no_vect_time is not None:
            if ts in arm_vect and vlen_fixed in arm_vect[ts]:
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in arm_vect[ts][vlen_fixed] and arm_vect[ts][vlen_fixed][thread_str]:
                        speedup_arm_threads.setdefault(ts, {})[thread] = arm_no_vect_time / arm_vect[ts][vlen_fixed][thread_str][0]

            if ts in arm_no_vect:
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in arm_no_vect[ts] and arm_no_vect[ts][thread_str]:
                        speedup_arm_no_vect.setdefault(ts, {})[thread] = arm_no_vect_time / arm_no_vect[ts][thread_str][0]

    return speedup_riscv_threads, speedup_riscv_16k, speedup_arm_threads, speedup_riscv_no_vect, speedup_arm_no_vect


# ------------------------------------------------------------------------------------------------------------
# Fase 1: recolectamos y calculamos los speedups de cada carpeta (label) SIN dibujar todavía. Esto permite fijar
# después un eje Y común entre stats10/stats25/stats_100 para comparar visualmente la tendencia real.
# ------------------------------------------------------------------------------------------------------------
label_data = {}
for label, target_dirs in groups:
    arm_no_vect, arm_vect, riscv_no_vect, riscv_vect, arm_vect_no_hor, riscv_vect_no_hor = collect_raw_data(target_dirs)
    if args.type == 'vlen':
        speedup_riscv, speedup_arm = compute_vlen_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect)
        label_data[label] = {'speedup_riscv': speedup_riscv, 'speedup_arm': speedup_arm}
    elif args.type == 'no_hor':
        s_r, s_r_nh, s_a, s_a_nh = compute_no_hor_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect, arm_vect_no_hor, riscv_vect_no_hor)
        label_data[label] = {
            'speedup_riscv': s_r,
            'speedup_riscv_no_hor': s_r_nh,
            'speedup_arm': s_a,
            'speedup_arm_no_hor': s_a_nh
        }
    else:
        s_riscv_threads, s_riscv_16k, s_arm_threads, s_riscv_no_vect, s_arm_no_vect = compute_threads_speedups(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect)
        label_data[label] = {
            'speedup_riscv_threads': s_riscv_threads,
            'speedup_riscv_16k': s_riscv_16k,
            'speedup_arm_threads': s_arm_threads,
            'speedup_riscv_no_vect': s_riscv_no_vect,
            'speedup_arm_no_vect': s_arm_no_vect,
        }

# Series relevantes según el tipo de gráfica y los flags activos
if args.type == 'vlen':
    series_keys = ['speedup_riscv', 'speedup_arm']
elif args.type == 'no_hor':
    series_keys = ['speedup_riscv', 'speedup_riscv_no_hor', 'speedup_arm', 'speedup_arm_no_hor']
else:
    series_keys = ['speedup_riscv_threads', 'speedup_riscv_16k', 'speedup_arm_threads']
    if args.no_vect:
        series_keys += ['speedup_riscv_no_vect', 'speedup_arm_no_vect']

# Límites de eje Y comunes por serie temporal, combinando los valores de TODAS las carpetas (10%/25%/100%)
y_limits = {}
for ts in timeseries:
    values = []
    for data in label_data.values():
        for key in series_keys:
            series = data.get(key, {})
            if ts in series:
                values.extend(series[ts].values())
    if values:
        y_limits[ts] = (min(values) / 1.15, max(values) * 1.15)


# ------------------------------------------------------------------------------------------------------------
# Fase 2: generamos las gráficas por carpeta, usando el mismo eje Y y las mismas posiciones de eje X
# ------------------------------------------------------------------------------------------------------------
for label, target_dirs in groups:
    data = label_data[label]

    # Carpetas de destino
    if args.average:
        output_dir = 'plots'
    else:
        output_dir = os.path.join('plots', label)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.average:
        print(f"\n--- Procesando directorio: {label} ---")
    else:
        print(f"\n--- Procesando promedio de todas las carpetas ---")

    #-------------------------- Generamos gráficas con VLEN en el eje X ----------------------------------#

    if args.type == 'vlen':
        speedup_riscv = data['speedup_riscv']
        speedup_arm = data['speedup_arm']

        # Generate plots for each timeseries
        for ts in timeseries:
            has_riscv = ts in speedup_riscv and speedup_riscv[ts]
            has_arm = ts in speedup_arm and speedup_arm[ts]
            if not (has_riscv or has_arm):
                print(f"Skipping plot for {ts}: No data available.")
                continue

            plt.figure(figsize=(10, 6))

            if has_riscv:
                riscv_present = [v for v in riscv_vlen if v in speedup_riscv[ts]]
                riscv_positions = [riscv_vlen.index(v) for v in riscv_present]
                riscv_y = [speedup_riscv[ts][v] for v in riscv_present]
                plt.plot(riscv_positions, riscv_y, 'b-o', linewidth=2, label='RISC-V')

            if has_arm:
                arm_present = [v for v in riscv_vlen if v in speedup_arm[ts]]
                arm_positions = [riscv_vlen.index(v) for v in arm_present]
                arm_y = [speedup_arm[ts][v] for v in arm_present]
                plt.plot(arm_positions, arm_y, 'r-s', linewidth=2, label='ARM')

            # Eje X fijo (misma rejilla en las 3 carpetas, aunque falte algún punto)
            plt.xticks(range(len(riscv_vlen)), [str(v) for v in riscv_vlen])

            # Add labels, title, and legend
            plt.xlabel('Vector Length (VLEN)', fontsize=34)
            plt.ylabel('Speedup', fontsize=34)
            plt.title(ts_display_names.get(ts, ts), fontsize=36)
            plt.yscale('log', base=10)
            plt.tick_params(axis='both', which='major', labelsize=26)
            plt.tick_params(axis='y', which='minor', labelsize=22)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=25, frameon=False, labelspacing=0.1, loc='upper left')
            if ts in y_limits:
                plt.ylim(*y_limits[ts])

            # Improve appearance
            plt.tight_layout()
            
            # Save plot as PDF and JPG
            save_path_pdf = os.path.join(output_dir, f'{ts}_{args.algorithm}_vlen{suffix}.pdf')
            save_path_jpg = os.path.join(output_dir, f'{ts}_{args.algorithm}_vlen{suffix}.jpg')
            plt.savefig(save_path_pdf, dpi=300)
            plt.savefig(save_path_jpg, dpi=300)
            plt.close()

        print(f"Plots saved in the '{output_dir}' directory")

    #-------------------------- Generamos gráficas NO_HOR en el eje X ----------------------------------#

    elif args.type == 'no_hor':
        speedup_riscv = data['speedup_riscv']
        speedup_riscv_no_hor = data['speedup_riscv_no_hor']
        speedup_arm = data['speedup_arm']
        speedup_arm_no_hor = data['speedup_arm_no_hor']

        for ts in timeseries:
            has_data = any(
                ts in data.get(k, {}) and data[k][ts]
                for k in ['speedup_riscv', 'speedup_riscv_no_hor', 'speedup_arm', 'speedup_arm_no_hor']
            )
            if not has_data:
                print(f"Skipping plot for {ts}: No data available.")
                continue

            plt.figure(figsize=(10, 6))

            if ts in speedup_riscv and speedup_riscv[ts]:
                present = [v for v in no_hor_vlen if v in speedup_riscv[ts]]
                positions = [no_hor_vlen.index(v) for v in present]
                values = [speedup_riscv[ts][v] for v in present]
                plt.plot(positions, values, 'b-o', linewidth=2, label='RVV')

            if ts in speedup_riscv_no_hor and speedup_riscv_no_hor[ts]:
                present = [v for v in no_hor_vlen if v in speedup_riscv_no_hor[ts]]
                positions = [no_hor_vlen.index(v) for v in present]
                values = [speedup_riscv_no_hor[ts][v] for v in present]
                plt.plot(positions, values, 'b--^', linewidth=2, label='RVV No-Hor')

            if ts in speedup_arm and speedup_arm[ts]:
                present = [v for v in no_hor_vlen if v in speedup_arm[ts]]
                positions = [no_hor_vlen.index(v) for v in present]
                values = [speedup_arm[ts][v] for v in present]
                plt.plot(positions, values, 'r-s', linewidth=2, label='SVE')

            if ts in speedup_arm_no_hor and speedup_arm_no_hor[ts]:
                present = [v for v in no_hor_vlen if v in speedup_arm_no_hor[ts]]
                positions = [no_hor_vlen.index(v) for v in present]
                values = [speedup_arm_no_hor[ts][v] for v in present]
                plt.plot(positions, values, 'r--v', linewidth=2, label='SVE No-Hor')

            # Eje X con las longitudes vectoriales fijas (256, 512, 1024, 2048)
            plt.xticks(range(len(no_hor_vlen)), [str(v) for v in no_hor_vlen])

            plt.xlabel('Vector Length ($vlen$)', fontsize=34)
            plt.ylabel('Speedup', fontsize=34)
            plt.title(f"{ts_display_names.get(ts, ts)} ({args.algorithm.upper()})", fontsize=36)
            plt.tick_params(axis='both', which='major', labelsize=26)
            plt.tick_params(axis='y', which='minor', labelsize=22)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=25, frameon=False, labelspacing=0.1, loc='upper left')

            if ts in y_limits:
                plt.ylim(*y_limits[ts])

            plt.tight_layout()

            # Guardar gráfica en PDF y JPG
            save_path_pdf = os.path.join(output_dir, f'{ts}_{args.algorithm}_no_horizontal{suffix}.pdf')
            save_path_jpg = os.path.join(output_dir, f'{ts}_{args.algorithm}_no_horizontal{suffix}.jpg')
            plt.savefig(save_path_pdf, dpi=300)
            plt.savefig(save_path_jpg, dpi=300)
            plt.close()

        print(f"Plots saved in the '{output_dir}' directory")

        print("\n=== NO-HORIZONTAL REDUCTION SPEEDUP ANALYSIS ===")
        for ts in timeseries:
            has_comp = (ts in speedup_riscv and ts in speedup_riscv_no_hor) or (ts in speedup_arm and ts in speedup_arm_no_hor)
            if has_comp:
                print(f"\nTimeseries: {ts}")
                for vlen in no_hor_vlen:
                    str_parts = []
                    if ts in speedup_riscv and vlen in speedup_riscv[ts]:
                        str_parts.append(f"RVV: {speedup_riscv[ts][vlen]:.2f}x")
                    if ts in speedup_riscv_no_hor and vlen in speedup_riscv_no_hor[ts]:
                        str_parts.append(f"RVV No-Hor: {speedup_riscv_no_hor[ts][vlen]:.2f}x")
                    if ts in speedup_arm and vlen in speedup_arm[ts]:
                        str_parts.append(f"SVE: {speedup_arm[ts][vlen]:.2f}x")
                    if ts in speedup_arm_no_hor and vlen in speedup_arm_no_hor[ts]:
                        str_parts.append(f"SVE No-Hor: {speedup_arm_no_hor[ts][vlen]:.2f}x")
                    if str_parts:
                        print(f"  vlen={vlen}: " + " | ".join(str_parts))

    #-------------------------- Generamos gráficas con THREADS en el eje X ----------------------------------#

    elif args.type == 'threads':
        speedup_riscv_threads = data['speedup_riscv_threads']
        speedup_riscv_16k = data['speedup_riscv_16k']
        speedup_arm_threads = data['speedup_arm_threads']
        speedup_riscv_no_vect = data['speedup_riscv_no_vect']
        speedup_arm_no_vect = data['speedup_arm_no_vect']

        for ts in timeseries:
            has_data = False
            if ts in speedup_riscv_threads and speedup_riscv_threads[ts]:
                has_data = True
            if ts in speedup_riscv_16k and speedup_riscv_16k[ts]:
                has_data = True
            if ts in speedup_arm_threads and speedup_arm_threads[ts]:
                has_data = True
            if args.no_vect:
                if ts in speedup_riscv_no_vect and speedup_riscv_no_vect[ts]:
                    has_data = True
                if ts in speedup_arm_no_vect and speedup_arm_no_vect[ts]:
                    has_data = True
                
            if not has_data:
                print(f"Skipping plot for {ts}: No data available.")
                continue
                
            plt.figure(figsize=(10, 6))

            if ts in speedup_riscv_threads and speedup_riscv_threads[ts]:
                present = [t for t in threads if t in speedup_riscv_threads[ts]]
                positions = [threads.index(t) for t in present]
                values = [speedup_riscv_threads[ts][t] for t in present]
                plt.plot(positions, values, 'b-o', linewidth=2, label='RVV_2k')

            if ts in speedup_riscv_16k and speedup_riscv_16k[ts]:
                present = [t for t in threads if t in speedup_riscv_16k[ts]]
                positions = [threads.index(t) for t in present]
                values = [speedup_riscv_16k[ts][t] for t in present]
                plt.plot(positions, values, 'y-o', linewidth=2, label='RVV_16k')

            if ts in speedup_arm_threads and speedup_arm_threads[ts]:
                present = [t for t in threads if t in speedup_arm_threads[ts]]
                positions = [threads.index(t) for t in present]
                values = [speedup_arm_threads[ts][t] for t in present]
                plt.plot(positions, values, 'r-s', linewidth=2, label='SVE_2k')

            if args.no_vect and ts in speedup_riscv_no_vect and speedup_riscv_no_vect[ts]:
                present = [t for t in threads if t in speedup_riscv_no_vect[ts]]
                positions = [threads.index(t) for t in present]
                values = [speedup_riscv_no_vect[ts][t] for t in present]
                plt.plot(positions, values, 'g--^', linewidth=2, label='RISC-V_NV')

            if args.no_vect and ts in speedup_arm_no_vect and speedup_arm_no_vect[ts]:
                present = [t for t in threads if t in speedup_arm_no_vect[ts]]
                positions = [threads.index(t) for t in present]
                values = [speedup_arm_no_vect[ts][t] for t in present]
                plt.plot(positions, values, 'orange', linestyle='--', marker='v', linewidth=2, label='ARM_NV')

            # Eje X fijo (misma rejilla en las 3 carpetas, aunque falte algún punto, ej. 64 threads)
            plt.xticks(range(len(threads)), [str(t) for t in threads])
            
            plt.xlabel('Number of Threads', fontsize=34)
            plt.ylabel('Speedup', fontsize=34)
            plt.title(ts_display_names.get(ts, ts), fontsize=36)
            #plt.yscale('log', base=10)
            plt.tick_params(axis='both', which='major', labelsize=26)
            plt.tick_params(axis='y', which='minor', labelsize=22)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=25, frameon=False, labelspacing=0.1, loc='upper left')
            #plt.legend(fontsize=20, frameon=False, labelspacing=0.1, loc='upper right', ncol=2, columnspacing=0.5, bbox_to_anchor=(1, 0.6)) #PARA PONERLO A LA DERECHA Y EN EL MEDIO
            if ts in y_limits:
                plt.ylim(*y_limits[ts])
            
            plt.tight_layout()
            
            # Save plot as PDF and JPG
            save_path_pdf = os.path.join(output_dir, f'{ts}_{args.algorithm}_threads{suffix}.pdf')
            save_path_jpg = os.path.join(output_dir, f'{ts}_{args.algorithm}_threads{suffix}.jpg')
            plt.savefig(save_path_pdf, dpi=300)
            plt.savefig(save_path_jpg, dpi=300)
            plt.close()

        print("\n=== SPEEDUP ANALYSIS ===")

        riscv_2k_improvements = []
        if speedup_riscv_threads and speedup_riscv_no_vect:
            print("\nRISC-V (VLEN=2048 vs Non-vectorized):")
            for ts in timeseries:
                if ts in speedup_riscv_threads and ts in speedup_riscv_no_vect:
                    print(f"  {ts}:")
                    for thread in threads:
                        if (thread in speedup_riscv_threads[ts] and 
                            thread in speedup_riscv_no_vect[ts]):
                            
                            vect_speedup = speedup_riscv_threads[ts][thread]
                            no_vect_speedup = speedup_riscv_no_vect[ts][thread]
                            improvement = vect_speedup / no_vect_speedup
                            riscv_2k_improvements.append(improvement)
                            print(f"    {thread} threads: {improvement:.2f}x improvement")

        riscv_16k_improvements = []
        if speedup_riscv_16k and speedup_riscv_no_vect:
            print("\nRISC-V (VLEN=16384 vs Non-vectorized):")
            for ts in timeseries:
                if ts in speedup_riscv_16k and ts in speedup_riscv_no_vect:
                    print(f"  {ts}:")
                    for thread in threads:
                        if (thread in speedup_riscv_16k[ts] and 
                            thread in speedup_riscv_no_vect[ts]):
                            
                            vect_speedup = speedup_riscv_16k[ts][thread]
                            no_vect_speedup = speedup_riscv_no_vect[ts][thread]
                            improvement = vect_speedup / no_vect_speedup
                            riscv_16k_improvements.append(improvement)
                            print(f"    {thread} threads: {improvement:.2f}x improvement")

        arm_improvements = []
        if speedup_arm_threads and speedup_arm_no_vect:
            print("\nARM (VLEN=2048 vs Non-vectorized):")
            for ts in timeseries:
                if ts in speedup_arm_threads and ts in speedup_arm_no_vect:
                    print(f"  {ts}:")
                    for thread in threads:
                        if (thread in speedup_arm_threads[ts] and 
                            thread in speedup_arm_no_vect[ts]):
                            
                            vect_speedup = speedup_arm_threads[ts][thread]
                            no_vect_speedup = speedup_arm_no_vect[ts][thread]
                            improvement = vect_speedup / no_vect_speedup
                            arm_improvements.append(improvement)
                            print(f"    {thread} threads: {improvement:.2f}x improvement")

        print("\n=== OVERALL AVERAGES ===")
        if riscv_2k_improvements:
            riscv_2k_mean = sum(riscv_2k_improvements) / len(riscv_2k_improvements)
            print(f"RISC-V average improvement (2k vs non-vect): {riscv_2k_mean:.2f}x")
            print(f"RISC-V 2k measurements count: {len(riscv_2k_improvements)}")
        else:
            print("No RISC-V 2k improvement data available")

        if riscv_16k_improvements:
            riscv_16k_mean = sum(riscv_16k_improvements) / len(riscv_16k_improvements)
            print(f"RISC-V average improvement (16k vs non-vect): {riscv_16k_mean:.2f}x")
            print(f"RISC-V 16k measurements count: {len(riscv_16k_improvements)}")
        else:
            print("No RISC-V 16k improvement data available")

        if arm_improvements:
            arm_mean = sum(arm_improvements) / len(arm_improvements)
            print(f"ARM average improvement (2k vs non-vect): {arm_mean:.2f}x")
            print(f"ARM measurements count: {len(arm_improvements)}")
        else:
            print("No ARM improvement data available")

        print(f"Thread speedup plots saved in the '{output_dir}' directory")