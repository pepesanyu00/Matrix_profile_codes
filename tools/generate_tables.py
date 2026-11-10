import os, glob, re

def parse_trace(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    functions = {}
    cur_fn = None
    cur_instrs = []
    for line in lines:
        raw = line.strip()
        m = re.match(r'^([a-zA-Z0-9_]+)(\(.*?\))?:\s*$', raw)
        if m:
            lbl = m.group(1).split('(')[0]
            if any(term in lbl for term in ['init_', 'diag_step', 'diag_loop', 'thread_reduce']):
                if cur_fn:
                    functions[cur_fn] = cur_instrs
                cur_fn = lbl
                cur_instrs = []
                continue
            elif cur_fn and not raw.startswith('.'):
                functions[cur_fn] = cur_instrs
                cur_fn = None
                cur_instrs = []
                continue
        if cur_fn:
            if line.startswith('\t') or line.startswith('    ') or line.startswith('        '):
                if raw and not raw.startswith('.') and not raw.endswith(':') and not raw.startswith('#'):
                    if raw != 'endbr64':
                        cur_instrs.append(raw)
    if cur_fn and cur_instrs:
        functions[cur_fn] = cur_instrs
    return functions

def analyze_all(compiler):
    pattern = f'outputs/trazas microbenchmarks completos/{compiler}/*.txt'
    files = sorted(glob.glob(pattern))
    data = {}
    for f in files:
        base = os.path.basename(f)
        fns = parse_trace(f)
        data[base] = fns
    return data

gcc_data = analyze_all('gcc')
clang_data = analyze_all('clang')

# Print SCAMP and SCRIMP tables
def print_summary(algo, data, comp):
    print(f"\n================ {algo.upper()} - {comp.upper()} 14.2/18.1 -O2 ================")
    print(f"{'Variante':45s} | {'MB1':>4s} | {'MB2 (estát)':>11s} | {'MB2 (dinám)':>12s} | {'MB3':>4s} | {'MB4':>4s}")
    print("-" * 92)
    # Target files for this algo
    for fname, fns in data.items():
        if algo in fname:
            mb1 = mb2_stat = mb2_dyn = mb3 = mb4 = "-"
            for fn, ins in fns.items():
                if 'builtin' in fn:
                    continue
                cnt = len(ins)
                if 'init_' in fn:
                    mb1 = str(cnt)
                elif 'diag_step' in fn:
                    mb2_stat = str(cnt)
                    # compute dyn
                    if 'avx2' in fn and 'vectorized' not in fn and comp == 'gcc':
                        # loop 4 iters, 8 body instrs (min 6, max 8)
                        base = cnt - 8
                        mb2_dyn = f"{base+4*6}-{base+4*8}"
                    elif 'avx512' in fn and 'vectorized' not in fn and comp == 'gcc':
                        # loop 8 iters, 8 body instrs (min 6, max 8)
                        base = cnt - 8
                        mb2_dyn = f"{base+8*6}-{base+8*8}"
                    elif 'vectorized' in fn:
                        mb2_dyn = f"{cnt-4}-{cnt}"
                    elif 'scalar' in fn:
                        mb2_dyn = f"{cnt-2}-{cnt}"
                    elif 'sve' in fn:
                        mb2_dyn = f"{cnt-2}-{cnt}"
                    elif 'rvv' in fn:
                        mb2_dyn = f"{cnt-3}-{cnt}"
                    else:
                        mb2_dyn = str(cnt)
                elif 'diag_loop' in fn:
                    mb3 = str(cnt)
                elif 'thread_reduce' in fn:
                    mb4 = str(cnt)
            print(f"{fname:45s} | {mb1:>4s} | {mb2_stat:>11s} | {mb2_dyn:>12s} | {mb3:>4s} | {mb4:>4s}")

print_summary('scamp', gcc_data, 'gcc')
print_summary('scrimp', gcc_data, 'gcc')
print_summary('scamp', clang_data, 'clang')
print_summary('scrimp', clang_data, 'clang')
