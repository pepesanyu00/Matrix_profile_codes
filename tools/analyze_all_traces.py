import os, glob, re, collections

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

def analyze_mb2_cf(fn_name, instrs):
    static_len = len(instrs)
    mnems = [i.split()[0].lower() for i in instrs]
    
    # Check if loop is present
    has_loop = False
    loop_trip = 1
    loop_body_min = 0
    loop_body_max = 0
    base_instrs = static_len
    
    # AVX V0 sequential search loop detection
    # In GCC:
    #   AVX2 V0: loop with cmp rax, 4 / jne .L10 (4 iterations)
    #   AVX512 V0: loop with cmp rsi, rax / jne .L10 (8 iterations)
    # In Clang: unrolled loop (no backwards jump)
    
    # Check backwards jumps
    labels = []
    backwards_jumps = []
    # Identify labels in original code if any or instructions
    # For GCC AVX2 / AVX512 V0:
    if 'avx2' in fn_name and 'vectorized' not in fn_name:
        if any('cmp' in i and '4' in i for i in instrs):
            # Loop of 4 iterations
            # Body: vmovsd, vcomisd, jbe, (mov, vmovsd), add, cmp, jne
            # min: 6 instrs, max: 8 instrs per iter
            # Loop body static count is 8 instructions (from vmovsd to jne)
            base_instrs = static_len - 8
            loop_trip = 4
            has_loop = True
            loop_body_min = 6
            loop_body_max = 8
    elif 'avx512' in fn_name and 'vectorized' not in fn_name:
        if any('cmp' in i and 'jne' in mnems for i in instrs):
            # Loop of 8 iterations
            base_instrs = static_len - 8
            loop_trip = 8
            has_loop = True
            loop_body_min = 6
            loop_body_max = 8

    if has_loop:
        dyn_min = base_instrs + loop_trip * loop_body_min
        dyn_max = base_instrs + loop_trip * loop_body_max
        branches_exec = loop_trip * 2 # jbe + jne each iteration
    else:
        # Straight line code with at most 1 conditional branch (or 0 for branchless SVE)
        br_count = sum(1 for m in mnems if m.startswith(('j', 'b.', 'beq', 'bne', 'blt', 'bge', 'ble', 'bgt', 'cb', 'tb')) and m != 'bsf')
        if br_count == 0:
            dyn_min = static_len
            dyn_max = static_len
            branches_exec = 0
        else:
            # Typically 1 branch for if (bitmask != 0) / if (index_max != -1)
            # When taken vs not taken:
            # In VecRed, if bitmask == 0 (no new max): skips rep bsf, mov, etc. (~4-5 instrs)
            # min is when no new max found, max is when new max found
            # Or vice versa: je skips 4 instructions
            if any('je' in m or 'beq' in m or 'b.eq' in m for m in mnems):
                dyn_min = static_len - 4
                dyn_max = static_len
            else:
                dyn_min = static_len
                dyn_max = static_len
            branches_exec = br_count

    return {
        'static': static_len,
        'has_loop': has_loop,
        'dyn_min': dyn_min,
        'dyn_max': dyn_max,
        'branches': branches_exec
    }

# Run across all files
gcc_traces = sorted(glob.glob('outputs/trazas microbenchmarks completos/gcc/*.txt'))
clang_traces = sorted(glob.glob('outputs/trazas microbenchmarks completos/clang/*.txt'))

print("=== PARSED FUNCTIONS SUMMARY ===")
print(f"Total GCC trace files: {len(gcc_traces)}")
print(f"Total Clang trace files: {len(clang_traces)}")

