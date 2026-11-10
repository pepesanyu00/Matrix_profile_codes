import collections
import os, glob, re
from generate_tables import parse_trace

def classify_instruction(inst, arch):
    inst_lower = inst.lower()
    m = inst_lower.split()[0]
    
    # 1. Useful FP compute
    if any(m.startswith(p) for p in ['vfmadd', 'vfmsub', 'vfnmsub', 'vmulpd', 'vaddpd', 'vsubpd', 'vdivpd', 'fmla', 'fmul', 'fadd', 'fsub', 'fdiv', 'vfmacc', 'vfmul', 'vfadd', 'vfsub', 'vfdiv']):
        return 'FP Útil'
    if any(m.startswith(p) for p in ['vmaxpd', 'vminpd', 'vmaxsd', 'vminsd', 'fmax', 'fmin', 'vfmax', 'vfmin']) and not any(p in inst_lower for p in ['fmaxv', 'fminv', 'vfredmax', 'vfredmin']):
        return 'Reducción / Shuffles'
        
    # 2. Reduction / Shuffles / Cross-lane
    if any(m.startswith(p) for p in ['fmaxv', 'fminv', 'vfredmax', 'vfredmin', 'vextractf', 'vpermilpd', 'vshufpd', 'vperm2f128', 'svlastb', 'vfmv.f.s', 'vfmv.s.f', 'fmv.d']):
        return 'Reducción / Shuffles'
        
    # 3. Mask compare / generation
    if any(m.startswith(p) for p in ['vcmppd', 'vcmpeqpd', 'vcmpgtpd', 'fcmgt', 'fcmeq', 'vmfeq', 'vmfgt', 'vmflt']):
        return 'Máscara: Comparación'
        
    # 4. Mask -> index
    if any(m.startswith(p) for p in ['vmovmskpd', 'kmovw', 'kmovb', 'tzcnt', 'bsf', 'vfirst.m', 'brkb', 'cntp', 'cmn', 'ccmp', 'cmp', 'ccmn']):
        return 'Máscara: Índice'
        
    # 5. Zeroing / movprfx / sel
    if m == 'movprfx' or (m == 'sel' and 'p0' in inst_lower):
        return 'Sobrecoste Puesta a Cero'
        
    # 6. Broadcasts
    if any(m.startswith(p) for p in ['vbroadcastsd', 'vpbroadcastq', 'ld1rd', 'ptrue', 'svdup']):
        return 'Broadcast / Setup VL'
    if m == 'vsetvli':
        return 'Broadcast / Setup VL'
        
    # 7. Algorithmic Loads / Stores
    if any(m.startswith(p) for p in ['vmovupd', 'vmovdqu64', 'vmaskmovpd', 'vpmaskmovq', 'ld1d', 'st1d', 'vle64.v', 'vse64.v', 'fsd', 'sd', 'str', 'vmovsd', 'movsd']):
        if any(sp in inst_lower for sp in ['rsp', 'rbp']) and not any(p in inst_lower for p in ['rsp+16', 'rsp+40', 'rbp+16']):
            return 'Pila / Spill'
        return 'Memoria Algorítmica'
        
    # 8. Stack Spills / Reloads
    if any(sp in inst_lower for sp in ['rsp-32', 'rsp-64', '[rsp-']):
        return 'Pila / Spill'
    if m in ['push', 'pop', 'leave']:
        return 'Gestión Escalar / ABI'
        
    # 9. Scalar GPR arithmetic & control
    return 'Gestión Escalar / ABI'

def breakdown_function(fn_name, instrs):
    cats = collections.Counter()
    for ins in instrs:
        cat = classify_instruction(ins, fn_name)
        cats[cat] += 1
    return cats

print("=== TAXONOMIC BREAKDOWN (GCC 14.2 MB2 DIAG_STEP) ===")
from generate_tables import gcc_data
for f, fns in sorted(gcc_data.items()):
    for fn, ins in fns.items():
        if 'diag_step' in fn and 'builtin' not in fn and 'scalar' not in fn:
            b = breakdown_function(fn, ins)
            print(f"\n{f} :: {fn} (Total: {len(ins)})")
            for cat, count in sorted(b.items(), key=lambda x: -x[1]):
                print(f"  {cat:26s}: {count:2d} ({count/len(ins)*100:4.1f}%)")
