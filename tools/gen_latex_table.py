#!/usr/bin/env python3
"""Genera la tabla unificada en formato LaTeX para el paper científico (\label{tab:comp_avx})"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg_dynamic_counts as C
import gen_ph_tables as G

def latex_cell(res, is_p3=False):
    if not res: return "---"
    st = res["static"]
    dmin, dmax = res["dyn_min"], res["dyn_max"]
    if is_p3 and "iter" in res:
        imin, imax = res["iter"]
        return f"{st} ({dmin}--{dmax}) [{imin}--{imax}]"
    elif dmin == dmax:
        return f"{st} ({dmin})"
    else:
        return f"{st} ({dmin}--{dmax})"

def generate_latex():
    rows = []
    for alg in ("SCAMP", "SCRIMP"):
        a = alg.lower()
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{10}{l}{\textbf{" + alg + r"}} \\")
        rows.append(r"\midrule")
        for name, stem, loopstem in G.variants(a):
            line = [name.replace("_x", r"\_x").replace("`", "")]
            # GCC
            p1_g = G.get("gcc", stem, "PH1")
            p2_g = G.get("gcc", stem, "PH2")
            p3_g = G.get("gcc", loopstem or stem, "PH3")
            p4_g = G.get("gcc", stem, "PH4")
            line.append(str(p1_g["static"]) if p1_g else "---")
            line.append(latex_cell(p2_g))
            line.append(latex_cell(p3_g, is_p3=True))
            line.append(str(p4_g["static"]) if p4_g else "---")
            
            # Clang
            p1_c = G.get("clang", stem, "PH1")
            p2_c = G.get("clang", stem, "PH2")
            p3_c = G.get("clang", loopstem or stem, "PH3")
            p4_c = G.get("clang", stem, "PH4")
            line.append(str(p1_c["static"]) if p1_c else "---")
            line.append(latex_cell(p2_c))
            line.append(latex_cell(p3_c, is_p3=True))
            line.append(str(p4_c["static"]) if p4_c else "---")
            
            rows.append(" & ".join(line) + r" \\")
            
    header = r"""\begin{table*}[t]
\centering
\caption{Instruction complexity across the four algorithmic phases for SCAMP and SCRIMP under GCC 14.2 and Clang 18.1 (-O2). Formatted as \textit{Static (Dyn\_min--Dyn\_max)} for PH2 and PH3. For PH3, the inner hot-loop iteration range is appended in brackets \textit{[Iter\_min--Iter\_max]}.}
\label{tab:comp_avx}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{\textbf{GCC 14.2 (-O2)}} & \multicolumn{4}{c}{\textbf{Clang 18.1 (-O2)}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Architecture Variant} & \textbf{PH1} & \textbf{PH2} & \textbf{PH3} \textit{[Iter]} & \textbf{PH4} & \textbf{PH1} & \textbf{PH2} & \textbf{PH3} \textit{[Iter]} & \textbf{PH4} \\
"""
    footer = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""
    return header + "\n".join(rows) + "\n" + footer

if __name__ == "__main__":
    tex = generate_latex()
    print(tex)
    open('/tmp/p/table_comp_avx.tex', 'w').write(tex)
