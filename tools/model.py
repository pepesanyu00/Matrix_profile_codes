"""Modelo de coste: T = a + c*X, X = celdas cubiertas (M1) o makespan dinamico en celdas (M2).
Calibra con 10% y 25% y predice el 100%. Referencia ingenua: T10 * X100/X10."""
from common import *

def fit():
    t=pd.read_csv(f'{OUT}/speedups_shuf.csv')
    t['mk']=t.cov_vect/t.max_par
    rows=[]
    for k,g in t.groupby(['isa','alg','ds','vlen','threads']):
        g=g.set_index('fraction')
        if not {10,25,100}<=set(g.index): continue
        T={f:g.loc[f,'t_vect'] for f in (10,25,100)}
        o=dict(zip(['isa','alg','ds','vlen','threads'],k)); o['upt10']=g.units_per_thread[10]
        for name,x in (('cov',g.cov_vect),('mk',g.mk)):
            naive=T[10]*x[100]/x[10]; o[f'err_naive_{name}']=naive/T[100]-1
            dx=x[25]-x[10]
            if abs(dx)<1e-4*x[100]: o[f'err_{name}']=np.nan; o[f'a_{name}']=np.nan; continue
            c=(T[25]-T[10])/dx; a=T[10]-c*x[10]
            o[f'err_{name}']=(a+c*x[100])/T[100]-1; o[f'a_{name}']=a/T[100]
        rows.append(o)
    return pd.DataFrame(rows)

if __name__=="__main__":
    m=fit(); m.to_csv(f'{OUT}/model_fit.csv',index=False)
    cols=['err_naive_cov','err_naive_mk','err_cov','err_mk']
    for c in cols: m['abs_'+c]=m[c].abs()*100
    ac=['abs_'+c for c in cols]
    print(len(m), m[ac].isna().sum().to_dict())
    print(m.groupby('ds')[ac].agg(['mean','median']).round(2).to_string())
    print(m.groupby(pd.cut(m.upt10,[0,1,4,16,64,1e9]),observed=True)[ac+['a_cov','a_mk']].mean().round(2).to_string())
    print(m.groupby('threads')[ac].mean().round(1).to_string())
