from common import *
from speedups import label
import itertools

t=label(pd.read_csv(f'{OUT}/speedups_shuf.csv'))
d=load_runs()
# Metrica B: vs escalar con mismos hilos
sec=d[(d.variant=='shuf')&(d.rep==1)&(d['mode']=='sec')].set_index(['fraction','isa','alg','ds','threads']).sim_seconds
t['t_secN']=[sec.get((r.fraction,r.isa,r.alg,r.ds,r.threads),np.nan) for r in t.itertuples()]
t['SB']=t.t_secN/t.t_vect
s=t[t.series.notna()].copy()

print("== Medias metricas A/B (audio+human, 2 algs, 7 hilos)")
print(s.groupby(['series','fraction'])[['S','SB']].agg(lambda x:gmean(x.dropna())).round(2).unstack().to_string())
print(s.groupby(['series','fraction'])[['S','SB']].mean().round(2).unstack().to_string())

print("== Hilos optimos (argmax S) por serie")
pk=s.loc[s.groupby(['ds','alg','series','fraction']).S.idxmax()].pivot_table(index=['ds','alg','series'],columns='fraction',values='threads')
pk['match10']=pk[10]==pk[100]; pk['match25']=pk[25]==pk[100]
# peak tolerance: 100%-peak S within 5% of best
print(pk.to_string()); print(pk[['match10','match25']].mean())

print("== Orden por pares entre series en cada (ds,alg,threads) vs 100%")
res=[]
for (ds,alg,th),g in s.groupby(['ds','alg','threads']):
    p=g.pivot(index='series',columns='fraction',values='S')
    if p.shape[0]<3 or p.isna().any().any(): continue
    for a,b in itertools.combinations(p.index,2):
        for f in (10,25):
            res.append(dict(ds=ds,alg=alg,th=th,pair=f'{a}>{b}',f=f,agree=np.sign(p.loc[a,f]-p.loc[b,f])==np.sign(p.loc[a,100]-p.loc[b,100]),
                            margin=abs(np.log(p.loc[a,100]/p.loc[b,100]))))
r=pd.DataFrame(res); print(r.groupby('f').agree.mean().round(3))
print(r.groupby(['f',pd.cut(r.th,[0,8,16,64])],observed=True).agree.mean().round(3))
print("desacuerdos con margen>10% en 100%:"); print(r[(~r.agree)&(r.margin>0.1)].groupby('f').size())
print(r[(~r.agree)&(r.margin>0.1)].sort_values('th').to_string())

print("== Ratio SVE/RVV a igual VLEN<=2048, 1 hilo (media geom. sobre ds)")
v=t[(t.threads==1)&(t.vlen<=2048)&(t.vlen>=256)]
p=v.pivot_table(index=['alg','vlen','ds'],columns=['isa','fraction'],values='S')
for f in (10,25,100): p[('ratio',f)]=p[('arm',f)]/p[('riscv',f)]
print(p['ratio'].unstack('ds').round(3).to_string())
