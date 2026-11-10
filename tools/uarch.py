from common import *
from speedups import label
CLK=2e9
def table(variant='shuf'):
    d=load_runs(); x=d[(d.variant==variant)&(d.rep==1)&(d['mode']=='vect')&d.ds.isin(['audio','human_activity'])].copy()
    rows=[]
    for r in x.itertuples():
        sv=sampling_row(r.dataset,r.vlen//64,r.fraction,variant=='shuf',r.threads)
        cells=sv['cov']*total_cells(r.dataset); cyc=r.sim_seconds*CLK
        rows.append(dict(isa=r.isa,alg=r.alg,ds=r.ds,vlen=r.vlen,threads=r.threads,fraction=r.fraction,upt=sv['bundles_sel']/r.threads,
            ipc_core=r.sim_insts/(cyc*r.threads), insts_cell=r.sim_insts/cells, vec_share=r.vec_insts/r.sim_insts,
            l1_mr=r.l1_miss/r.l1_acc if r.l1_acc else np.nan, l2_mr=r.l2_miss/r.l2_acc if r.l2_acc else np.nan, l3_mr=r.l3_miss/r.l3_acc if r.l3_acc else np.nan,
            dram_B_cell=(r.dram_rd+r.dram_wr)/cells, dram_BW=(r.dram_rd+r.dram_wr)/r.sim_seconds/1e9,
            misp_kinst=1e3*r.br_misp/r.sim_insts, iq_kinst=1e3*r.iq_full/r.sim_insts, lsq_kinst=1e3*r.lsq_full/r.sim_insts,
            fubusy_kinst=1e3*r.fu_busy/r.sim_insts, cyc_cell=cyc*r.threads/cells))
    return label(pd.DataFrame(rows))
if __name__=="__main__":
    t=table(); t.to_csv(f'{OUT}/uarch_shuf.csv',index=False)
    M=['ipc_core','insts_cell','vec_share','l1_mr','l2_mr','l3_mr','dram_B_cell','misp_kinst','iq_kinst','lsq_kinst','fubusy_kinst','cyc_cell']
    k=['isa','alg','ds','vlen','threads']
    ref=t[t.fraction==100].set_index(k)[M]
    res=[]
    for f in (10,25):
        g=t[t.fraction==f].set_index(k); j=g[M]/ref.loc[g.index.intersection(ref.index)][M]; j['upt']=g.upt; j['f']=f; res.append(j.reset_index())
    r=pd.concat(res).replace([np.inf,-np.inf],np.nan)
    r.to_csv(f'{OUT}/uarch_ratio.csv',index=False)
    for name,m in (('u>=16',r.upt>=16),('4<=u<16',(r.upt>=4)&(r.upt<16)),('u<4',r.upt<4)):
        print('==',name, m.sum()); print(r[m].groupby('f')[M].median().round(3).to_string())
    print('== solo 1 hilo, vlen<=2048 (cualquier u):'); m=(r.threads==1)&(r.vlen<=2048); print(r[m].groupby('f')[M].agg(['min','max']).round(3).T.to_string())
