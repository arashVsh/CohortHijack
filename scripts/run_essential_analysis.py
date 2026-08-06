from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

KEYS=['dataset','seed','classifier','target_index']

def bootstrap_ci(values, n_boot=5000, seed=2026):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if len(v)==0:return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    b=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(n_boot)])
    return tuple(np.quantile(b,[.025,.975]))

def summarize_main(df,budget=.05):
    x=df[np.isclose(df.budget,budget)].copy()
    rows=[]
    for keys,g in x.groupby(['dataset','classifier','audit']):
        per=g.groupby(KEYS,as_index=False).agg(flip=('target_flipped','mean'),collateral=('collateral_flip_rate','mean'),removed=('removed_fraction','mean'))
        lo,hi=bootstrap_ci(per.flip)
        rows.append(dict(dataset=keys[0],classifier=keys[1],strategy=keys[2],budget=budget,n_targets=len(per),flip_rate=per.flip.mean(),ci_low=lo,ci_high=hi,mean_collateral=per.collateral.mean(),mean_removed=per.removed.mean()))
    return pd.DataFrame(rows)

def paired_tests(df,budget=.05):
    x=df[np.isclose(df.budget,budget)]
    rand=x[x.audit=='random'].groupby(KEYS).target_flipped.mean().rename('random')
    rows=[]
    for strategy in ['nearest','same_class']:
        s=x[x.audit==strategy].groupby(KEYS).target_flipped.mean().rename('structured')
        z=pd.concat([rand,s],axis=1).dropna()
        for (dataset,clf),g in z.groupby(level=[0,2]):
            d=g.structured-g.random
            try: stat,p=wilcoxon(d,alternative='greater',zero_method='zsplit')
            except ValueError: stat,p=np.nan,np.nan
            lo,hi=bootstrap_ci(d)
            rows.append(dict(dataset=dataset,classifier=clf,strategy=strategy,n=len(g),mean_difference=d.mean(),ci_low=lo,ci_high=hi,wilcoxon_stat=stat,p_value=p))
    return pd.DataFrame(rows)

def search_summary(df):
    rows=[]
    for (clf,audit),g in df.groupby(['classifier','audit']):
        final=g.sort_values('removed_fraction').groupby(KEYS,as_index=False).tail(1)
        success=final.target_flipped.astype(float)
        lo,hi=bootstrap_ci(success)
        successful=g[g.target_flipped].sort_values('removed_fraction').groupby(KEYS,as_index=False).head(1)
        rows.append(dict(classifier=clf,method=audit,n_targets=len(final),success_rate=success.mean(),ci_low=lo,ci_high=hi,median_successful_removal=successful.removed_fraction.median(),mean_successful_removal=successful.removed_fraction.mean(),mean_success_collateral=successful.collateral_flip_rate.mean()))
    return pd.DataFrame(rows)

def class_analysis(main,search):
    final=search.sort_values('removed_fraction').groupby(KEYS+['audit'],as_index=False).tail(1)
    ms=final[final.audit=='multistart_greedy']
    rows=[]
    for (clf,label),g in ms.groupby(['classifier','target_clean_label']):
        rows.append(dict(classifier=clf,cell_type=label,n=len(g),class_frequency=g.target_class_frequency.mean(),clean_confidence=g.target_clean_confidence.mean(),attack_success=g.target_flipped.mean()))
    out=pd.DataFrame(rows)
    corr=[]
    for clf,g in out.groupby('classifier'):
        for x in ['class_frequency','clean_confidence']:
            r,p=spearmanr(g[x],g.attack_success,nan_policy='omit')
            corr.append(dict(classifier=clf,predictor=x,spearman_rho=r,p_value=p,n_cell_types=len(g)))
    return out,pd.DataFrame(corr)

def ablation_summary(root:Path,budget=.05):
    rows=[]
    for p in sorted(root.glob('*/aggregate_results.csv')):
        d=pd.read_csv(p); d=d[np.isclose(d.budget,budget)]
        for keys,g in d.groupby(['run_label','dataset','classifier','audit']):
            rows.append(dict(condition=keys[0],dataset=keys[1],classifier=keys[2],strategy=keys[3],budget=budget,flip_rate=np.average(g.target_flip_rate,weights=g.n),mean_collateral=np.average(g.mean_collateral_flip_rate,weights=g.n),n=int(g.n.sum())))
    return pd.DataFrame(rows)

def celltypist_summary(df):
    rows=[]
    for keys,g in df.groupby(['target_stratum','strategy','budget']):
        per=g.groupby(['seed','target_index']).agg(flip=('target_flipped','mean'),independent=('independent_prediction_changed','mean'),collateral=('collateral_flip_rate','mean')).reset_index()
        lo,hi=bootstrap_ci(per.flip)
        rows.append(dict(stratum=keys[0],strategy=keys[1],budget=keys[2],n_targets=len(per),flip_rate=per.flip.mean(),ci_low=lo,ci_high=hi,independent_change_rate=per.independent.mean(),mean_collateral=per.collateral.mean()))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--main',required=True); ap.add_argument('--search-v2',required=True); ap.add_argument('--ablations',required=True); ap.add_argument('--celltypist',required=True); ap.add_argument('--output-dir',default='outputs/publication_analysis')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(Path(a.main)/'per_target_results.csv'); s=pd.read_csv(Path(a.search_v2)/'per_target_results.csv'); c=pd.read_csv(Path(a.celltypist)/'per_target_results.csv')
    tables={
      'main_attack_summary':summarize_main(m), 'paired_structured_vs_random':paired_tests(m), 'search_v2_summary':search_summary(s),
      'ablation_summary':ablation_summary(Path(a.ablations)), 'celltypist_summary':celltypist_summary(c)}
    pc,cor=class_analysis(m,s); tables['per_class_vulnerability']=pc; tables['vulnerability_correlations']=cor
    for n,d in tables.items(): d.to_csv(out/f'{n}.csv',index=False)
    summary={'main_rows':len(m),'search_rows':len(s),'celltypist_rows':len(c),'tables':list(tables)}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Wrote analysis tables to {out}')
if __name__=='__main__': main()
