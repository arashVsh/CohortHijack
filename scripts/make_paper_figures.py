from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def save(fig,out,name):
    fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(out/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis-dir',required=True); ap.add_argument('--output-dir',default='outputs/paper_figures'); a=ap.parse_args()
    ad=Path(a.analysis_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

    # Fig 1: main structured-vs-random at 5%
    d=pd.read_csv(ad/'main_attack_summary.csv'); d=d[d.strategy.isin(['random','nearest','same_class'])]
    d['setting']=d.dataset.str.upper()+' / '+d.classifier.str.replace('_',' ').str.title()
    settings=list(dict.fromkeys(d.setting)); strategies=['random','nearest','same_class']; labels=['Random','Nearest','Same class']
    x=np.arange(len(settings)); width=.24
    fig,ax=plt.subplots(figsize=(8.2,4.6))
    for i,(st,lab) in enumerate(zip(strategies,labels)):
        q=d[d.strategy==st].set_index('setting').reindex(settings)
        ax.bar(x+(i-1)*width,q.flip_rate*100,width,label=lab)
    ax.set_ylabel('Target flip rate (%)'); ax.set_xticks(x,settings,rotation=18,ha='right'); ax.set_title('Structured companion-cell removal is stronger than random removal'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
    save(fig,out,'figure1_structured_vs_random')

    # Fig 2: context-weight dose response on Paul15 at 5%
    ab=pd.read_csv(ad/'ablation_summary.csv'); base=d.copy();
    mapping={'context_weight_0':0.0,'context_weight_0p5':0.5,'context_weight_2':2.0}
    q=ab[(ab.dataset=='paul15') & ab.condition.isin(mapping) & ab.strategy.isin(['nearest','same_class'])].copy(); q['weight']=q.condition.map(mapping)
    b=d[(d.dataset=='paul15') & d.strategy.isin(['nearest','same_class'])][['classifier','strategy','flip_rate']].copy(); b['weight']=1.0
    q=pd.concat([q[['classifier','strategy','flip_rate','weight']],b],ignore_index=True)
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    for (clf,st),g in q.groupby(['classifier','strategy']):
        g=g.sort_values('weight'); ax.plot(g.weight,g.flip_rate*100,marker='o',label=f"{clf.replace('_',' ').title()} — {st.replace('_',' ')}")
    ax.set_xlabel('Context weight'); ax.set_ylabel('Target flip rate (%)'); ax.set_xticks([0,.5,1,2]); ax.set_title('Vulnerability increases with cohort-context dependence'); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.25)
    save(fig,out,'figure2_context_weight_ablation')

    # Fig 3: real CellTypist validation
    ct=pd.read_csv(ad/'celltypist_summary.csv'); ct['stratum_label']=ct.stratum.map({'changed_by_majority_vote':'Context-sensitive','stable_under_majority_vote':'Initially stable'}).fillna(ct.stratum)
    ct['strategy_label']=ct.strategy.map({'random':'Random','nearest':'Nearest','same_majority_label':'Same label'}).fillna(ct.strategy)
    cats=[]
    for stratum in ['Context-sensitive','Initially stable']:
      for budget in sorted(ct.budget.unique()): cats.append((stratum,budget))
    x=np.arange(len(cats)); width=.24
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    for i,st in enumerate(['Random','Nearest','Same label']):
        vals=[]
        for stratum,budget in cats:
            z=ct[(ct.stratum_label==stratum)&(ct.strategy_label==st)&np.isclose(ct.budget,budget)]
            vals.append(float(z.flip_rate.iloc[0]*100) if len(z) else np.nan)
        ax.bar(x+(i-1)*width,vals,width,label=st)
    ax.set_ylabel('Majority-voted target flip rate (%)'); ax.set_xticks(x,[f'{s}\n{int(b*100)}% removal' for s,b in cats]); ax.set_title('CellTypist majority voting is sensitive to query-cohort composition'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
    save(fig,out,'figure3_celltypist_validation')
    print(f'Wrote figures to {out}')
if __name__=='__main__': main()
