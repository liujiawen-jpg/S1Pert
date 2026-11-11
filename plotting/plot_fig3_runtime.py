#!/usr/bin/env python3
"""Fig. 3 reads actual Stage-7 timing logs, including explicit amortization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .paper_plot_common import *
except ImportError:
    from paper_plot_common import *

METHODS=[('S1Pert','#C0392B'),('ALIGNED','#7D3C98'),('SUMMER','#AA7700')]

def render(work_dir, timing=None, output=None, font='Times New Roman',fallback='DejaVu Serif'):
    root=work_root(work_dir);timing=Path(timing or root/'paper/runtime/timings.json').resolve()
    output=Path(output or root/'paper/figures').resolve();data=json.loads(timing.read_text())
    if data['status']!='PASS':raise ValueError('Do not plot an incomplete timing execution.')
    font_info=setup_font(font,fallback);rows=data['jobs'];cells=['hepg2','k562_essential'];summary=[]
    # Loaded-model and load-inclusive latencies cannot be silently mixed.
    scopes={r['online_scope'] for r in rows if r['method'] in dict(METHODS) and r['cell'] in cells}
    if len(scopes)!=1:raise ValueError('Mixed online timing scopes; run a consistent benchmark plan before plotting.')
    scope=next(iter(scopes))
    for method,_ in METHODS:
        for cell in cells:
            group=[r for r in rows if r['method']==method and r['cell']==cell]
            if not group:raise ValueError(f'Missing measured runtime: {method}/{cell}')
            for metric in ['online_seconds_per_10000','end_to_end_seconds']:
                vals=[float(r[metric]) for r in group if r.get(metric) is not None]
                if len(vals)!=len(group) or any(not np.isfinite(x) or x<=0 for x in vals):
                    raise ValueError(f'Incomplete/nonpositive measured timing: {method}/{cell}/{metric}')
                summary.append(dict(method=method,cell=cell,metric=metric,n_splits=len(vals),
                    mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None))
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6));fig.subplots_adjust(left=.085,right=.97,bottom=.18,top=.78,wspace=.35)
    for axis in axes:style(axis);axis.set_yscale('log');axis.set_xticks(range(2),['HepG2','K562'])
    axes[0].set_title('a',loc='left');axes[1].set_title('b',loc='left')
    axes[0].set_ylabel('Online inference (s / 10,000 pairs)' if scope=='loaded_model_online' else 'Inference wall time incl. load\n(s / 10,000 pairs)')
    axes[1].set_ylabel('End-to-end runtime (hours / split)')
    for i,(method,color) in enumerate(METHODS):
        for j,metric in enumerate(['online_seconds_per_10000','end_to_end_seconds']):
            for k,cell in enumerate(cells):
                row=next(r for r in summary if (r['method'],r['cell'],r['metric'])==(method,cell,metric))
                divisor=3600 if j else 1;y=row['mean']/divisor;sd=row['sample_sd']/divisor if row['sample_sd'] is not None else None
                x=k+(i-1)*.23
                axes[j].bar(x,y,.21,color=color,label=method if k==0 else None)
                if sd is not None:axes[j].errorbar(x,y,yerr=sd,fmt='none',color=color,capsize=2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
    save_figure(fig,output,'fig_runtime_efficiency');plt.close(fig)
    write_csv(output/'fig3_timing_summary.csv',summary)
    report=dict(status='PASS',source=str(timing),source_sha256=sha256(timing),font=font_info,online_scope=scope,
        end_to_end='Measured offline preparation amortized by declared reuse count + training when applicable + complete test inference, including factory setup and recorded isolated-process overhead.',
        error_bars='Sample SD across measured splits; no error bar is invented for n=1.',output=str(output))
    write_json(output/'fig3_validation.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir');p.add_argument('--timing',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--font',default='Times New Roman');p.add_argument('--font-fallback',default='DejaVu Serif')
    a=p.parse_args();print(json.dumps(render(a.work_dir,a.timing,a.output,a.font,a.font_fallback),indent=2))
