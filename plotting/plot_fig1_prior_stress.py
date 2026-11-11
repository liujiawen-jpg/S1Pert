#!/usr/bin/env python3
"""Fig. 1 from Stage-5 measured summaries; all values originate in raw predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .paper_plot_common import *
except ImportError:
    from paper_plot_common import *
from matplotlib.lines import Line2D
from matplotlib.text import Text

METHODS=[('S1Pert','S1Pert (Ours)','#C0392B','o','-'),('P1','P1','#7F8C8D','s','--'),
         ('ALIGNED','ALIGNED','#7D3C98','*',':'),('GenePT-Prot','GenePT-Prot','#2471A3','^','-'),
         ('STATE','STATE','#3B3B98','h','-.'),('GEARS','GEARS','#16A085','D','-')]
ENTROPY=['entropy_low','entropy_medium','entropy_high']
SUPPORT=['support_ge50','support_10_49','support_1_9','support_0']

def annotate_points(ax, points):
    """Layout fitted to actual coordinates rather than historical numeric values."""
    fig=ax.figure;fig.canvas.draw();renderer=fig.canvas.get_renderer()
    panel=ax.get_window_extent(renderer);anchors=[ax.transData.transform((r['x'],r['y'])) for r in points]
    pending=[];occupied=[];audit=[]
    for row in points:
        annotation=ax.annotate(f"{row['label']}\n(Chg AUC: {row['change_auc']:.3f})",
            (row['x'],row['y']),xytext=(0,0),textcoords='offset points',ha='center',va='center',
            fontsize=9.5,color=row['color'],bbox=dict(facecolor='white',edgecolor='none',alpha=.98,pad=1.5),zorder=10)
        annotation.update_positions(renderer);box=Text.get_window_extent(annotation,renderer).padded(4)
        pending.append((box.width*box.height,row,annotation))
    for _,row,annotation in sorted(pending,key=lambda item:-item[0]):
        anchor=ax.transData.transform((row['x'],row['y']))
        grid=[(x,y) for x in np.arange(panel.x0+5,panel.x1-4,6) for y in np.arange(panel.y0+5,panel.y1-4,6)]
        grid.sort(key=lambda xy:(xy[0]-anchor[0])**2+(xy[1]-anchor[1])**2)
        for x,y in grid:
            offset=((x-anchor[0])*72/fig.dpi,(y-anchor[1])*72/fig.dpi)
            annotation.set_position(offset);annotation.update_positions(renderer)
            box=Text.get_window_extent(annotation,renderer).padded(4)
            if box.x0<panel.x0 or box.y0<panel.y0 or box.x1>panel.x1 or box.y1>panel.y1:continue
            if any(box.overlaps(other) for other in occupied):continue
            if any(box.padded(2).contains(*point) for point in anchors):continue
            break
        else:raise ValueError('Cannot fit scatter labels: increase figure width/font spacing.')
        occupied.append(box)
        label_center=ax.transData.inverted().transform((x,y))
        ax.annotate('',(row['x'],row['y']),xytext=label_center,textcoords='data',zorder=7,
            arrowprops=dict(arrowstyle='-',color=row['color'],linewidth=.6))
        audit.append(dict(model=row['model'],text=annotation.get_text(),bbox=list(map(float,box.extents))))
    return audit

def render(work_dir, summary=None, output=None, expected_splits=20, font='Times New Roman',fallback='DejaVu Serif'):
    root=work_root(work_dir);summary=Path(summary or root/'paper/prior_stress/summary.csv').resolve()
    output=Path(output or root/'paper/figures').resolve();font_info=setup_font(font,fallback)
    rows=read_csv(summary);index={};used={};coordinates=[]
    for row in rows:
        if row['cell']=='ALL_CELLS':
            key=(row['model'],row['regime'],row['metric'])
            if key in index:raise ValueError(f'Duplicate summary {key}')
            index[key]=row
    def value(model,regime,metric='macro_f1'):
        key=(model,regime,metric)
        row=index.get(key)
        if row is None or int(row['n_splits'])!=expected_splits or row['mean']=='':
            raise ValueError(f'Missing complete {expected_splits}-split metric: {key}')
        result={**row,'mean':float(row['mean']),'std':float(row['std']) if row['std'] else None,
            'wilcoxon_holm_p':float(row['wilcoxon_holm_p']) if row.get('wilcoxon_holm_p') else None}
        if not 0<=result['mean']<=1 or result['std'] is None:
            raise ValueError('Figure needs finite [0,1] mean and measured sample SD (>=2 splits).')
        used[key]=result;return result
    fig=plt.figure(figsize=(18,5.8))
    gs=fig.add_gridspec(1,3,left=.055,right=.985,bottom=.20,top=.78,wspace=.36,width_ratios=[1.05,1.15,1])
    a=fig.add_subplot(gs[0,0]);c=fig.add_subplot(gs[0,2]);middle=gs[0,1].subgridspec(1,2,width_ratios=[3,1.1],wspace=.25)
    b=fig.add_subplot(middle[0,0]);br=fig.add_subplot(middle[0,1])
    for ax in [a,b,br,c]:style(ax)
    fig.legend([Line2D([],[],color=co,marker=mk,ls=ls,label=label) for _,label,co,mk,ls in METHODS],
        [label for _,label,*_ in METHODS],ncol=6,loc='upper center',bbox_to_anchor=(.51,.98),frameon=False)
    a.set_title('a',loc='left');b.set_title('b',loc='left');c.set_title('c',loc='left')
    a.set_xlabel('Pol AUC');a.set_ylabel('Overall Macro-F1');a.axvline(.5,color='#AAA',ls='--',lw=.7)
    points=[]
    for model,label,color,marker,style_ in METHODS:
        x=value(model,'resolved','within_gene_polarity_auc');y=value(model,'resolved');chg=value(model,'resolved','within_gene_change_auc')
        a.errorbar(x['mean'],y['mean'],xerr=x['std'],yerr=y['std'],fmt=marker,color=color,capsize=2,ms=7)
        points.append(dict(model=model,label='S1Pert' if model=='S1Pert' else label,color=color,x=x['mean'],y=y['mean'],change_auc=chg['mean']))
        coordinates.append(dict(panel='a',model=model,regime='resolved',metric='macro_f1',x=x['mean'],y=y['mean'],x_sd=x['std'],y_sd=y['std']))
    xmin=min(value(m,'resolved','within_gene_polarity_auc')['mean']-value(m,'resolved','within_gene_polarity_auc')['std'] for m,*_ in METHODS)
    xmax=max(value(m,'resolved','within_gene_polarity_auc')['mean']+value(m,'resolved','within_gene_polarity_auc')['std'] for m,*_ in METHODS)
    ymin=min(value(m,'resolved')['mean']-value(m,'resolved')['std'] for m,*_ in METHODS)
    ymax=max(value(m,'resolved')['mean']+value(m,'resolved')['std'] for m,*_ in METHODS)
    a.set_xlim(xmin-.055,xmax+.055);a.set_ylim(ymin-.13,ymax+.13)
    annotations=annotate_points(a,points)
    b.set_ylabel('Macro-F1');b.set_xticks(range(3),['Low\nentropy','Med.\nentropy','High\nentropy']);b.set_ylim(0,1.12)
    br.set_ylabel('Down-F1');br.set_xticks([0],['Counter-\npropensity']);br.set_ylim(0,1.12);br.set_facecolor('#F3F4F5')
    br.yaxis.set_label_position('right');br.tick_params(axis='y',labelleft=False,labelright=True,left=False,right=True)
    width=.78/len(METHODS);comparisons=[]
    for i,(model,label,color,marker,style_) in enumerate(METHODS):
        for j,regime in enumerate(ENTROPY+['counter_propensity']):
            axis=b if j<3 else br;metric='macro_f1' if j<3 else 'down_f1';v=value(model,regime,metric)
            x=(j if j<3 else 0)+(i-2.5)*width
            axis.bar(x,v['mean'],width*.86,color=color,hatch='//' if model=='P1' else None)
            axis.errorbar(x,v['mean'],yerr=v['std'],fmt='none',color=color,capsize=1.3,lw=.7)
            coordinates.append(dict(panel='b' if j<3 else 'b_reversal',model=model,regime=regime,metric=metric,x=x,y=v['mean'],y_sd=v['std']))
            if model!='S1Pert':
                own=value('S1Pert',regime,metric)
                comparisons.append(dict(model=model,regime=regime,metric=metric,reference='S1Pert',
                    reference_mean=own['mean'],baseline_mean=v['mean'],paired_n=v['paired_n'],
                    holm_p=v['wilcoxon_holm_p'],family='Full registry family retained from evaluator.'))
    c.set_ylabel('Macro-F1');c.set_xlabel(r'Training support $S_{ij}^{q}$'+'\n'+r'(including cold start $S_{ij}^{q}=0$)')
    c.set_xticks(range(4),['≥50','10–49','1–9','0']);c.set_ylim(-.08,1.04)
    for i,(model,label,color,marker,style_) in enumerate(METHODS):
        vals=[value(model,r) for r in SUPPORT];xx=np.arange(4)+(i-2.5)*.018
        c.errorbar(xx,[v['mean'] for v in vals],yerr=[v['std'] for v in vals],color=color,marker=marker,ls=style_,capsize=1.5,lw=1.5)
        for x,r,v in zip(xx,SUPPORT,vals):coordinates.append(dict(panel='c',model=model,regime=r,metric='macro_f1',x=float(x),y=v['mean'],y_sd=v['std']))
    save_figure(fig,output,'fig_prior_stress');plt.close(fig)
    write_csv(output/'fig1_plotted_values.csv',used.values());write_csv(output/'fig1_coordinates.csv',coordinates)
    write_csv(output/'fig1_paired_tests.csv',comparisons)
    report=dict(status='PASS',source=str(summary),source_sha256=sha256(summary),expected_splits=expected_splits,
        font=font_info,methods=[m for m,*_ in METHODS],panel_titles=['a','b','c'],
        counter_propensity_metric='down_f1',error_bars='sample SD (ddof=1)',annotation_audit=annotations,
        source_driven=True,output=str(output),holm_family='All registered comparators; no subset recomputation.')
    write_json(output/'fig1_validation.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir');p.add_argument('--summary',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--expected-splits',type=int,default=20)
    p.add_argument('--font',default='Times New Roman');p.add_argument('--font-fallback',default='DejaVu Serif')
    a=p.parse_args();print(json.dumps(render(a.work_dir,a.summary,a.output,a.expected_splits,a.font,a.font_fallback),indent=2))
