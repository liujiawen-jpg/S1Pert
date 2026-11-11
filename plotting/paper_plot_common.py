"""Plotting utilities; no bundled proprietary fonts or stored metric arrays."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.paper_common import read_csv, write_csv, write_json, sha256, work_root
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

def setup_font(requested='Times New Roman', fallback='DejaVu Serif'):
    try:
        path=font_manager.findfont(font_manager.FontProperties(family=requested),fallback_to_default=False)
        selected=requested
    except ValueError:
        path=font_manager.findfont(font_manager.FontProperties(family=fallback),fallback_to_default=False)
        selected=fallback
    plt.rcParams.update({'font.family':selected,'font.size':11,'axes.labelsize':12,
        'axes.titlesize':14,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':12,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'dejavuserif'})
    return dict(requested=requested,selected=selected,font_file=path,bundled_font=False)

def style(ax):
    ax.grid(axis='y',color='#B7C0C7',alpha=.4,linestyle=':',linewidth=.6)
    ax.set_axisbelow(True)

def save_figure(fig, output, stem):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg']:
        fig.savefig(output/f'{stem}.{ext}',bbox_inches='tight')
    fig.savefig(output/f'{stem}.png',dpi=220,bbox_inches='tight')

