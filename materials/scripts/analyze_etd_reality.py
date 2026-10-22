"""Plot saved Fourier-defect controls; data-only analysis, no PDE execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root',type=Path,required=True)
args=ap.parse_args()
receipt=json.loads((args.root/'cpu-controls/results.json').read_text())
rows=receipt['long']
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharex=True,sharey='row',layout='constrained')
    for col,noise in enumerate([False,True]):
        for corrected,color,label in [(False,'#b54a40','Original stored spectrum'),(True,'#237d65','Real-spectrum constraint')]:
            row=next(r for r in rows if r['noise']==noise and r['corrected']==corrected)
            trace=row['trace'];times=[t['time'] for t in trace]
            defects=[max(t['hermitian_error'],1e-16) for t in trace]
            lo=[t['min_composition'] for t in trace];hi=[t['max_composition'] for t in trace]
            failure=row.get('failure',[])
            if failure:
                f=failure[0];times.append(f['time']);defects.append(f['after_hermitian_error'])
                lo.append(f['after_min']);hi.append(f['after_max'])
            elif row.get('steps'):
                times.append(6000.);defects.append(defects[-1]);lo.append(row['min']);hi.append(row['max'])
            axes[0,col].plot(times,defects,color=color,lw=2,label=label)
            axes[1,col].plot(times,lo,color=color,lw=1.5)
            axes[1,col].plot(times,hi,color=color,lw=1.5)
            if failure:
                axes[0,col].scatter(times[-1],defects[-1],marker='x',color=color,s=45,zorder=4)
                axes[1,col].scatter([times[-1]]*2,[lo[-1],hi[-1]],marker='x',color=color,s=35,zorder=4)
        axes[0,col].set_title('With thermal noise' if noise else 'Without thermal noise')
        axes[0,col].set_yscale('log');axes[0,col].set_ylim(1e-17,1e19)
        axes[1,col].axhline(0,color='.4',ls=':',lw=.8);axes[1,col].axhline(1,color='.4',ls=':',lw=.8)
        axes[1,col].set_xlabel('Dimensionless time')
        for ax in axes[:,col]:ax.grid(alpha=.15);ax.set_xlim(0,6100)
    axes[0,0].set_ylabel('Redundant Fourier-component magnitude\n(exact zero shown at $10^{-16}$)')
    axes[1,0].set_ylabel('Minimum and maximum composition')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('A numerical representation defect can mimic loss of physical bounds\n48 × 48 diagnostic; same equation, timestep and noise seed within each comparison',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(args.root/f'fourier-reality-controls.{ext}',dpi=180)
    plt.close(fig)
print('Saved Fourier-reality control figure')
