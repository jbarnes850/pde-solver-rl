"""Figures for the materials-decision section of the RLVP post.

Every plotted value comes from materials/out/post-data.json, which
collect_post_data.py builds from materials/data. Literals below set layout.
Style follows the figure spec in the blog's AGENTS.md (IBM Plex Sans, text in #36332f and #6f675e, at most three
hues, no red/green pair, type at least 11 px when the figure renders 700 px wide, raster at 2x the 800 px column).
Titles live in the post captions. main() checks the type size of every saved figure.
"""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rlvp-materials-mpl'))
import matplotlib as mpl  # noqa: E402
mpl.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'materials/out/post-data.json'  # written by collect_post_data.py
OUT = ROOT/'materials/out/figures'
PAPER, INK, SOFT, MUTED, RULE = '#fbfaf7', '#151514', '#36332f', '#6f675e', '#ded8cd'
BLUE, WARM, NEUTRAL = '#2457ff', '#c2562f', '#aaa59d'
FONTS = ROOT/'materials/fonts'  # optional IBM Plex Sans TTFs; DejaVu Sans is used if absent
for _f in sorted(FONTS.glob('*.ttf')):
    mpl.font_manager.fontManager.addfont(str(_f))
SANS = 'IBM Plex Sans' if any(f.name == 'IBM Plex Sans' for f in mpl.font_manager.fontManager.ttflist) else 'DejaVu Sans'
MIN_PX_AT_700 = 11
DISPLAY_PX = 1600  # 2x the post column
FIELD = mpl.colors.LinearSegmentedColormap.from_list('phase', ['#1d1c1a', '#6f675e', '#e9e3d8', '#fbfaf7'])
PAPER_EDGE = '#fbfaf7'
mpl.rcParams.update({
    'font.family': [SANS, 'DejaVu Sans'], 'font.size': 12.5, 'text.color': SOFT, 'mathtext.fontset': 'custom',
    'mathtext.rm': SANS, 'mathtext.it': SANS + ':italic',
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
    'axes.edgecolor': RULE, 'axes.labelcolor': SOFT, 'axes.labelsize': 13, 'xtick.labelsize': 12.5,
    'ytick.labelsize': 12.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'xtick.major.pad': 8, 'ytick.major.pad': 8, 'axes.linewidth': .8})


def grid(ax):
    ax.yaxis.grid(True, color=RULE, lw=.8)
    ax.set_axisbelow(True)


def components(d):
    """Matched rerun: chains with a verified magnetic term against model calls per chain."""
    m = d['magnetic_matched']
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    offsets = {'adapter|fresh': 7, 'adapter|feedback': -9, 'base|feedback': 9, 'base|fresh': -9}
    for arm, color in [('base', WARM), ('adapter', BLUE)]:
        for policy, style in [('fresh', '-'), ('feedback', '--')]:
            r = sorted([x for x in m['rows'] if x['arm'] == arm and x['policy'] == policy], key=lambda x: x['budget'])
            b = [x['budget'] for x in r]
            p = [x['passed_components']/x['lineages'] for x in r]
            ax.plot(b, p, style, color=color, lw=2.6 if policy == 'fresh' else 1.8, marker='o', ms=5,
                    alpha=1 if policy == 'fresh' else .8)
            name = {'base': 'Base model', 'adapter': 'Ours (RLVP)'}[arm]
            how = {'feedback': 'revising with check feedback', 'fresh': 'fresh samples'}[policy]
            ax.annotate(f'{name}, {how}', (b[-1], p[-1]), xytext=(10, offsets[f'{arm}|{policy}']),
                        textcoords='offset points', color=color, fontsize=12, va='center')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlim(.85, 30)
    ax.spines['bottom'].set_bounds(1, 8)
    ax.set_ylim(-.03, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel('Model calls per attempt chain')
    ax.xaxis.label.set_x((np.log2(8**.5)-np.log2(.85))/(np.log2(30)-np.log2(.85)))  # centered under the data
    ax.set_ylabel('Chains with a verified magnetic term')
    grid(ax)
    fig.tight_layout()
    return fig


def checks(d):
    """The Fourier representation defect, with thermal noise: emphasized defect, muted corrected reference."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for r in d['fourier_defect']:
        if not r['noise']:
            continue
        t = [x['time'] for x in r['trace']]
        e = [max(x['hermitian_error'], 1e-16) for x in r['trace']]
        if r['failure']:
            t.append(r['failure']['time'])
            e.append(r['failure']['after_hermitian_error'])
        elif r['final']:
            t.append(r['final']['time'])
            e.append(e[-1])
        if r['corrected']:
            ax.plot(t, e, color=MUTED, lw=1.8)
            ax.annotate('with the match enforced, zero throughout', (t[-1], e[-1]), xytext=(-4, 8), textcoords='offset points',
                        ha='right', color=MUTED, fontsize=12)
        else:
            ax.plot(t, e, color=INK, lw=2.6)
            ax.scatter([t[-1]], [e[-1]], marker='x', s=70, color=INK, zorder=5, lw=2)
            ax.annotate('without the fix', (t[len(t)//2], e[len(e)//2]), xytext=(-10, 10),
                        textcoords='offset points', ha='right', color=SOFT, fontsize=12)
            ax.annotate('composition leaves its physical range', (t[-1], e[-1]), xytext=(12, 0), textcoords='offset points',
                        ha='left', va='center', color=SOFT, fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(1e-18, 1e20)
    ax.set_yticks([1e-16, 1e-8, 1, 1e8, 1e16])
    ax.set_yticklabels([r'$10^{-16}$', r'$10^{-8}$', '1', r'$10^{8}$', r'$10^{16}$'])
    ax.set_xlim(0, 8200)
    ax.set_xticks(range(0, 6001, 1000))
    ax.spines['bottom'].set_bounds(0, 6000)
    ax.set_xlabel('Simulated time (model units)')
    ax.set_ylabel('Mismatch between mirrored waves')
    grid(ax)
    fig.tight_layout()
    return fig


def main():
    d = json.loads(DATA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {'materials-fig1-components': components(d), 'materials-fig2-checks': checks(d)}
    if 'onset' in d:
        from post_figures_onset import onset_figures
        figs.update(onset_figures(d['onset'], ROOT/'materials/data/onset-map/window-fields.npz',
                                  d.get('onset_ramp_sensitivity')))
    for name, fig in figs.items():
        width_in = check_type(fig, name, enforce=name in POST_FIGURES)
        fig.savefig(OUT/f'{name}.png', dpi=DISPLAY_PX/width_in, bbox_inches='tight', pad_inches=.3)
        plt.close(fig)
        print('wrote', OUT/f'{name}.png')


# Figures that appear in the post. The others illustrate the technical report only.
POST_FIGURES = {'materials-fig1-components', 'materials-fig2-checks', 'materials-fig3-window'}


def check_type(fig, name, pad=.3, enforce=True):
    """Smallest text size in px when the saved figure renders 700 px wide; post figures fail below MIN_PX_AT_700."""
    renderer = fig.canvas.get_renderer()
    width_in = fig.get_tightbbox(renderer).width + 2*pad
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    smallest = min(sizes)*700/(width_in*72)
    assert smallest >= MIN_PX_AT_700 or not enforce, f'{name}: smallest text {smallest:.1f} px at 700 px wide'
    print(f'{name}: {width_in:.2f} in wide, smallest text {smallest:.1f} px at 700 px')
    return width_in


if __name__ == '__main__':
    main()
