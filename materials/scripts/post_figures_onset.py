"""Onset-study figures: route map by world, and production utility by observation policy.

Input is the analyze_onset_study.py report embedded in post-data.json. Style
constants come from post_figures.py.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from post_figures import BLUE, FIELD, INK, MUTED, NEUTRAL, SOFT, WARM, grid

# The faulty term (the base model's first-attempt error) is drawn in the base-model color. The onset is neutral so
# no figure pairs the warm orange with a green.
VIOLET, TEAL = WARM, NEUTRAL
RULE_C = '#ded8cd'
GREEN = TEAL


def route_map(o):
    worlds = [tuple(w) for w in o['worlds']]
    routes = o['routes']
    sweeps = [r for r in routes if r.startswith('sweep')]
    Ts = np.array([float(r[5:]) for r in sweeps])
    offsets = sorted({w[0] for w in worlds})
    kinetics = sorted({w[1] for w in worlds})
    F, P, E = (np.array(o[k]) for k in ('fit_mean', 'production_mean', 'erroneous_fit'))
    fig, axes = plt.subplots(len(kinetics), len(offsets), figsize=(4.3*len(offsets), 3.1*len(kinetics)),
                             sharex=True, sharey=True, squeeze=False)
    ymax = np.nanmax(np.r_[F.ravel(), P.ravel(), E.ravel()])*1.12
    for i, w in enumerate(worlds):
        ax = axes[kinetics.index(w[1]), offsets.index(w[0])]
        idx = [routes.index(r) for r in sweeps]
        j = routes.index('ramp10Kph')
        ax.axhspan(min(F[i, j], P[i, j]), max(F[i, j], P[i, j])+1e-9, color=MUTED, alpha=.18, lw=0)
        ax.axhline(F[i, j], color=SOFT, lw=1.2, ls=':')
        ax.plot(Ts, F[i, idx], '-o', color=INK, lw=2.2, ms=4)
        ax.plot(Ts, P[i, idx], 'o', mfc='none', mec=INK, ms=6, alpha=.7)
        ax.plot(Ts, E[i, idx], '-o', color=VIOLET, lw=1.6, ms=3.5, alpha=.85)
        on = o['linear_onset_K'][str(w[0])]
        ax.axvline(on, color=GREEN, lw=1.3)
        ax.axvline(o['onset_omitted_K'], color=VIOLET, lw=1, ls='--', alpha=.7)
        ax.set_ylim(0, ymax)
        grid(ax)
        if kinetics.index(w[1]) == 0:
            ax.set_title(f'Curie offset {w[0]:+g} K'.replace('-', '−'), color=SOFT, fontsize=12)
        if offsets.index(w[0]) == 0:
            ax.set_ylabel(f'mobility × {w[1]:g}\nenergy product (kJ/m³)')
        if kinetics.index(w[1]) == len(kinetics)-1:
            ax.set_xlabel('Sweep start temperature (K)')

    handles = [mpl.lines.Line2D([], [], color=INK, lw=2.2, marker='o', label='Verified simulator (fit specimens)'),
               mpl.lines.Line2D([], [], color=INK, lw=0, marker='o', mfc='none', label='Production specimens'),
               mpl.lines.Line2D([], [], color=VIOLET, lw=1.6, marker='o', label="Simulator with the base model's first-attempt error"),
               mpl.lines.Line2D([], [], color=SOFT, lw=1.2, ls=':', label='Fast ramp, 10 K/h, fit specimens'),
               mpl.patches.Patch(color=MUTED, alpha=.18, lw=0, label='Fast ramp, fit to production'),
               mpl.lines.Line2D([], [], color=GREEN, lw=1.3, label='In-field onset'),
               mpl.lines.Line2D([], [], color=VIOLET, lw=1, ls='--', label='Onset without the Zeeman term')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .1, 1, 1))
    return fig


def policies(o, sens=None):
    """Net gain over the verified simulator without measurement, per observation, for the frozen map and the
    pre-registered ramp corrections. 95% intervals resample production specimens, paired across policies."""
    obs = ['none', 'kinetics', 'curie', 'onset', 'pilot']
    names = {'none': 'No measurement', 'kinetics': 'Measure kinetics', 'curie': 'Measure Curie temperature',
             'onset': 'Measure in-field onset', 'pilot': 'Two-route pilot, no simulator'}
    panels = [('Frozen map', o['policies'])]
    if sens:
        panels += [('Ramp + 1.5 (measured bias)', sens['1.5']['policies']), ('Ramp + 3', sens['3.0']['policies'])]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4.3), sharey=True, squeeze=False)
    for ax, (title, pol) in zip(axes[0], panels):
        ref = pol['correct|none']['net_utility']
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
        for sim, color, dy, label in [('correct', INK, -.13, 'Verified simulator'),
                                      ('erroneous', VIOLET, .13, "Simulator with the base model's first-attempt error")]:
            for row, b in enumerate(obs):
                if b == 'pilot' and sim == 'erroneous':
                    continue
                v = pol[f'{sim}|{b}']
                c, y = (MUTED, row) if b == 'pilot' else (color, row+dy)
                lo, hi = v['net_gain_vs_verified_none_ci95']
                ax.plot([lo, hi], [y]*2, color=c, lw=2.2, alpha=.45, solid_capstyle='round')
                ax.plot(v['net_utility']-ref, y, 'o', color=c, ms=6.5, label=label if row == 0 else None)
        ax.set_title(title, loc='left', fontsize=11.5, color=SOFT, pad=10)
        ax.set_xlim(-9, 1)
        ax.set_xticks([-8, -6, -4, -2, 0])
        ax.xaxis.grid(True, color=RULE_C, lw=.8)
        ax.set_axisbelow(True)
    axes[0, 0].set_yticks(range(len(obs)))
    axes[0, 0].set_yticklabels([names[b] for b in obs])
    axes[0, 0].invert_yaxis()
    axes[0, len(panels)//2].set_xlabel('Gain over the verified simulator without measurement (kJ/m³)')
    axes[0, 0].legend(frameon=False, loc='lower left', bbox_to_anchor=(-.04, 1.1), ncol=2, fontsize=10.5)
    fig.subplots_adjust(left=2.75/fig.get_figwidth(), right=.985, top=.8, bottom=.17, wspace=.16)
    return fig


def onset_figures(o, fields_folder, sens=None):
    return {'materials-fig3-window': window(o, fields_folder), 'materials-fig4-measurements': policies(o, sens),
            'materials-route-map': route_map(o), 'materials-field-stage': field_stage(o)}


def field_stage(o):
    """Contrast grown during the four field hours against sweep start, one line per kinetics, by Curie offset."""
    worlds = [tuple(w) for w in o['worlds']]
    routes = o['routes']
    sweeps = [r for r in routes if r.startswith('sweep')]
    Ts = np.array([float(r[5:]) for r in sweeps])
    idx = [routes.index(r) for r in sweeps]
    C = np.array(o['field_stage_end']['contrast_at_4h'])
    U = np.array(o['production_mean'])
    offsets = sorted({w[0] for w in worlds})
    kinetics = sorted({w[1] for w in worlds})
    shades = dict(zip(kinetics, ['#b9b2a8', '#7d756b', INK][-len(kinetics):]))
    fig, axes = plt.subplots(2, len(offsets), figsize=(4.3*len(offsets), 6.4), sharex=True, sharey='row', squeeze=False)
    for i, w in enumerate(worlds):
        col = offsets.index(w[0])
        for row, M in enumerate([C, U]):
            ax = axes[row, col]
            ax.plot(Ts, M[i, idx], '-o', color=shades[w[1]], lw=2, ms=4, label=f'mobility × {w[1]:g}')
            ax.axvline(o['linear_onset_K'][str(w[0])], color=TEAL, lw=1.2)
            grid(ax)
        axes[0, col].set_title(f'Curie offset {w[0]:+g} K'.replace('-', '−'), color=SOFT, fontsize=12)
    axes[0, 0].set_ylabel('Composition contrast\nat the end of the field')
    axes[1, 0].set_ylabel('Synthetic energy product\n(production, kJ/m³)')
    axes[1, 0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    for ax in axes[1]:
        ax.set_xlabel('Sweep start temperature (K)')
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def window(o, folder, world=(-5.0, 1.0), picks=('sweep929', 'sweep941')):
    """Teaser for the decision: (a) contrast grown during the field, (b) final utility, by sweep start and mobility,
    averaged over Curie offsets, with the band of in-field onsets; (c) final fields for a cold and an onset start."""
    worlds = [tuple(w) for w in o['worlds']]
    routes = o['routes']
    sweeps = [r for r in routes if r.startswith('sweep')]
    Ts = np.array([float(r[5:]) for r in sweeps])
    idx = [routes.index(r) for r in sweeps]
    P = np.array(o['production_mean'])
    C = np.array(o['field_stage_end']['contrast_at_4h'])
    kinetics = sorted({w[1] for w in worlds})
    shades = dict(zip(kinetics, ['#c9c2b8', '#8a8178', INK][-len(kinetics):]))
    onsets = [o['linear_onset_K'][k] for k in o['linear_onset_K']]
    fig = plt.figure(figsize=(10.4, 4.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, .62], wspace=.42, hspace=.36)
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
    for ax, M, ylabel in [(axes[0], C, 'Composition contrast after\nfour hours in the field'),
                          (axes[1], P, 'Synthetic energy product,\nproduction specimens (kJ/m³)')]:
        ax.axvspan(min(onsets), max(onsets), color=NEUTRAL, alpha=.22, lw=0)
        for k in kinetics:
            rows = [i for i, w in enumerate(worlds) if w[1] == k]
            ax.plot(Ts, M[rows][:, idx].mean(0), '-o', color=shades[k], lw=2.4 if k == 1 else 2, ms=4.5)
        ax.set_xlim(Ts[0]-(6.5 if ax is axes[1] else 2.5), Ts[-1]+1.5)
        ax.set_xticks(np.arange(930, Ts[-1]+1, 5))
        ax.spines['bottom'].set_bounds(Ts[0], Ts[-1])
        ax.set_xlabel('Slow cooling starts at (K)')
        ax.set_ylabel(ylabel)
        grid(ax)
    top = axes[0].get_ylim()[1]
    axes[0].annotate('onset under\nthe field', ((min(onsets)+max(onsets))/2, top), xytext=(0, -4),
                     textcoords='offset points', ha='center', va='top', color=MUTED, fontsize=12)
    axes[1].set_ylim(0, 12.8)
    axes[1].set_yticks(range(0, 11, 2))
    floor = C[:, idx[-1]].mean()
    axes[0].annotate('thermal noise only', (Ts[-1], floor), xytext=(-2, -7), textcoords='offset points', ha='right',
                     va='top', color=MUTED, fontsize=12)
    axes[0].set_ylim(floor-.016, None)
    # Direct labels at the cold end, where the three mobilities separate most.
    for k in kinetics:
        rows = [i for i, w in enumerate(worlds) if w[1] == k]
        m = P[rows][:, idx].mean(0)
        axes[1].annotate(f'× {k:g}', (Ts[0], m[0]), xytext=(-7, 0), textcoords='offset points', ha='right',
                         va='center', color=SOFT if k == max(kinetics) else MUTED, fontsize=12)
    top_k = max(kinetics)
    m = P[[i for i, w in enumerate(worlds) if w[1] == top_k]][:, idx].mean(0)
    axes[1].annotate('mobility', (Ts[0], m[0]), xytext=(-7, 16), textcoords='offset points', ha='right',
                     va='center', color=MUTED, fontsize=12)
    saved = np.load(folder)  # window-fields.npz: final fields of production specimen 801, nominal world
    for row, r in enumerate(picks):
        cx = fig.add_subplot(gs[row, 2])
        m = [str(x) for x in saved['routes']].index(r)
        L = float(saved['L_nm'])
        cx.imshow(saved['fields'][m].T, origin='lower', cmap=FIELD, vmin=.15, vmax=.85, interpolation='nearest',
                  extent=(0, L, 0, L))
        cx.set_xticks([])
        cx.set_yticks([])
        for sp in cx.spines.values():
            sp.set_visible(True)
            sp.set_color(RULE_C)
        cx.set_title(f'start at {float(r[5:]):g} K', color=SOFT, fontsize=12)
        if row == len(picks)-1:
            cx.plot([0, 100], [-.06*L]*2, color=INK, lw=2, clip_on=False, solid_capstyle='butt')
            cx.text(50, -.1*L, '100 nm', ha='center', va='top', fontsize=12, color=SOFT)
            cx.text(L, -.1*L, 'field →', ha='right', va='top', fontsize=12, color=MUTED)
        cx.set_xlim(0, L)
        cx.set_ylim(0, L)
    for ax, letter in zip(axes+[fig.axes[2]], 'abc'):
        ax.text(-.02 if ax in axes else 0, 1.06 if ax in axes else 1.2, letter, transform=ax.transAxes, fontsize=13,
                fontweight='bold', color=SOFT, ha='left')
    return fig