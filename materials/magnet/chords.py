"""Directional finite chords in periodic binary fields; a separate observable.

An infinite periodic line wholly inside the phase is reported as censored, never
as a box-length chord. Chord anisotropy is not a general particle aspect ratio.
"""
import numpy as np


def directional_chords(mask, spacing):
    mask=np.asarray(mask,bool)
    assert mask.ndim==2 and all(float(d)>0 for d in spacing)
    result=[]
    for axis,dx in enumerate(spacing):
        lines=np.moveaxis(mask,axis,0).T
        lengths=[];infinite_pixels=0
        for line in lines:
            if line.all():
                infinite_pixels+=len(line)
                continue
            changes=np.flatnonzero(line!=np.roll(line,1))
            if not len(changes):continue
            spans=np.diff(np.r_[changes,changes[0]+len(line)])
            lengths.extend(spans[line[changes]])
        pixels=np.asarray(lengths,int)
        starts=int(np.sum(mask & ~np.roll(mask,1,axis=axis)))
        assert len(pixels)==starts
        assert int(pixels.sum())+infinite_pixels==int(mask.sum())
        nm=pixels*dx
        result.append({'axis':axis,'n_finite_chords':len(pixels),
            'mean_finite_chord_nm':float(nm.mean()) if len(nm) else None,
            'quantiles_nm':np.quantile(nm,[.1,.5,.9]).tolist() if len(nm) else None,
            'phase_fraction_on_infinite_lines':infinite_pixels/max(1,int(mask.sum()))})
    means=[r['mean_finite_chord_nm'] for r in result]
    valid=all(v is not None for v in means) and not any(r['phase_fraction_on_infinite_lines'] for r in result)
    return {'phase_fraction':float(mask.mean()),'directions':result,
        'uncensored_chord_ratio':means[0]/means[1] if valid else None,
        'interpretation':'Finite chord statistics, with explicit infinite-line phase fractions. The ratio equals the geometric aspect for isolated identical axis-aligned ellipses in the continuum limit; it is otherwise an orientation descriptor, not a particle fit or magnetic closure.'}


def controls():
    spacing=.25
    x=(np.arange(512)-256)[:,None]*spacing
    y=(np.arange(256)-128)[None,:]*spacing
    ellipse=(x/30)**2+(y/10)**2<1
    circle=(x/10)**2+(y/10)**2<1
    stripe=np.broadcast_to(abs(y)<10,ellipse.shape)
    masks={'ellipse':ellipse,'translated_ellipse':np.roll(ellipse,241,axis=0),
        'transposed_ellipse':ellipse.T,'circle':circle,'percolating_stripe':stripe}
    rows={name:directional_chords(mask,(spacing,spacing)) for name,mask in masks.items()}
    assert abs(rows['ellipse']['uncensored_chord_ratio']-3)<.04
    assert rows['ellipse']==rows['translated_ellipse']
    assert abs(rows['ellipse']['uncensored_chord_ratio']*rows['transposed_ellipse']['uncensored_chord_ratio']-1)<1e-12
    assert rows['circle']['uncensored_chord_ratio']==1
    assert rows['percolating_stripe']['uncensored_chord_ratio'] is None
    assert rows['percolating_stripe']['directions'][0]['phase_fraction_on_infinite_lines']==1
    assert rows['percolating_stripe']['directions'][1]['mean_finite_chord_nm']==19.75
    return rows
