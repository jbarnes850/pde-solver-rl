"""Static detectors for recurring implementation bugs in spectral programs (all policies), validated by
P(strict | flagged): a detector that fires on strict programs is noisy.

kscale  : wavenumbers built as fftfreq(N) / fftfreq(N, 1.0) / 2*pi*arange(N)/N / linspace(0, 2pi, N) without a
          factor N or /dx (on [0,1) the angular wavenumber is 2*pi*fftfreq(N, d=1/N))
imagdiff: diffusion symbol multiplied by 1j inside exp (heat treated as Schroedinger)
compound: state multiplied by an absolute-time propagator exp(.. * t) inside the output loop
possign : advection phase with +1j*beta (wrong direction for numpy's FFT convention)
fft1d_axes: 1-D fft/ifft called with axes= or a tuple axis (multi-D FFT API confusion)
"""
import collections
import json
import re
from pathlib import Path

S = Path(__file__).resolve().parent


def strip(code):
    return [l.split("#", 1)[0].rstrip() for l in code.splitlines()]


NVAR = r"(?:N\w*|n\w*|len\([^)]*\)|\w+\.shape\[-?\d\]|M\w*)"


def kscale(lines):
    s = "\n".join(lines)
    for l in lines:
        m = re.search(r"fftfreq\(\s*" + NVAR + r"\s*(?:,\s*(?:d\s*=\s*)?(1(?:\.0*)?|L\w*|1\.0\s*/\s*1(?:\.0)?)\s*)?\)", l)
        if m and m.group(1) is not None and m.group(1).startswith("L"):
            if not re.search(r"\b" + m.group(1) + r"\s*=\s*1(\.0*)?\b", s):
                continue
        if m:
            if re.search(r"\*\s*N\w*|N\w*\s*\*|/\s*dx|/\s*d[xy]\b|/\s*h\b|/\s*\(\s*1(\.0)?\s*/\s*N", l):
                continue
            var = re.match(r"\s*(\w+)\s*=", l)
            if var and re.search(r"\b" + var.group(1) + r"\s*(\*=\s*N|=\s*" + var.group(1) + r"\s*\*\s*N|/=\s*d)", s):
                continue
            return True
        if re.search(r"2(\.0)?\s*\*\s*np\.pi\s*\*\s*np\.arange\(\s*N\w*\s*\)\s*/\s*N", l) or \
                re.search(r"\bk\w*\s*=\s*np\.linspace\(\s*0(\.0)?\s*,\s*2(\.0)?\s*\*\s*np\.pi", l):
            return True
    return False


def imagdiff(lines):
    for l in lines:
        if "exp(" in l and re.search(r"1j\s*\*[^+\n]*?(\*\*\s*2|k_?sq\w*|\bk2\w*|\bK2\w*|ksq|lap\w*)", l) and \
                not re.search(r"1j\s*\*[^+\-\n]*?beta", l):
            if not re.search(r"1j\s*\*\s*\(?\s*beta|beta\w*\s*\*\s*1j", l):
                return True
    return False


def compound(lines):
    for l in lines:
        if re.search(r"(\w+)\s*=\s*\1\s*\*\s*[^\n]*exp\([^\n]*\b(t|t_target|t_val|t_now|time|t_n|t_cur\w*)\b(?!\s*-)", l) and "dt" not in l:
            return True
        if re.search(r"(\w+)\s*\*=\s*[^\n]*exp\([^\n]*\b(t|t_target|t_val|t_now|time|t_n)\b(?!\s*-)", l) and "dt" not in l:
            return True
    return False


def possign(lines):
    for l in lines:
        if "exp(" in l and "beta" in l and re.search(r"exp\(\s*\(?\s*\+?\s*1j\s*\*", l):
            return True
    return False


def fft1d_axes(lines):
    s = "\n".join(lines)
    return bool(re.search(r"(?<![\w.])i?fft\(\s*[^\n]*axes\s*=|(?<![\w.])i?fft\(\s*[^\n]*axis\s*=\s*\(", s) or
                re.search(r"\.i?fft\(\s*[^\n]*axes\s*=", s))


DET = {"kscale": kscale, "imagdiff": imagdiff, "compound": compound, "possign": possign, "fft1d_axes": fft1d_axes}


def main():
    rows = [json.loads(l) for l in open(S / "rows.jsonl")]
    m = {(o["policy"], o["case"], o["idx"]): o for o in map(json.loads, open(S / "methods.jsonl"))}
    LIN = ["heat1d", "advection_diffusion1d", "advection2d", "advection_diffusion2d"]
    allfam = sorted({r["family"] for r in rows})
    tab = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r["policy"] not in ("ours160", "ours128", "base", "qwen38", "glm", "mimo"):
            continue
        L = strip(r["code"])
        flags = {k: f(L) for k, f in DET.items()}
        key = (r["policy"], r["family"])
        tab[key]["n"] += 1
        tab[key]["strict"] += r["outcome"] == "strict"
        for k, v in flags.items():
            if v:
                tab[key][k] += 1
                tab[key][k + "_strict"] += r["outcome"] == "strict"
                tab[key][k + "_" + r["outcome"]] += 1
    for p in ("base", "ours128", "ours160", "qwen38", "glm", "mimo"):
        print("==", p)
        tot = collections.Counter()
        for f in allfam:
            c = tab[(p, f)]
            tot.update(c)
            print(f"  {f:32s} n={c['n']:3d} strict={c['strict']:3d} " + "  ".join(
                f"{k} {c[k]}(s{c[k + '_strict']})" for k in DET))
        print(f"  {'ALL':32s} n={tot['n']} " + "  ".join(f"{k} {tot[k]}(s{tot[k + '_strict']})" for k in DET))
    json.dump({f"{p}|{f}": dict(c) for (p, f), c in tab.items()}, open(S / "bugs.json", "w"), indent=1)


if __name__ == "__main__":
    main()


def possign2(lines):
    s = "\n".join(lines)
    for m in re.finditer(r"1j\s*\*\s*(?:\(\s*)?(?:beta\w*|k\w*\s*\*\s*beta\w*|K\w*\s*\*\s*beta\w*)|beta\w*\s*\*\s*1j", s):
        pre = s[:m.start()].rstrip()
        if pre.endswith("-"):
            continue
        if pre.endswith("(") and pre[:-1].rstrip().endswith("-"):
            continue
        if pre.endswith("("):
            # (1j*beta*k - nu k^2) * t with no leading minus -> wrong; but -( ... ) handled above
            pass
        line = s[s.rfind("\n", 0, m.start()) + 1: s.find("\n", m.end())]
        if re.search(r"ifft|conj|exp\(\s*-", line):
            continue
        return True
    return False


def double2pi(lines):
    s = "\n".join(lines)
    for m in re.finditer(r"^\s*(\w+)\s*=\s*[^\n]*(2(\.0)?\s*\*\s*np\.pi[^\n]*fftfreq|fftfreq[^\n]*2(\.0)?\s*\*\s*np\.pi)", s, flags=re.M):
        v = m.group(1)
        if re.search(r"2(\.0)?\s*\*\s*np\.pi\s*\*\s*" + v + r"\b|\b" + v + r"\s*\*\s*2(\.0)?\s*\*\s*np\.pi", s[m.end():]):
            return True
    return False


DET.update({"possign2": possign2, "double2pi": double2pi})
if __name__ == "__main__":
    pass
