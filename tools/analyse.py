#!/usr/bin/env python3
"""Field Vision capture analyser.

Usage: python3 tools/analyse.py field-vision-YYYYMMDD-HHMMSS.json [--out DIR]

Prints a summary and writes (next to the file, or into --out):
  <name>_map.png       top-down map of sonar echoes (heading x range) + 3D particle snapshot projected flat
  <name>_profile.png   echo-profile heatmap: ping (time) vs range, in dB above noise floor
  <name>_spectrum.png  spectrum over time (64 log bins) with heading trace
"""
import argparse, json, math, os, sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": "#ddd", "axes.labelcolor": "#ddd", "xtick.color": "#aaa", "ytick.color": "#aaa",
                     "axes.edgecolor": "#555", "font.family": "monospace", "font.size": 9})


def load(path):
    with open(path) as f:
        d = json.load(f)
    h = d.get("header") or {}
    if d.get("kind") == "diag":   # DIAG > DOWNLOAD JSON (v2.5+): no REC frames, last ~150 pings, DIAG text
        h = dict(h, diagText=d.get("diag", ""), durationS=h.get("durationS") or 0)
    return h, d.get("frames", []), d.get("pings", []), d.get("particles", [])


def summary(h, frames, pings, parts):
    L = []
    a, s, o = h.get("audio", {}), h.get("sonar", {}), h.get("orientation", {})
    L.append(f"Field Vision {h.get('version')} capture {h.get('started')}  ({h.get('durationS')} s{' DEMO' if h.get('demo') else ''})")
    L.append(f"Device: {h.get('ua')}")
    L.append(f"Audio: {a.get('state')} {a.get('sampleRate')} Hz, channels {a.get('channels')}, mic {a.get('mic')}")
    L.append(f"Orientation: {o.get('source')}  front offset {o.get('frontOffsetDeg')} deg  absolute={o.get('absolute')}")
    L.append(f"Sonar: {'ON' if s.get('on') else 'off'} {s.get('band')} {s.get('f0')}-{s.get('f1')} Hz, chirp {s.get('chirpMs')} ms, "
             f"vol {s.get('vol')} ({s.get('gain')}), sens {s.get('sens')}, latency {s.get('latencyMs')} ms, test: {s.get('test')}")
    if frames:
        hd = np.array([f[1] for f in frames], float)
        rms = np.array([f[6] for f in frames if f[6] is not None], float)
        cover = len(set((hd // 10).astype(int) % 36)) * 10
        L.append(f"Frames: {len(frames)} (~{len(frames)/max(h.get('durationS',1),1):.0f}/s). Heading swept over ~{cover} of 360 deg "
                 f"(range {hd.min():.0f}-{hd.max():.0f}). Elev {min(f[2] for f in frames):.0f}..{max(f[2] for f in frames):.0f} deg.")
        if len(rms):
            L.append(f"Mic RMS: median {np.median(rms):.1f} dBFS, max {rms.max():.1f} dBFS")
        bands = np.array([f[7:11] for f in frames], float)
        L.append("Passive band level above floor, mean/max dB: " + ", ".join(
            f"{n} {bands[:,i].mean():.1f}/{bands[:,i].max():.1f}" for i, n in enumerate(["LOW", "MID", "HIGH", "ULTRA"])))
        dom = [f[11] for f in frames if f[11]]
        if dom:
            L.append(f"Dominant frequency: median {np.median(dom):.0f} Hz (present in {len(dom)}/{len(frames)} frames)")
    if pings:
        ok = [p for p in pings if p["ok"]]
        reasons = Counter(p["reason"] for p in pings if not p["ok"])
        L.append(f"Sonar pings: {len(pings)} total, {len(ok)} processed, rejected: " + (", ".join(f"{k} x{v}" for k, v in reasons.items()) or "none"))
        dfl = [p["direct"] / p["floor"] for p in pings if p.get("direct") and p.get("floor")]
        if dfl:
            L.append(f"Phone hears itself (direct/floor): median {np.median(dfl):.1f}x")
        best = [p["best"] for p in ok if p.get("best") is not None]
        if best:
            L.append(f"Best echo per ping (x floor): median {np.median(best):.1f}, max {max(best):.1f}; threshold median {np.median([p['thr'] for p in ok if p.get('thr')] or [0]):.1f}")
        pk = [(p["hdg"], r, snr) for p in ok for r, snr, *_ in p["peaks"]]
        plotted = sum(1 for p in ok if p.get("plotted"))
        L.append(f"Echo peaks found: {len(pk)} in {sum(1 for p in ok if p['peaks'])} pings; pings with plotted echoes: {plotted}")
        if pk:
            rs = np.array([x[1] for x in pk])
            L.append(f"Echo ranges: min {rs.min():.2f} m, median {np.median(rs):.2f} m, max {rs.max():.2f} m")
            sectors = Counter(int(x[0] // 30) * 30 for x in pk)
            near = {}
            for hh, r, _ in pk:
                k = int(hh // 30) * 30
                near[k] = min(near.get(k, 9), r)
            L.append("Nearest echo per 30 deg sector: " + ", ".join(f"{k:03d}deg {near[k]:.2f} m (n={sectors[k]})" for k in sorted(near)))
    cnts = [p["cnt"] for p in pings if p.get("cnt")]
    if cnts:   # v2.5+: where candidate echoes die
        tot = Counter()
        for c in cnts:
            tot.update(c)
        rep = sum(1 for p in pings for pk in p.get("peaks", []) if len(pk) > 2 and not pk[2])
        L.append("Peak fates (v2.5): " + ", ".join(f"{k} {tot[k]}" for k in ["found", "ripple", "thr", "shape", "gate", "code", "ok"]) + f", failed repeat {rep}")
        cand = [c for p in pings for c in p.get("cands", [])]
        if cand:
            L.append("Candidate fate by range (0.5 m bins): " + "; ".join(
                f"{b/2:.1f}-{b/2+0.5:.1f} m: " + ",".join(f"{w} {n}" for w, n in Counter(c[2] for c in cand if int(c[0] * 2) == b).most_common())
                for b in sorted(set(int(c[0] * 2) for c in cand))))
    cl = h.get("clusters") or []
    if cl:   # v2.6+: objects (merged echo clusters) at STOP
        cl = sorted(cl, key=lambda c: -c.get("n", 0))
        L.append(f"Objects at STOP: {len(cl)}; strongest: " + ", ".join(f"{c['hdg']:.0f}deg {c['r']:.2f} m (x{c['n']})" for c in cl[:8]))
    if h.get("rollingKeepS") and h.get("droppedOld"):
        L.append(f"Note: rolling buffer kept the last {h['rollingKeepS']} s; {h['droppedOld']} older frames/pings were dropped")
    if h.get("diagEnd"):
        h = dict(h, diagText=h["diagEnd"])
    if h.get("diagText"):
        L.append("--- DIAG at export ---\n" + h["diagText"])
    if parts:
        t = Counter(p[4] for p in parts)
        L.append(f"Particle snapshot: {len(parts)} points ({t.get(0,0)} passive, {t.get(1,0)} sonar)")
    return "\n".join(L)


def plot_map(h, pings, parts, out):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    maxr = h.get("sonar", {}).get("maxR", 4) or 4
    for r in range(1, int(maxr) + 1):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#333", lw=0.7, ls="--"))
        ax.text(r * 0.71, -r * 0.71, f"{r} m", color="#777", fontsize=7)
    # passive particles (3D snapshot flattened, band shells are not distances: drawn faint)
    pp = np.array([p for p in parts if p[4] == 0], float) if parts else np.zeros((0, 5))
    if len(pp):
        sc = maxr / 2.85
        ax.scatter(pp[:, 0] * sc, -pp[:, 2] * sc, s=0.3, c="#555", alpha=0.5, lw=0, label="passive (band shells, not distance)")
    xs, ys, cs = [], [], []
    for p in pings:
        if not p["ok"]:
            continue
        th = math.radians(p["hdg"])
        for r, snr, *_ in p["peaks"]:
            xs.append(r * math.sin(th)); ys.append(r * math.cos(th)); cs.append(min(1, math.log10(max(snr, 1)) / 3))
    if xs:
        ax.scatter(xs, ys, s=6, c=np.array(cs), cmap="gray", vmin=0, vmax=1, lw=0, label="sonar echo peaks")
    sp = np.array([p for p in parts if p[4] == 1], float) if parts else np.zeros((0, 5))
    if len(sp):
        sc = maxr / 2.85
        ax.scatter(sp[:, 0] * sc, -sp[:, 2] * sc, s=1.2, c="#fff", alpha=0.6, lw=0, label="sonar particles (snapshot)")
    for p in pings[::3]:
        th = math.radians(p["hdg"]); ax.plot([0, 0.25 * math.sin(th)], [0, 0.25 * math.cos(th)], color="#444", lw=0.5)
    ax.plot(0, 0, "o", color="white", ms=5)
    ax.set_xlim(-maxr * 1.05, maxr * 1.05); ax.set_ylim(-maxr * 1.05, maxr * 1.05); ax.set_aspect("equal")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)  [heading 0 = up]")
    ax.set_title("Top-down echo map", color="#eee")
    if xs or len(pp) or len(sp):
        ax.legend(loc="upper right", fontsize=6, facecolor="black", edgecolor="#444", labelcolor="#ccc", markerscale=4)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_profile(pings, out):
    prof = [p for p in pings if p.get("profile")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    if not prof:
        ax.text(0.5, 0.5, "no sonar echo profiles in this capture\n(sonar off, or every ping rejected before the echo stage)",
                ha="center", va="center", color="#bbb", transform=ax.transAxes)
    else:
        M = np.array([p["profile"] for p in prof], float)
        D = 20 * np.log10(np.maximum(M, 1e-3))
        t = [p["t"] for p in prof]
        im = ax.imshow(D.T, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=max(20, np.percentile(D, 99.5)),
                       extent=[0, len(prof), 0, M.shape[1] * 0.01])
        for i, p in enumerate(prof):
            for r, *_ in p["peaks"]:
                ax.plot(i + 0.5, r, marker="_", color="white", ms=5, mew=1)
        cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("dB above noise floor"); cb.ax.yaxis.set_tick_params(color="#aaa")
        ax.set_xlabel(f"ping # ({t[0]:.1f}-{t[-1]:.1f} s)"); ax.set_ylabel("range (m)")
        ax2 = ax.twiny(); ax2.set_xlim(0, len(prof))
        ticks = list(range(0, len(prof), max(1, len(prof) // 8)))
        ax2.set_xticks([i + 0.5 for i in ticks]); ax2.set_xticklabels([f"{prof[i]['hdg']:.0f}°" for i in ticks], fontsize=7)
        ax2.set_xlabel("heading at ping", fontsize=8)
    ax.set_title("Echo profile (white ticks = detected peaks)", color="#eee", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_spectrum(h, frames, out):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7, 5), dpi=130, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    if frames:
        t = np.array([f[0] for f in frames]); S = np.array([f[12] for f in frames], float)
        hz = h.get("legend", {}).get("spec64Hz") or list(range(S.shape[1]))
        S = np.where(S <= -159, np.nan, S)
        im = ax.imshow(S.T, aspect="auto", origin="lower", cmap="gray", extent=[t[0], t[-1] if len(t) > 1 else t[0] + 0.1, 0, S.shape[1]],
                       vmin=np.nanpercentile(S, 5), vmax=np.nanpercentile(S, 99.5))
        idx = [i for i in range(0, len(hz), 8)]
        ax.set_yticks([i + 0.5 for i in idx]); ax.set_yticklabels([f"{hz[i]/1000:.1f}k" if hz[i] >= 1000 else str(hz[i]) for i in idx])
        ax.set_ylabel("frequency (Hz)")
        cb = fig.colorbar(im, ax=[ax, axh], pad=0.01); cb.set_label("dBFS")
        axh.plot(t, [f[1] for f in frames], color="#ddd", lw=0.8); axh.set_ylabel("heading°"); axh.set_ylim(0, 360); axh.set_yticks([0, 90, 180, 270, 360])
        axh.set_xlabel("time (s)")
    ax.set_title("Spectrum over time (black gaps = sonar band masked)", color="#eee", fontsize=9, loc="left")
    fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    h, frames, pings, parts = load(a.file)
    out = a.out or os.path.dirname(os.path.abspath(a.file)); os.makedirs(out, exist_ok=True)
    base = os.path.join(out, os.path.splitext(os.path.basename(a.file))[0])
    print(summary(h, frames, pings, parts))
    plot_map(h, pings, parts, base + "_map.png")
    plot_profile(pings, base + "_profile.png")
    plot_spectrum(h, frames, base + "_spectrum.png")
    print(f"\nWrote {base}_map.png, {base}_profile.png, {base}_spectrum.png")


if __name__ == "__main__":
    main()
