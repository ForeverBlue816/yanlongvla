"""Figure v3 (user 2026-10-28; v2 = user 2026-10-27, "Nature level"): fig_readout_sensitivity_v2.{pdf,png} + three supplementary PNGs + JSON of every printed
number. (a) GR00T N1.7 readout sensitivity per reader block x flow step (FP head, B-AM Ω); (b) G_out DoF-energy shares per module with
the top-10%-row concentration; (c) pi0.5-LIBERO expert cross-attention of reader block 0 at the last flow step over the agent-view
image tokens for FP / 2-bit PTQ / ours [v2, dropped in v3: at expert layer 0, last step, only 7% of the attention reaches the third-person image and the
map is near-uniform, so the three maps are indistinguishable]. v3 (c1): token-group attention of the expert's first reader layer at the last flow step,
mean over the 256 held-out observations (same latent per observation across models), paired difference of the language share vs FP with a 95%
bootstrap interval; (c2, optional): the image-reading layer (largest FP third-person-image mass over the 256 calibration observations), FP map and
|A_2bit - A_FP|, |A_ours - A_FP| on the task-260 frame, only if that layer's image mass is >= 20%. Font: TeX Gyre Termes (Times-compatible); mathtext stix."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path('/projects/yanlongvla')
D = json.loads((ROOT / 'runs/plustrain_20261008/groot/ptq/amvq/diagnostics.json').read_text())
CAP = ROOT / 'runs/plustrain_20261008/diagnostics/fig_c'
OUT = ROOT / 'research/figures'
NAME = 'fig_readout_sensitivity_v3'
C1 = Path(os.environ.get('FIG_C1_DIR', ROOT / 'runs/plustrain_20261008/diagnostics/fig_c1'))
MODEL_COL = {'fp': '#2F2F2F', 'ptq2bit': '#A7AEB9', 'ours': '#5B93C9'}
GROUP_COL = {'language': ('#2F2F2F', 0.85), 'base_image': ('#D8ECFA', 1.0), 'wrist_image': ('#EBD1E3', 1.0), 'action_tokens': ('#FFF4A8', 1.0)}
GROUP_LBL = {'language': 'language', 'base_image': 'third-person image', 'wrist_image': 'wrist image', 'action_tokens': 'action tokens'}

# ---- palette (user-fixed) and typography
PINK, BLUE, YELLOW, LGREY, MGREY, DARK, BG, VALGREY = '#EBD1E3', '#D8ECFA', '#FFF4A8', '#E5E8F0', '#A7AEB9', '#2F2F2F', '#F7F9F6', '#707070'
STEP = ['#D8ECFA', '#9FC5EA', '#5B93C9', '#2F2F2F']
CAT = {'translation': BLUE, 'rotation': PINK, 'gripper': YELLOW}
for f in sorted(Path.home().glob('.fonts/termes/*.otf')):
    font_manager.fontManager.addfont(str(f))
families = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ('Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral') if f in families))
plt.rcParams.update({'font.family': FONT, 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'mathtext.fontset': 'stix', 'text.color': DARK, 'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
                     'axes.edgecolor': LGREY, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
                     'axes.facecolor': BG, 'figure.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
BOX = dict(boxstyle='round,pad=0.3,rounding_size=0.4', fc='white', ec=LGREY, lw=0.5)
record = dict(version='v3', font=FONT, mathtext='stix', figure_inches=[7.2, 2.5], width_ratios=[1.3, 0.85, 0.95], wspace=0.3)


def letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom', color=DARK)


fig = plt.figure(figsize=(7.2, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 0.95], wspace=0.3, left=0.075, right=0.985, top=0.9, bottom=0.2)

# ================= (a) readout sensitivity per reader block x flow step
ax = fig.add_subplot(gs[0, 0])
rs = D['readout_sensitivity']
blocks = sorted(int(b) for b in rs['blocks']); kind = {int(b): k for b, k in rs['blocks'].items()}
vals = np.array([[rs['values'][f'{b}:{t}'] for b in blocks] for t in range(4)])   # (4 steps, 16 blocks)
x = np.arange(len(blocks))
for i, b in enumerate(blocks):
    if kind[b] == 'image':
        ax.add_patch(Rectangle((i - 0.5, 0), 1, 1, transform=ax.get_xaxis_transform(), color=LGREY, alpha=0.6, zorder=0, lw=0))
ax.set_yscale('log')
ax.grid(axis='y', color=LGREY, lw=0.5, zorder=0)
for t in range(4):
    ax.plot(x, vals[t], color=STEP[t], lw=1.2, marker='o', ms=3.0, mec=DARK, mew=0.4, zorder=3 + t, clip_on=False)
ends = np.log10(vals[:, -1]); order = np.argsort(ends); pos = ends.copy(); gap = 0.13
for q in range(1, 4):
    i, j = order[q - 1], order[q]
    if pos[j] - pos[i] < gap:
        pos[j] = pos[i] + gap
for t in range(4):
    ax.annotate(f'step {t}', xy=(x[-1], vals[t, -1]), xytext=(x[-1] + 0.35, 10 ** pos[t]), textcoords='data', va='center', ha='left', fontsize=7, color={0: '#7FA8CF', 1: '#5B93C9'}.get(t, STEP[t]),   # v3: step 0 / step 1 labels in a darker tone of their line colour
                annotation_clip=False, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks])
ax.set_xlabel('reader block')
ax.set_ylabel(r'action sensitivity of readout, $\|\Omega^{1/2} J_{v \leftarrow o}\|_F^2$')
ax.set_xlim(-0.6, len(blocks) + 1.2)   # v3: right margin trimmed from +1.7 to +1.2 (the end labels must stay clear of panel (b)'s y labels)
ax.set_ylim(vals.min() / 3.2, 5e-5)   # v3: fixed upper limit
ax.spines['bottom'].set_bounds(-0.6, len(blocks) - 0.4)
i0 = blocks.index(0)
ax.annotate('first text block:\nmost sensitive at every step', xy=(x[i0] + 0.2, vals[3, i0] * 1.12), xytext=(x[i0] + 2.2, 2.9e-5), fontsize=7, color=DARK, ha='left', va='center', linespacing=1.0,
            bbox=BOX, arrowprops=dict(arrowstyle='-', color=MGREY, lw=0.5, shrinkA=0, shrinkB=1))
growth = float(vals[3].mean() / vals[0].mean())
ax.text(0.98, 0.05, f'step 3 / step 0 = {growth:.1f}×', transform=ax.transAxes, fontsize=7, color=DARK, ha='right', va='bottom', bbox=BOX)
letter(ax, 'a')
record['a'] = dict(blocks=blocks, block_kind={str(b): kind[b] for b in blocks}, values={f'block{b}_step{t}': float(vals[t, i]) for i, b in enumerate(blocks) for t in range(4)},
                   step3_over_step0_mean_over_blocks=growth, top_block_step3=int(blocks[int(np.argmax(vals[3]))]), image_band_alpha=0.6,
                   image_band_meaning='grey band = reader block reads image tokens; white = text tokens', step_colors=STEP, label_colors=['#7FA8CF', '#5B93C9', STEP[2], STEP[3]], ylim=[float(vals.min() / 3.2), 5e-5])

# ================= (b) G_out DoF-energy shares per module + top-10% row concentration
ax2 = fig.add_subplot(gs[0, 1])
rows = [r for r in D['g_out'] if r['key'] in ('backbone', 'head_libero_spatial')]
mods = [('vision', 'vision tower'), ('llm', 'LLM 0–15'), ('vl_self_attention', 'VL self-attn'), ('dit', 'DiT')]
groups = ['translation', 'rotation', 'gripper']
shares = np.array([[np.mean([r['dof_share'][g] for r in rows if r['module'] == m]) for g in groups] for m, _ in mods])
counts = [sum(1 for r in rows if r['module'] == m) for m, _ in mods]
top10 = [float(np.median([r['top10pct'] for r in rows if r['module'] == m])) for m, _ in mods]
y = np.arange(len(mods))[::-1]; left = np.zeros(len(mods))
for gi, g in enumerate(groups):
    w = shares[:, gi]
    ax2.barh(y, w, left=left, height=0.6, color=CAT[g], edgecolor=MGREY, linewidth=0.4, label=g, zorder=3)
    for yi, (l, v) in enumerate(zip(left, w)):
        if v > 0.09:
            ax2.text(l + v / 2, y[yi], f'{100 * v:.0f}%', ha='center', va='center', fontsize=7, color=DARK, zorder=4)
    left = left + w
ax2.set_yticks(y); ax2.set_yticklabels([f'{lbl}\n({n})' for (_, lbl), n in zip(mods, counts)], linespacing=0.95)
ax2.set_xlim(0, 1); ax2.set_xticks([0, 0.5, 1.0]); ax2.set_xticklabels(['0', '50', '100%'])
ax2.set_xlabel(r'share of $G_{\mathrm{out}}$ energy')
ax2.spines['left'].set_visible(False); ax2.tick_params(axis='y', length=0)
ax2.grid(axis='x', color=LGREY, lw=0.5, zorder=0)
for yi, c in zip(y, top10):
    ax2.text(1.03, yi, f'{c:.2f}', transform=ax2.get_yaxis_transform(), ha='left', va='center', fontsize=7, color=VALGREY, clip_on=False)
ax2.text(1.03, 1.0, 'top-10%\nrows', transform=ax2.transAxes, ha='left', va='bottom', fontsize=6.5, color=VALGREY, linespacing=0.95)
ax2.set_ylim(-1.45, len(mods) - 0.45)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), frameon=False, ncol=3, handlelength=0.9, handleheight=0.7, columnspacing=0.7, handletextpad=0.35, borderaxespad=0)
letter(ax2, 'b')
record['b'] = dict(layers_counted='backbone + head_libero_spatial rows of diagnostics.json', layer_counts={m: n for (m, _), n in zip(mods, counts)},
                   dof_shares_mean={m: dict(zip(groups, map(float, shares[i]))) for i, (m, _) in enumerate(mods)},
                   top10pct_row_concentration_median={m: c for (m, _), c in zip(mods, top10)}, colors=CAT)

# ================= (c1) token-group attention of the expert's first reader layer, last flow step (256 held-out observations)
LABELS = [('fp', 'FP'), ('ptq2bit', '2-bit PTQ'), ('ours', 'ours')]
SEGS = ['language', 'base_image', 'wrist_image', 'action_tokens']
GI = dict(base_image=0, wrist_image=1, padding_image=2, language=3, action_tokens=4)   # column order of paper_fig_c1_capture.GROUPS
have_c1 = (C1 / 'capture.json').exists() and (C1 / 'groups.npz').exists()


def boot_ci(d, n=2000, seed=20260930):
    """paired bootstrap over observations of a per-observation difference d (same latent per observation across models); percentile 95% interval"""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (n, len(d))); means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_delta(m, lo, hi):
    return f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]'


if have_c1:
    capc = json.loads((C1 / 'capture.json').read_text()); G = np.load(C1 / 'groups.npz'); si = list(capc['steps']).index(9)
    H = {tag: np.asarray(G[f'{tag}_heldout'][:, si], np.float64) for tag, _ in LABELS}   # (256, 18, 5)
    N = H['fp'].shape[0]; assert all(v.shape[0] == N for v in H.values())
    assert max(float(v[..., GI['padding_image']].max()) for v in H.values()) < 1e-6, 'padding-image tokens received attention'
    LAYER = 0
    slot_c = gs[0, 2]
    # (c2) layer selection (FP only, calibration set, last step): the layer with the largest mean third-person-image mass
    FPC = np.asarray(G['fp_calibration'][:, si], np.float64)                          # (256, 18, 5)
    img_by_layer = FPC[..., GI['base_image']].mean(0); sel_layer = int(np.argmax(img_by_layer)); sel_mass = float(img_by_layer[sel_layer])
    have_c2 = sel_mass >= 0.20 and (CAP / 'attention.npz').exists()
    if have_c2:
        sub_c = slot_c.subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55); slot_c1, slot_c2 = sub_c[0, 0], sub_c[1, 0]
    else:
        slot_c1 = slot_c
    sub1 = slot_c1.subgridspec(1, 2, width_ratios=[0.68, 0.32], wspace=0.0)            # right 32%: paired differences vs FP (text)
    ax3 = fig.add_subplot(sub1[0, 0]); fig.add_subplot(sub1[0, 1]).set_axis_off()
    y = np.arange(3)[::-1]; rec = dict(layer=LAYER, step=9, observations=N, set='held-out (datasets/libero-heldout, 256 frames, one non-calibration episode per task)',
                                       latent_rule=capc['latent_rule'], segments=SEGS, segment_colors=GROUP_COL, per_model={}, delta_vs_fp_language_share={}, bootstrap=dict(draws=2000, seed=20260930, kind='paired over observations, percentile 95%'))
    for yi, (tag, lbl) in enumerate(LABELS):
        m = H[tag][:, LAYER]; left = 0.0; shares = {s: float(m[:, GI[s]].mean()) for s in SEGS}
        for s in SEGS:
            col, al = GROUP_COL[s]; w = shares[s]
            ax3.barh(y[yi], w, left=left, height=0.62, color=col, alpha=al, edgecolor=MGREY if s != 'language' else 'none', linewidth=0.4, zorder=3, label=GROUP_LBL[s] if yi == 0 else None)
            if s == 'language':
                ax3.text(left + w / 2, y[yi], f'{100 * w:.0f}%', ha='center', va='center', fontsize=7, color='white', zorder=4)
            left += w
        rec['per_model'][tag] = dict(model=capc['models'][tag], mean_share=shares, row_sum_mean=float(m[:, [GI[s] for s in SEGS]].sum(1).mean()))
        if tag != 'fp':
            d = m[:, GI['language']] - H['fp'][:, LAYER, GI['language']]; mean, lo, hi = boot_ci(d)
            ax3.text(1.03, y[yi], f'Δ {mean:+.2f}\n[{lo:+.2f}, {hi:+.2f}]', transform=ax3.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=VALGREY, linespacing=0.95, clip_on=False)
            rec['delta_vs_fp_language_share'][tag] = dict(mean=mean, ci95=[lo, hi], printed=fmt_delta(mean, lo, hi), fraction_of_observations_positive=float((d > 0).mean()))
    ax3.text(1.03, 1.0, 'Δ language\nvs FP, 95% CI', transform=ax3.transAxes, ha='left', va='bottom', fontsize=6.5, color=VALGREY, linespacing=0.95)
    ax3.set_yticks(y); ax3.set_yticklabels([lbl for _, lbl in LABELS])
    ax3.set_xlim(0, 1); ax3.set_xticks([0, 0.5, 1.0]); ax3.set_xticklabels(['0', '50', '100%'])
    ax3.set_xlabel('share of expert attention\n(reader block 0, last flow step)', labelpad=2, linespacing=0.95)
    ax3.spines['left'].set_visible(False); ax3.tick_params(axis='y', length=0); ax3.grid(axis='x', color=LGREY, lw=0.5, zorder=0)
    ax3.set_ylim(-1.6, 2.55)
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=2, handlelength=0.9, handleheight=0.7, columnspacing=0.6, handletextpad=0.35, borderaxespad=0, fontsize=6.5)
    letter(ax3, 'c')
    # per-layer table (all 18 expert layers, last step) for the JSON and the supplementary line plot
    per_layer = {tag: {s: H[tag][:, :, GI[s]].mean(0).tolist() for s in SEGS} for tag, _ in LABELS}
    per_layer_delta = {}
    for tag in ('ptq2bit', 'ours'):
        per_layer_delta[tag] = [dict(zip(('mean', 'lo', 'hi'), boot_ci(H[tag][:, l, GI['language']] - H['fp'][:, l, GI['language']]))) for l in range(18)]
    rec['per_layer_mean_share_last_step'] = per_layer; rec['per_layer_delta_vs_fp_language_share'] = per_layer_delta
    rec['c2_selection'] = dict(rule='FP only, 256 calibration observations, last step: expert layer with the largest mean third-person-image attention mass; (c2) only if >= 0.20',
                               fp_calibration_image_mass_by_layer=img_by_layer.tolist(), selected_layer=sel_layer, selected_layer_image_mass=sel_mass, c2_rendered=bool(have_c2))
    record['c1'] = rec
    if have_c2:
        att = np.load(CAP / 'attention.npz'); capf = json.loads((CAP / 'capture.json').read_text()); b0, b1 = capf['token_layout']['base_image']
        from PIL import Image
        frame = np.asarray(Image.open(CAP / 'frame.png'))
        maps = {tag: att[f'{tag}_step9'][sel_layer][b0:b1].reshape(16, 16) for tag, _ in LABELS}
        shown = [('fp', maps['fp'], 'FP'), ('ptq2bit', np.abs(maps['ptq2bit'] - maps['fp']), '|2-bit PTQ − FP|'), ('ours', np.abs(maps['ours'] - maps['fp']), '|ours − FP|')]
        vmax = max(float(m.max()) for _, m, _ in shown)
        CMAP = LinearSegmentedColormap.from_list('attn', [(0.0, to_rgba(YELLOW, 0.0)), (0.35, to_rgba(YELLOW, 0.55)), (0.7, to_rgba(PINK, 0.55)), (1.0, to_rgba(DARK, 0.55))])
        sub2 = slot_c2.subgridspec(1, 3, wspace=0.045); rec2 = dict(layer=sel_layer, step=9, vmin=0.0, vmax=vmax, unit=capf['unit'], policy_call=capf['k'], per_model={})
        for i, (tag, m, lbl) in enumerate(shown):
            a = fig.add_subplot(sub2[0, i]); a.imshow(frame, interpolation='bilinear'); a.imshow(m, cmap=CMAP, vmin=0, vmax=vmax, interpolation='bilinear', extent=(-0.5, 223.5, 223.5, -0.5))
            a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, 223.5); a.set_ylim(223.5, -0.5)
            for s in a.spines.values(): s.set_visible(True); s.set_edgecolor(DARK); s.set_linewidth(0.5)
            a.text(0.5, 1.03, lbl, transform=a.transAxes, ha='center', va='bottom', fontsize=6.5, color=DARK)
            l1 = float(np.abs(maps[tag] - maps['fp']).sum()); txt = f'image mass {float(maps[tag].sum()):.2f}' if tag == 'fp' else f'L1 to FP {l1:.3f}'
            a.text(0.5, -0.05, txt, transform=a.transAxes, ha='center', va='top', fontsize=6.5, color=DARK)
            rec2['per_model'][tag] = dict(model=capf['models'][tag], image_segment_mass=float(maps[tag].sum()), l1_to_fp=l1, printed=txt)
        record['c2'] = rec2
else:
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_axis_off(); ax3.text(0.5, 0.5, 'panel (c1): capture pending', ha='center', va='center', fontsize=7, color=VALGREY)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / f'{NAME}.pdf'); fig.savefig(OUT / f'{NAME}.png', dpi=300)
plt.close(fig)

# ---- supplementary: language share per expert layer (last step), three models, mean over the 256 held-out observations with 95% bootstrap band of the paired difference
if have_c1:
    f2, a2 = plt.subplots(figsize=(3.4, 2.0)); f2.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.2)
    xs = np.arange(18)
    for tag, lbl in LABELS:
        a2.plot(xs, per_layer[tag]['language'], color=MODEL_COL[tag], lw=1.2, marker='o', ms=2.6, mec=DARK, mew=0.3, label=lbl, zorder=3)
    a2.set_xticks(xs[::2]); a2.set_xlabel('expert layer'); a2.set_ylabel('language share of attention'); a2.grid(axis='y', color=LGREY, lw=0.5, zorder=0)
    a2.set_ylim(0, 1); a2.legend(frameon=False, loc='best', fontsize=6.5)
    a2.set_title(f'last flow step, mean over {N} held-out observations', fontsize=7, color=VALGREY, loc='left')
    f2.savefig(OUT / f'{NAME}_supp_language_share_per_layer.png', dpi=300); plt.close(f2)
(OUT / f'{NAME}.json').write_text(json.dumps(record, indent=1))
print(json.dumps(dict(font=FONT, growth=growth, have_c1=have_c1, c2=record.get('c1', {}).get('c2_selection'))))
