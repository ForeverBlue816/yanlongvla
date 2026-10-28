"""Figure v2 (user 2026-10-27, "Nature level"): fig_readout_sensitivity_v2.{pdf,png} + three supplementary PNGs + JSON of every printed
number. (a) GR00T N1.7 readout sensitivity per reader block x flow step (FP head, B-AM Ω); (b) G_out DoF-energy shares per module with
the top-10%-row concentration; (c) pi0.5-LIBERO expert cross-attention of reader block 0 at the last flow step over the agent-view
image tokens for FP / 2-bit PTQ / ours, on the policy call immediately before the first gripper-close event of the FP rollout
(runs/plustrain_20261008/diagnostics/fig_c, produced by paper_fig_c_capture.py). Font: TeX Gyre Termes (Times-compatible); mathtext stix."""
import json
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
NAME = 'fig_readout_sensitivity_v2'

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
record = dict(font=FONT, mathtext='stix', figure_inches=[7.2, 2.5], width_ratios=[1.35, 0.85, 1.1], wspace=0.3)


def letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom', color=DARK)


fig = plt.figure(figsize=(7.2, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.1], wspace=0.3, left=0.075, right=0.985, top=0.9, bottom=0.2)

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
    ax.annotate(f'step {t}', xy=(x[-1], vals[t, -1]), xytext=(x[-1] + 0.45, 10 ** pos[t]), textcoords='data', va='center', ha='left', fontsize=7, color=STEP[t] if t else '#7FA6C9',
                annotation_clip=False, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks])
ax.set_xlabel('reader block')
ax.set_ylabel(r'action sensitivity of readout, $\|\Omega^{1/2} J_{v \leftarrow o}\|_F^2$')
ax.set_xlim(-0.6, len(blocks) + 1.7)   # room for the right-hand end labels inside the axes
ax.set_ylim(vals.min() / 3.2, vals.max() * 5.5)
ax.spines['bottom'].set_bounds(-0.6, len(blocks) - 0.4)
i0 = blocks.index(0)
ax.annotate('first text block:\nmost sensitive at every step', xy=(x[i0] + 0.2, vals[3, i0] * 1.12), xytext=(x[i0] + 1.9, vals[3, i0] * 2.3), fontsize=7, color=DARK, ha='left', va='center', linespacing=1.0,
            bbox=BOX, arrowprops=dict(arrowstyle='-', color=MGREY, lw=0.5, shrinkA=0, shrinkB=1))
growth = float(vals[3].mean() / vals[0].mean())
ax.text(0.98, 0.05, f'step 3 / step 0 = {growth:.1f}×', transform=ax.transAxes, fontsize=7, color=DARK, ha='right', va='bottom', bbox=BOX)
letter(ax, 'a')
record['a'] = dict(blocks=blocks, block_kind={str(b): kind[b] for b in blocks}, values={f'block{b}_step{t}': float(vals[t, i]) for i, b in enumerate(blocks) for t in range(4)},
                   step3_over_step0_mean_over_blocks=growth, top_block_step3=int(blocks[int(np.argmax(vals[3]))]), image_band_alpha=0.6,
                   image_band_meaning='grey band = reader block reads image tokens; white = text tokens', step_colors=STEP)

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

# ================= (c) pi0.5 expert cross-attention overlays
# fixed boxes in the 224x224 agent-view frame (x0, y0, x1, y1), set by hand on frame.png and recorded in the JSON
# Fixed 2026-10-28 06:10 UTC from fig_c/frame.png (policy call 7, the call whose chunk holds the first gripper-close command), BEFORE any attention was
# rendered; not moved afterwards (user 2026-10-27 §3). 224x224 model-input frame, (x0, y0, x1, y1) in pixels; a 14-px patch counts when its centre is inside.
#   target  = the black bowl under the gripper (patch columns 3-4, rows 9-10)
#   gripper = the two finger pads and the gripper body above the bowl (patch columns 1-6, rows 7-8)
BOXES = dict(target=(40, 126, 72, 150), gripper=(17, 98, 94, 126))
CMAP = LinearSegmentedColormap.from_list('attn', [(0.0, to_rgba(YELLOW, 0.0)), (0.35, to_rgba(YELLOW, 0.55)), (0.7, to_rgba(PINK, 0.55)), (1.0, to_rgba(DARK, 0.55))])
LABELS = [('fp', 'FP'), ('ptq2bit', '2-bit PTQ'), ('ours', 'ours')]


def box_mass(m16, box):
    """attention mass of the 16x16 patch map inside a pixel box: patches whose 14-px cell centre lies in the box (share of the whole attention row)."""
    cx = np.arange(16) * 14 + 7; inside = np.zeros((16, 16), bool)
    for i in range(16):
        for j in range(16):
            inside[i, j] = box[0] <= cx[j] < box[2] and box[1] <= cx[i] < box[3]
    return float(m16[inside].sum()), inside


def panel_c(fig, slot, layer, step, title, record_key):
    sub = slot.subgridspec(1, 3, wspace=0.045)
    cap = json.loads((CAP / 'capture.json').read_text()); att = np.load(CAP / 'attention.npz')
    from PIL import Image
    frame = np.asarray(Image.open(CAP / 'frame.png'))
    lay = cap['token_layout']; b0, b1 = lay['base_image']
    maps = {tag: att[f'{tag}_step{step}'][layer][b0:b1].reshape(16, 16) for tag, _ in LABELS}
    vmax = max(float(m.max()) for m in maps.values()); rec = dict(layer=layer, step=step, vmin=0.0, vmax=vmax, alpha=0.55, boxes=BOXES, per_model={})
    axes = []
    for i, (tag, lbl) in enumerate(LABELS):
        a = fig.add_subplot(sub[0, i]); axes.append(a)
        a.imshow(frame, interpolation='bilinear'); a.imshow(maps[tag], cmap=CMAP, vmin=0, vmax=vmax, interpolation='bilinear', extent=(-0.5, 223.5, 223.5, -0.5))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(True); s.set_edgecolor(DARK); s.set_linewidth(0.5)
        a.set_xlim(-0.5, 223.5); a.set_ylim(223.5, -0.5)
        tm, tin = box_mass(maps[tag], BOXES['target']); gm, gin = box_mass(maps[tag], BOXES['gripper'])
        seg = float(maps[tag].sum())                     # attention mass of the whole agent-view image segment (256 tokens)
        tshare, gshare = tm / seg, gm / seg               # printed: share of the agent-view image attention inside each fixed box
        a.text(0.5, 1.03, lbl, transform=a.transAxes, ha='center', va='bottom', fontsize=7.5, color=DARK)
        a.text(0.5, -0.05, f'target {tshare:.2f}\ngripper {gshare:.2f}', transform=a.transAxes, ha='center', va='top', fontsize=7, color=DARK, linespacing=1.15)
        dfp = float(np.abs(maps[tag] - maps['fp']).sum())
        rec['per_model'][tag] = dict(model=cap['models'][tag], target_share_of_image_attention=tshare, gripper_share_of_image_attention=gshare,
                                     target_mass=tm, gripper_mass=gm, target_cells=int(tin.sum()), gripper_cells=int(gin.sum()),
                                     target_uniform_share=float(tin.sum() / 256), gripper_uniform_share=float(gin.sum() / 256),
                                     l1_to_fp_over_image_segment=dfp, argmax_cell_row_col=[int(v) for v in np.unravel_index(int(maps[tag].argmax()), (16, 16))],
                                     image_segment_mass=seg, map_max=float(maps[tag].max()),
                                     language_mass=float(att[f'{tag}_step{step}'][layer][lay['language'][0]:lay['language'][1]].sum()),
                                     wrist_mass=float(att[f'{tag}_step{step}'][layer][lay['wrist_image'][0]:lay['wrist_image'][1]].sum()),
                                     action_token_mass=float(att[f'{tag}_step{step}'][layer][lay['action_tokens'][0]:lay['action_tokens'][1]].sum()))
    axes[0].text(0.0, 1.30, title, transform=axes[0].transAxes, ha='left', va='bottom', fontsize=7, color=VALGREY) if title else None
    record[record_key] = rec
    return axes


BOXES_SET = all(b[2] > b[0] and b[3] > b[1] for b in BOXES.values())
have_c = (CAP / 'capture.json').exists() and (CAP / 'attention.npz').exists() and BOXES_SET   # boxes are fixed from the raw frame BEFORE any overlay is rendered
if (CAP / 'capture.json').exists() and not BOXES_SET:
    print('capture present but target/gripper boxes not yet set from frame.png — panel (c) left as a placeholder', file=sys.stderr)
if have_c:
    axes_c = panel_c(fig, gs[0, 2], layer=0, step=9, title='', record_key='c')
    letter(axes_c[0], 'c')
    cap = json.loads((CAP / 'capture.json').read_text())
    record['c'].update(unit=cap['unit'], policy_call=cap['k'], rollout=cap['rollout'], gpu=cap['gpu'], token_layout=cap['token_layout'],
                       per_model_checks={t: {k: v for k, v in m.items() if k != 'audit'} for t, m in cap['per_model'].items()},
                       overlay='expert attention of reader block 0, last flow step (step 9 of 10), mean over 8 heads and 10 action tokens, agent-view image-token segment (16x16), bilinear upsampling; shared colour range across the three panels')
else:
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_axis_off(); ax3.text(0.5, 0.5, 'panel (c): capture pending' if not (CAP / 'capture.json').exists() else 'panel (c): boxes pending', ha='center', va='center', fontsize=7, color=VALGREY); letter(ax3, 'c')

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / f'{NAME}.pdf'); fig.savefig(OUT / f'{NAME}.png', dpi=300)
plt.close(fig)

# ---- supplementary panels (c only): block 8 last step, block 0 first step, block 8 first step
if have_c:
    for layer, step, suffix in ((8, 9, 'block8_laststep'), (0, 0, 'block0_firststep'), (8, 0, 'block8_firststep')):
        f2 = plt.figure(figsize=(2.6, 1.3)); g2 = f2.add_gridspec(1, 1, left=0.02, right=0.98, top=0.74, bottom=0.20)
        panel_c(f2, g2[0, 0], layer=layer, step=step, title=f'reader block {layer}, flow step {step} of 10', record_key=f'supp_{suffix}')
        f2.savefig(OUT / f'{NAME}_supp_{suffix}.png', dpi=300); plt.close(f2)
(OUT / f'{NAME}.json').write_text(json.dumps(record, indent=1))
print(json.dumps(dict(font=FONT, growth=growth, top10=top10, counts=counts, have_c=have_c)))
