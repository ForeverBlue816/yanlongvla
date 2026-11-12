"""Shared figure style (LoopQ paper). NOTE: no figures/loopq_style.py existed on the cluster when the readout visualisation was made
(2026-11-12), so this file defines the names the request used; replace the values with the paper's own palette if they differ."""
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

INK = '#1F2430'           # text
NAVY = '#1F3A6E'          # LoopQ
CORAL_DARK = '#D9573F'    # 2-bit base
CORAL_LIGHT = '#F4B8A8'   # error sticks, glow
FP_FILL = '#FFFFFF'       # FP ribbon
FP_EDGE = '#59626E'       # FP outline
GRID = '#DADDE3'
CMAP_DIV = LinearSegmentedColormap.from_list('loopq_div', [NAVY, '#8FA3C8', '#F7F7F7', '#F0A593', CORAL_DARK])
CMAP_SEQ = LinearSegmentedColormap.from_list('loopq_seq', ['#10203F', NAVY, '#5C7FB8', '#F2D7A0', '#F0A070', CORAL_DARK])      # low -> high (placeholder; added for the attention overlays)


def apply(size=7.0):
    for f in sorted(Path.home().glob('.fonts/termes/*.otf')):
        font_manager.fontManager.addfont(str(f))
    families = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral') if f in families), 'serif')
    matplotlib.rcParams.update({'font.family': font, 'font.size': size, 'axes.labelsize': size, 'xtick.labelsize': size - 0.5, 'ytick.labelsize': size - 0.5, 'legend.fontsize': size - 0.5,
                                'mathtext.fontset': 'stix', 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK,
                                'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.transparent': True, 'figure.dpi': 150})
    return font


def save(fig, stem, dpi=600):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.01)
