"""Regenerate Figures 1-3 from the archived per-seed results."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'data' / 'results.csv'
OUT = ROOT / 'output' / 'figures'

MODELS = ['B', 'STAL', 'STAL_SD', 'P2', 'TWIN']          # fixed order in every figure
LABEL = {'B': 'B', 'STAL': 'STAL', 'STAL_SD': 'STAL-SD', 'P2': 'P2', 'TWIN': 'TWIN'}
SEEDS = {'B': ['0', '1', '2'], 'STAL': ['0', '1', '2'], 'STAL_SD': ['0', '1', '2'], 'P2': ['0', '1', '2'],
         'TWIN': ['0', '1']}

INK, INK2, MUTED, GRID = '#1F2328', '#57606A', '#8C959F', '#E5E7EB'
BAR = '#4C78A8'                                           # one series -> one hue, no legend
ACCENT, WINDOW = '#D55E00', '#0072B2'                     # schematic only (Okabe-Ito)

plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'savefig.dpi': 300,
})
CM = 1 / 2.54


def load():
    rows = defaultdict(dict)
    with TABLE.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['study'] == 'baseline50':
                rows[(r['arm'], r['setting'], r['metric'])][r['seed']] = float(r['value'])
    return rows


def per_seed(rows, arm, setting, metric):
    d = rows[(arm, setting, metric)]
    return [d[s] for s in SEEDS[arm]]


def style_axis(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def bars_with_seeds(ax, values, fmt):
    x = range(len(MODELS))
    means = [sum(v) / len(v) for v in values]
    ax.bar(x, means, width=0.62, color=BAR, edgecolor='white', linewidth=1.0, zorder=2)
    for i, v in enumerate(values):
        offs = [(-0.12 + 0.24 * k / max(len(v) - 1, 1)) for k in range(len(v))]
        ax.scatter([i + o for o in offs], v, s=10, color=INK, zorder=3, linewidths=0)
        ax.text(i, max(v) + 0.025 * ax.get_ylim()[1], fmt.format(means[i]), ha='center', va='bottom',
                color=INK, fontsize=7.5)
    ax.set_xticks(list(x), [LABEL[m] for m in MODELS])
    style_axis(ax)


def fig2(rows):
    setting = 'matched FP against B in same seed'
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.2 * CM))
    for ax, metric, title, top in ((axes[0], 'E1vt_R50', 'E1vt recall at IoU 0.5', 38),
                                   (axes[1], 'E1vt_R75', 'E1vt recall at IoU 0.75', 6.8)):
        ax.set_ylim(0, top)
        bars_with_seeds(ax, [per_seed(rows, m, setting, metric) for m in MODELS], '{:.1f}')
        ax.set_title(title, loc='left')
        ax.set_ylabel('Recall (%)')
    save(fig, 'fig2_matched_recall')


def fig3(rows):
    panels = [
        ('500 val images', 'GMACs_dense', 'Static computation\n(GMAC/image)', 250),
        ('500 val images', 'energy_mJ', 'Estimated energy\n(mJ/image)', 860),
        ('500 val images batch=4', 'pipeline_ms_per_image', 'Latency, batch 4\n(ms/image)', 46),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 5.4 * CM), sharey=True)
    y = list(range(len(MODELS)))[::-1]                     # B on top
    for ax, (setting, metric, title, right) in zip(axes, panels):
        values = [per_seed(rows, m, setting, metric) for m in MODELS]
        means = [sum(v) / len(v) for v in values]
        ax.barh(y, means, height=0.62, color=BAR, edgecolor='white', linewidth=1.0, zorder=2)
        for yi, v, mu in zip(y, values, means):
            offs = [(-0.12 + 0.24 * k / max(len(v) - 1, 1)) for k in range(len(v))]
            ax.scatter(v, [yi + o for o in offs], s=8, color=INK, zorder=3, linewidths=0)
            ax.text(max(v) + 0.02 * right, yi, f'{mu:.1f}', va='center', ha='left', color=INK, fontsize=7.5)
        ax.set_xlim(0, right)
        ax.set_title(title, loc='left')
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, [LABEL[m] for m in MODELS])
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig3_cost')


def fig1():
    """Stride-8 reference points around a 6 x 5 px object with no strictly interior point."""
    fig, ax = plt.subplots(figsize=(8.6 * CM, 8.2 * CM))
    for g in (0, 8, 16, 24, 32):
        ax.axvline(g, color=GRID, lw=0.6, zorder=0)
        ax.axhline(g, color=GRID, lw=0.6, zorder=0)
    pts = [4, 12, 20, 28]
    ax.scatter([x for x in pts for _ in pts], [y for _ in pts for y in pts], s=14, color=MUTED, zorder=2,
               label='Reference point (stride 8)')
    x1, y1, x2, y2 = 13, 14, 19, 19                          # object box, image coordinates
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax.add_patch(Rectangle((cx - 8, cy - 8), 16, 16, fill=False, ls=(0, (4, 2)), lw=1.2, ec=WINDOW, zorder=3,
                           label='Admission window (16 × 16 px)'))
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.6, ec=INK, zorder=4,
                           label='Object box (6 × 5 px)'))
    ax.scatter([12], [12], s=34, color=ACCENT, zorder=5, label='Admitted outside point')
    # extend the near edges to the point so the two negative distances are visible
    ax.plot([x1, x1], [11.0, y1], color=INK, lw=0.7, ls=':', zorder=3)
    ax.plot([11.0, x1], [y1, y1], color=INK, lw=0.7, ls=':', zorder=3)
    arrow = dict(arrowstyle='<->', lw=0.9, shrinkA=0, shrinkB=0, color=ACCENT)
    ax.annotate('', xy=(12, 11.3), xytext=(x1, 11.3), arrowprops=arrow)
    ax.text(12.5, 10.9, 'left: −1 px', color=ACCENT, ha='center', va='bottom', fontsize=7.5)
    ax.annotate('', xy=(11.3, 12), xytext=(11.3, y1), arrowprops=arrow)
    ax.text(10.9, 13.0, 'top: −2 px', color=ACCENT, ha='right', va='center', fontsize=7.5)
    ax.set_xlim(6, 26)
    ax.set_ylim(26, 6)
    ax.set_xticks([8, 12, 16, 20, 24])
    ax.set_yticks([8, 12, 16, 20, 24])
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_aspect('equal')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7, handlelength=1.6,
              columnspacing=1.2)
    save(fig, 'fig1_geometry')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    fig1()
    fig2(data)
    fig3(data)
    print('written to', OUT)
