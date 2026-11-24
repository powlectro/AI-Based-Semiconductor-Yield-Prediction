"""Generate the design-space exploration dashboard and report figures.

Reads `reports/step7_sequential_mac.csv` (parallel + sequential sweep) and
writes:
  - `dashboard/design_space.html` (standalone HTML with interactive SVG hover tips)
  - `reports/figures/design_space_{recall,cost,throughput}_vs_area.png`

All plotted accuracy/cost points are VALIDATION metrics (the numbers allowed to
choose feature count and bit width). If `reports/step6_test_results.json`
exists, the held-out test score is displayed in a separate summary panel beside
the frozen 16×12 design and is never mixed into the selectable design-space plots.
"""
import csv
import json
from pathlib import Path

import matplotlib

# Non-interactive backend so the script runs cleanly in headless terminals/CI.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
HTML_PATH = ROOT / 'dashboard' / 'design_space.html'
PARALLEL = '#1f4e79'
SEQUENTIAL = '#d35400'


def load_rows():
    """Load the combined parallel and sequential hardware/validation sweep table."""
    path = REPORTS / 'step7_sequential_mac.csv'
    if not path.is_file():
        raise SystemExit('reports/step7_sequential_mac.csv is missing. Run scripts/07_sequential_mac.py first.')
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = {
            'architecture': row['architecture'],
            'features': int(row['features']),
            'bits': int(row['bits']),
            'recall': float(row['recall']),
            'false_positive_rate': float(row['false_positive_rate']),
            'screening_cost': int(float(row['screening_cost'])),
            'agreement_with_float': float(row['agreement_with_float']),
            'latency_cycles': int(row['latency_cycles']),
            'interval_cycles': int(row['interval_cycles']),
            'throughput_msps': float(row['throughput_msps']),
            'area_um2': float(row['area_um2']),
            'critical_path_ns': float(row['critical_path_ns']),
            'fmax_mhz': float(row['fmax_mhz']),
            'cell_count': int(row['cell_count']),
        }
        # Flag the frozen 16-feature, 12-bit operating point for visual highlighting.
        item['frozen'] = item['features'] == 16 and item['bits'] == 12
        item['label'] = f"{item['features']}×{item['bits']}"
        parsed.append(item)
    return parsed


def load_test():
    """Load the one-shot held-out test summary if Step 6 has been run."""
    path = REPORTS / 'step6_test_results.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def draw(rows, xkey, ykey, xlabel, ylabel, title, path):
    """Save a publication-ready Matplotlib scatter plot comparing both architectures."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for architecture, color, marker in (
        ('parallel', PARALLEL, 'o'),
        ('sequential', SEQUENTIAL, 's'),
    ):
        subset = [row for row in rows if row['architecture'] == architecture]
        ax.scatter(
            [row[xkey] for row in subset],
            [row[ykey] for row in subset],
            s=70,
            c=color,
            marker=marker,
            label=architecture,
            zorder=3,
        )
        for row in subset:
            if row['frozen']:
                # Draw a gold halo around the frozen 16x12 point and stagger the
                # two architecture labels so they do not collide at equal y values.
                ax.scatter(
                    [row[xkey]], [row[ykey]],
                    s=220, facecolors='none', edgecolors='#8a6d1d', linewidths=2.0, zorder=4,
                )
                if architecture == 'parallel':
                    offset, align = (12, 12), 'left'
                else:
                    offset, align = (-12, -16), 'right'
                ax.annotate(
                    f"{architecture} 16×12",
                    (row[xkey], row[ykey]),
                    textcoords='offset points',
                    xytext=offset,
                    ha=align,
                    fontsize=8,
                    color=color,
                )
    # Add extra headroom/footroom so staggered annotations stay inside the frame.
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin or 1
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.16 * span)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def svg_scatter(rows, xkey, ykey, xlabel, ylabel, title):
    """Render an inline SVG scatter plot with native hover tooltips for design_space.html."""
    width, height = 760, 420
    left, right, top, bottom = 70, 20, 46, 54
    xs = [row[xkey] for row in rows]
    ys = [row[ykey] for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin) * 0.08 or 1
    ypad = (ymax - ymin) * 0.16 or 0.05

    # Anchor area and non-negative metrics at 0 so axis ticks never show negative values.
    xmin = 0 if xkey == 'area_um2' else xmin - xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad
    if ymin < 0:
        ymin = 0
    yspan = ymax - ymin

    def tick_text(value):
        if yspan >= 20:
            return f'{value:.0f}'
        if yspan >= 1:
            return f'{value:.1f}'
        return f'{value:.2f}'

    # Linear data-to-pixel transforms (SVG y grows downward).
    def px(value):
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def py(value):
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<text x="{left}" y="28" font-size="16" font-family="Georgia, serif">{title}</text>',
    ]
    for tick in range(6):
        value = xmin + (xmax - xmin) * tick / 5
        x = px(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" font-size="11" fill="#444">{value:.0f}</text>'
        )
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = py(value)
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#444">{tick_text(value)}</text>'
        )
    parts.append(
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="12">{xlabel}</text>'
    )
    parts.append(
        f'<text x="18" y="{(top + height - bottom) / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})">{ylabel}</text>'
    )
    for row in rows:
        color = PARALLEL if row['architecture'] == 'parallel' else SEQUENTIAL
        radius = 7 if row['frozen'] else 5
        tip = (
            f"{row['architecture']} {row['label']}: recall {row['recall']:.3f}, "
            f"cost {row['screening_cost']}, area {row['area_um2']:.0f} µm², "
            f"{row['throughput_msps']:.1f} Msamples/s, latency {row['latency_cycles']} cycles"
        )
        parts.append(
            f'<circle cx="{px(row[xkey]):.1f}" cy="{py(row[ykey]):.1f}" r="{radius}" '
            f'fill="{color}"><title>{tip}</title></circle>'
        )
        if row['frozen']:
            parts.append(
                f'<circle cx="{px(row[xkey]):.1f}" cy="{py(row[ykey]):.1f}" r="12" '
                f'fill="none" stroke="#8a6d1d" stroke-width="2"><title>{tip}</title></circle>'
            )
            if row['architecture'] == 'parallel':
                text_x, anchor, text_y = px(row[xkey]) + 16, 'start', py(row[ykey]) - 8
            else:
                text_x, anchor, text_y = px(row[xkey]) - 16, 'end', py(row[ykey]) + 16
            parts.append(
                f'<text x="{text_x:.1f}" y="{text_y:.1f}" text-anchor="{anchor}" font-size="11" '
                f'fill="{color}">{row["architecture"]} 16×12</text>'
            )
    parts.append('</svg>')
    return '\n'.join(parts)


def table_rows(rows):
    """Build the HTML table rows sorted by feature count, precision, and architecture."""
    ordered = sorted(rows, key=lambda row: (row['features'], row['bits'], row['architecture']))
    body = []
    for row in ordered:
        mark = ' frozen' if row['frozen'] else ''
        body.append(
            '<tr class="{arch}{mark}">'
            '<td>{architecture}</td><td>{features}</td><td>{bits}</td>'
            '<td>{recall:.3f}</td><td>{false_positive_rate:.3f}</td><td>{screening_cost}</td>'
            '<td>{agreement_with_float:.3f}</td><td>{latency_cycles}</td><td>{interval_cycles}</td>'
            '<td>{throughput_msps:.1f}</td><td>{area_um2:.0f}</td><td>{critical_path_ns:.2f}</td>'
            '<td>{fmax_mhz:.0f}</td></tr>'.format(arch=row['architecture'], mark=mark, **row)
        )
    return '\n'.join(body)


def test_panel(test):
    """Render the sealed held-out test summary block separately from the validation sweep."""
    if test is None:
        return (
            '<p class="note">The held-out test report is not in this folder yet. '
            'The plot remains the validation design space.</p>'
        )
    floated = test['float']
    fixed = test['fixed_point']
    limits = ''.join(f'<li>{item}</li>' for item in test['limitations'])
    return f'''
    <section class="panel">
      <h2>Held-out score of the frozen design</h2>
      <p>16 features, 12 bits, threshold {test['threshold']:.2f}.
      This score was not used to place or choose the points on the plots.</p>
      <dl>
        <div><dt>Float recall</dt><dd>{floated['failure_recall']:.3f}</dd></div>
        <div><dt>Float false-positive rate</dt><dd>{floated['false_positive_rate']:.3f}</dd></div>
        <div><dt>Float screening cost</dt><dd>{floated['screening_cost']}</dd></div>
        <div><dt>12-bit agreement</dt><dd>{test['agreement_with_float']:.3f}</dd></div>
        <div><dt>12-bit recall</dt><dd>{fixed['failure_recall']:.3f}</dd></div>
        <div><dt>12-bit screening cost</dt><dd>{fixed['screening_cost']}</dd></div>
        <div><dt>Parallel RTL</dt><dd>{test['parallel_rtl']}</dd></div>
        <div><dt>Sequential RTL</dt><dd>{test['sequential_rtl']}</dd></div>
      </dl>
      <ul>{limits}</ul>
    </section>
    '''


def write_html(rows, test):
    """Assemble the self-contained HTML dashboard (no external JS/CSS dependencies)."""
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SECOM classifier design space</title>
  <style>
    body {{ margin: 0; background: #f7f4ee; color: #1c1c1c; font: 16px/1.45 Georgia, serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
    h1 {{ font-weight: 600; font-size: 32px; margin-bottom: 8px; }}
    h2 {{ font-size: 22px; margin: 28px 0 8px; }}
    .legend {{ display: flex; gap: 18px; align-items: center; margin: 8px 0 18px; }}
    .swatch {{ width: 12px; height: 12px; display: inline-block; margin-right: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .card {{ background: white; border: 1px solid #e4dccb; padding: 12px 12px 4px; }}
    table {{ width: 100%; border-collapse: collapse; font: 13px/1.35 ui-sans-serif, sans-serif; background: white; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr.frozen {{ background: #fff6dc; }}
    .panel {{ background: white; border: 1px solid #e4dccb; padding: 8px 18px 18px; }}
    dl {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px 16px; margin: 12px 0; }}
    dt {{ font: 12px/1.3 ui-sans-serif, sans-serif; color: #666; }}
    dd {{ margin: 0; font-size: 22px; }}
    .note {{ color: #444; }}
    @media (max-width: 800px) {{ dl {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Prediction accuracy versus hardware cost</h1>
  <p>Each point is one logistic-regression export. Recall, false-positive rate, and
  screening cost are <strong>validation</strong> results. Area and timing are Nangate45
  typical estimates with no wire-load model and no clock tree. Gold rings mark the
  frozen 16-feature, 12-bit design. Hover a point for its counts.</p>
  <div class="legend">
    <span><i class="swatch" style="background:{PARALLEL}"></i>Parallel, 1-cycle datapath</span>
    <span><i class="swatch" style="background:{SEQUENTIAL}"></i>Sequential MAC, N-cycle datapath</span>
  </div>
  <div class="grid">
    <div class="card">{svg_scatter(rows, 'area_um2', 'recall', 'Area (µm²)', 'Validation failure recall', 'Failure recall versus area')}</div>
    <div class="card">{svg_scatter(rows, 'area_um2', 'screening_cost', 'Area (µm²)', 'Validation screening cost', 'Screening cost versus area')}</div>
    <div class="card">{svg_scatter(rows, 'area_um2', 'throughput_msps', 'Area (µm²)', 'Throughput (million samples/s)', 'Throughput versus area')}</div>
  </div>
  {test_panel(test)}
  <h2>Design points</h2>
  <table>
    <thead>
      <tr>
        <th>Architecture</th><th>Features</th><th>Bits</th><th>Recall</th><th>FPR</th>
        <th>Cost</th><th>Float agreement</th><th>Latency</th><th>Interval</th>
        <th>Msamples/s</th><th>Area µm²</th><th>Path ns</th><th>fmax MHz</th>
      </tr>
    </thead>
    <tbody>
      {table_rows(rows)}
    </tbody>
  </table>
  <p class="note">Screening cost = 10 × missed failures + 1 × false alarms on the 314-row
  validation split. Throughput = fmax / initiation interval. The sequential interval
  equals the feature count. A selected sensor is not a physical cause, and these
  costs are not fab prices.</p>
</main>
</body>
</html>
'''
    HTML_PATH.write_text(html, encoding='utf-8')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    test = load_test()
    draw(
        rows, 'area_um2', 'recall',
        'Area (µm²)', 'Validation failure recall',
        'Failure recall versus Nangate45 area',
        FIGURES / 'design_space_recall_vs_area.png',
    )
    draw(
        rows, 'area_um2', 'screening_cost',
        'Area (µm²)', 'Validation screening cost',
        'Screening cost versus Nangate45 area',
        FIGURES / 'design_space_cost_vs_area.png',
    )
    draw(
        rows, 'area_um2', 'throughput_msps',
        'Area (µm²)', 'Throughput (million samples/s)',
        'Throughput versus Nangate45 area',
        FIGURES / 'design_space_throughput_vs_area.png',
    )
    write_html(rows, test)
    print(f'Saved: {HTML_PATH}')
    print(f'Saved: {FIGURES / "design_space_recall_vs_area.png"}')
    print(f'Saved: {FIGURES / "design_space_cost_vs_area.png"}')
    print(f'Saved: {FIGURES / "design_space_throughput_vs_area.png"}')


if __name__ == '__main__':
    main()
