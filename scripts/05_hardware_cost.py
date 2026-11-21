"""Stage 5: area, timing, and latency for the clocked fixed-point classifier

Train with scripts/03_train_compare.py first. This calls
scripts/04_export_fixed_point.py for each (features, bits) point, simulates
hardware/classifier_top_tb.sv against those validation vectors, and maps
hardware/linear_classifier_top.sv with Yosys and the Nangate45 typical library

The held-out TEST partition is not scored. reports/step4_quantization.csv is
restored afterward so this sweep does not replace the Step 4 table
"""
import csv
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
LIB_DIR = ROOT / 'hardware' / 'lib'
SYNTH_DIR = ROOT / 'hardware' / 'synth'
LIB_NAME = 'NangateOpenCellLibrary_typical.lib'
LIB_URL = (
    'https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/'
    'master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib'
)

# D-pin capacitance of Nangate45 DFF_X1, in femtofarads (lib capacitive_load_unit)
DFF_D_LOAD_FF = 1.14029
FEATURE_COUNTS = (8, 16, 32)
BIT_WIDTHS = (8, 12, 16)
STEP4_CSV = REPORTS / 'step4_quantization.csv'
OUT_CSV = REPORTS / 'step5_hardware_cost.csv'
OUT_MD = REPORTS / 'step5_hardware_cost.md'


def require_tool(name):
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f'{name} was not found on PATH. Install it in the course conda env '
            f'(conda install -n ece8803-ai4 -c conda-forge yosys) and activate that env. '
            f'Icarus Verilog (iverilog, vvp) is also required.'
        )

    return path


def ensure_liberty():
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    lib = LIB_DIR / LIB_NAME
    if lib.is_file() and lib.stat().st_size > 1_000_000:
        return lib

    print(f'Downloading Nangate45 typical liberty to {lib}')
    request = urllib.request.Request(LIB_URL, headers={'User-Agent': 'ece8803-hardware-cost'})
    with urllib.request.urlopen(request) as src, lib.open('wb') as dst:
        shutil.copyfileobj(src, dst)

    if lib.stat().st_size < 1_000_000:
        raise SystemExit(f'Liberty download looks too small: {lib}')

    return lib


def brace_block(text, start):
    i = text.find('{', start)
    if i < 0:
        raise ValueError('no brace block')

    depth = 0

    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]

    raise ValueError('unbalanced liberty braces')


def table_center(timing_body, table_name):
    marker = timing_body.find(table_name + '(')
    if marker < 0:
        raise ValueError(f'missing {table_name}')

    block = timing_body[marker:]
    values = block.split('values', 1)[1].split(';', 1)[0]
    rows = [[float(x) for x in row.split(',')] for row in re.findall(r'"([^"]+)"', values)]

    if not rows or not rows[0]:
        raise ValueError(f'empty {table_name}')

    return rows[len(rows) // 2][len(rows[0]) // 2]


def dff_timing(lib_text, cell_name='DFF_X1'):
    """Middle-of-table clock-to-Q and setup for one Nangate flip-flop, in ns."""
    cell = brace_block(lib_text, lib_text.find(f'cell ({cell_name})'))
    q_pin = brace_block(cell, cell.find('pin (Q)'))
    q_timing = q_pin.split('timing_type', 1)[1]

    if 'rising_edge' not in q_timing[:80]:
        raise ValueError('DFF Q pin has no rising_edge arc')

    clk_to_q = max(table_center(q_timing, 'cell_rise'), table_center(q_timing, 'cell_fall'))
    d_pin = brace_block(cell, cell.find('pin (D)'))
    setup = None

    for part in d_pin.split('timing ()')[1:]:
        if 'setup_rising' not in part[:400]:
            continue

        setup = max(table_center(part, 'rise_constraint'), table_center(part, 'fall_constraint'))
        break

    if setup is None:
        raise ValueError('DFF D pin has no setup_rising arc')

    return clk_to_q, setup


def read_step4_row():
    with STEP4_CSV.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))

    if len(rows) != 1:
        raise SystemExit(f'Expected one Step 4 row after a single-bit export, found {len(rows)}')

    return rows[0]


def simulate_wrapper(folder, features, bits, samples):
    sim = folder / 'classifier_top_sim'
    cmd = [
        'iverilog', '-g2012', '-s', 'classifier_top_tb',
        f'-DBITS={bits}', f'-DN={features}', f'-DSAMPLES={samples}',
        '-I', str(folder), '-o', str(sim),
        str(ROOT / 'hardware/linear_classifier.sv'),
        str(ROOT / 'hardware/linear_classifier_top.sv'),
        str(ROOT / 'hardware/classifier_top_tb.sv'),
    ]
    subprocess.run(cmd, cwd=folder, check=True)
    run = subprocess.run(['vvp', str(sim)], cwd=folder, check=True, capture_output=True, text=True)
    print(run.stdout.strip())
    match = re.search(r'PASS: (\d+) Python/RTL vector comparisons, latency_cycles=(\d+)', run.stdout)

    if not match or int(match.group(1)) != samples:
        raise SystemExit(f'Wrapper simulation did not pass for k={features} bits={bits}')

    return int(match.group(2))


def synthesize(folder, features, bits, lib, clk_to_q, setup):
    out = SYNTH_DIR / f'k{features:02d}_b{bits:02d}'
    out.mkdir(parents=True, exist_ok=True)
    constr = out / 'nangate.constr'
    constr.write_text(f'set_driving_cell BUF_X1\nset_load {DFF_D_LOAD_FF}\n', encoding='ascii')
    # A file, not an inline script: Yosys treats ';' as its own command separator
    # `map` is the direct liberty mapper. The default script's dch/upsize passes are
    # very slow on the 32-feature multiplier and are not required for this estimate
    abc_script = out / 'map.abc'
    abc_script.write_text('strash\nmap\ntopo\nstime -p\n', encoding='ascii')
    script = out / 'synth.ys'
    script.write_text(
        '\n'.join([
            f'read_liberty -lib {lib}',
            f'verilog_defaults -add -I{folder}',
            f'read_verilog -sv {ROOT / "hardware/linear_classifier.sv"}',
            f'read_verilog -sv {ROOT / "hardware/linear_classifier_top.sv"}',
            f'hierarchy -check -top linear_classifier_top -chparam BITS {bits} -chparam N {features}',
            # -noabc: synth's built-in abc uses a slow script on the wide multiplier
            # The explicit abc line below does the liberty mapping instead
            'synth -top linear_classifier_top -noabc',
            'flatten',
            f'dfflibmap -liberty {lib}',
            f'abc -liberty {lib} -constr {constr} -script {abc_script}',
            f'stat -liberty {lib}',
            f'write_verilog -noattr {out / "netlist.v"}',
        ]) + '\n',
        encoding='utf-8',
    )
    log_path = out / 'yosys.log'
    subprocess.run(['yosys', '-q', '-l', str(log_path), '-s', str(script)], check=True)
    log = log_path.read_text(encoding='utf-8', errors='replace')
    delays = [float(x) for x in re.findall(r'Delay\s*=\s*([0-9.]+)\s*ps', log)]

    if not delays:
        raise SystemExit(f'ABC did not report a path delay for k={features} bits={bits}. See {log_path}')

    comb_ns = max(delays) / 1000.0
    stat = log.split('Printing statistics.')[-1]
    cells_match = re.search(r'^\s+(\d+)\s+\S+\s+cells\s*$', stat, re.M)
    area_match = re.search(r"Chip area for module '\\linear_classifier_top':\s+([0-9.]+)", stat)

    if not cells_match or not area_match:
        raise SystemExit(f'Could not parse Yosys stat for k={features} bits={bits}')

    scope = 0
    scope_match = re.search(r'^\s+(\d+)\s+\S+\s+\$scopeinfo\s*$', stat, re.M)

    if scope_match:
        scope = int(scope_match.group(1))

    flops = 0

    for count, name in re.findall(r'^\s+(\d+)\s+\S+\s+(DFF\w+)\s*$', stat, re.M):
        flops += int(count)

    expected_flops = features * bits + 1

    if flops != expected_flops:
        raise SystemExit(f'Expected {expected_flops} flip-flops, Yosys reported {flops}')

    critical_ns = comb_ns + clk_to_q + setup

    return {
        'cell_count': int(cells_match.group(1)) - scope,
        'flip_flop_count': flops,
        'area_um2': float(area_match.group(1)),
        'comb_delay_ns': comb_ns,
        'clk_to_q_ns': clk_to_q,
        'setup_ns': setup,
        'critical_path_ns': critical_ns,
        'fmax_mhz': 1000.0 / critical_ns,
    }


def choose_tradeoff(rows):
    exact = [row for row in rows if abs(row['agreement_with_float'] - 1.0) < 1e-9]
    pool = exact or rows

    return min(pool, key=lambda row: (row['screening_cost'], row['area_um2'], row['bits'], row['features']))


def write_reports(rows, yosys_version):
    fields = [
        'features', 'bits', 'recall', 'false_positive_rate', 'screening_cost',
        'agreement_with_float', 'latency_cycles', 'cell_count', 'flip_flop_count',
        'area_um2', 'comb_delay_ns', 'clk_to_q_ns', 'setup_ns', 'critical_path_ns',
        'fmax_mhz', 'wrapper_simulation',
    ]

    with OUT_CSV.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    best = choose_tradeoff(rows)
    lines = [
        '# Step 5 hardware cost',
        '',
        'Clocked wrapper `hardware/linear_classifier_top.sv` around the existing',
        'combinational classifier. Each row is one logistic-regression export.',
        'Validation metrics are the same quantities as `reports/step4_quantization.csv`',
        '(failure recall, false-positive rate, illustrative screening cost).',
        'The held-out test set was not scored.',
        '',
        f'Synthesis: {yosys_version}. Library: Nangate45 typical',
        '(`NangateOpenCellLibrary`, 45 nm). Area is the liberty cell area in µm².',
        'The critical path is the ABC longest path from a flip-flop Q to a flip-flop D,',
        'plus the DFF_X1 clock-to-Q and setup taken from the middle of those liberty tables.',
        f'ABC load on timing endpoints is the DFF_X1 D-pin capacitance ({DFF_D_LOAD_FF} fF).',
        'Mapping uses ABC\'s direct `map` command rather than Yosys\' slower resizer.',
        'A longer optimizer run could shrink the area or the path a bit.',
        'There is no wire-load model and no clock tree. These are estimates, not a tapeout.',
        '',
        '| Features | Bits | Recall | False-positive rate | Screening cost | Float agreement | Latency (cycles) | Cells | Flip-flops | Area (µm²) | Critical path (ns) | fmax (MHz) |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    for row in rows:
        lines.append(
            '| {features} | {bits} | {recall:.4f} | {false_positive_rate:.4f} | {screening_cost} | '
            '{agreement_with_float:.4f} | {latency_cycles} | {cell_count} | {flip_flop_count} | '
            '{area_um2:.1f} | {critical_path_ns:.3f} | {fmax_mhz:.1f} |'.format(**row)
        )

    lines.extend([
        '',
        f"Best validation/hardware tradeoff under this estimate: **{best['features']} features, "
        f"{best['bits']} bits** (screening cost {best['screening_cost']}, area {best['area_um2']:.0f} µm², "
        f"about {best['fmax_mhz']:.0f} MHz). The rule is: among rows that match every floating-point "
        'validation decision, pick the lowest screening cost and then the smallest area.',
        '',
        'Latency is 1 clock cycle for every row: the feature word is registered on one rising edge',
        'and the failure bit is registered on the next rising edge.',
        '',
    ])

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')

    return best


def main():
    require_tool('yosys')
    require_tool('iverilog')
    require_tool('vvp')

    for features in FEATURE_COUNTS:
        model = ROOT / 'artifacts' / f'step3_logistic_k{features:02d}.joblib'

        if not model.is_file():
            raise SystemExit(f'{model} is missing. Run python scripts/03_train_compare.py first.')

    lib = ensure_liberty()
    clk_to_q, setup = dff_timing(lib.read_text(encoding='utf-8', errors='replace'))
    saved_step4 = STEP4_CSV.read_bytes() if STEP4_CSV.is_file() else None
    version = subprocess.run(['yosys', '-V'], check=True, capture_output=True, text=True).stdout.splitlines()[0].strip()
    version = version.split(' (', 1)[0]
    rows = []

    try:
        for features in FEATURE_COUNTS:
            for bits in BIT_WIDTHS:
                subprocess.run(
                    [sys.executable, str(ROOT / 'scripts/04_export_fixed_point.py'),
                     '--features', str(features), '--bits', str(bits)],
                    check=True,
                )
                quality = read_step4_row()
                folder = ROOT / 'hardware' / 'generated' / f'k{features:02d}_b{bits:02d}'
                samples = int(quality['validation_samples'])
                latency = simulate_wrapper(folder, features, bits, samples)
                hardware = synthesize(folder, features, bits, lib, clk_to_q, setup)
                row = {
                    'features': features,
                    'bits': bits,
                    'recall': float(quality['recall']),
                    'false_positive_rate': float(quality['false_positive_rate']),
                    'screening_cost': int(float(quality['screening_cost'])),
                    'agreement_with_float': float(quality['agreement_with_float']),
                    'latency_cycles': latency,
                    'wrapper_simulation': 'passed',
                    **hardware,
                }
                rows.append(row)
                print(
                    f"k={features:02d} bits={bits:02d} recall={row['recall']:.3f} "
                    f"fpr={row['false_positive_rate']:.3f} cost={row['screening_cost']} "
                    f"cells={row['cell_count']} flops={row['flip_flop_count']} "
                    f"area={row['area_um2']:.1f} critical_path_ns={row['critical_path_ns']:.3f} "
                    f"fmax_mhz={row['fmax_mhz']:.1f} latency={latency}"
                )

    finally:
        if saved_step4 is None:
            if STEP4_CSV.is_file() and not rows:
                STEP4_CSV.unlink()
        else:
            STEP4_CSV.write_bytes(saved_step4)

    if len(rows) != len(FEATURE_COUNTS) * len(BIT_WIDTHS):
        raise SystemExit('Sweep did not finish every features/bits point.')
        
    best = write_reports(rows, version)
    print(f'\nSaved: {OUT_CSV}')
    print(f'Saved: {OUT_MD}')
    print(f"Recommended: {best['features']} features, {best['bits']} bits.")
    print('Step 4 CSV was left as it was before this sweep. TEST was not evaluated.')


if __name__ == '__main__':
    main()
