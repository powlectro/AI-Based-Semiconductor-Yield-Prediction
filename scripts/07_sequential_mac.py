"""Compare the sequential MAC with the parallel 1-cycle classifier.

For every logistic export in {8, 16, 32} features × {8, 12, 16} bits this
script:
  1. Exports validation vectors and quantized constants without touching test.
  2. Verifies bit-exact equivalence of `hardware/linear_classifier_seq.sv`
     against Python in Icarus Verilog (`hardware/classifier_seq_tb.sv`).
  3. Synthesizes the sequential MAC with Yosys + Nangate45 using the same
     ABC `map` script and DFF_X1 timing conventions as Step 5.
  4. Re-synthesizes the frozen 16×12 parallel wrapper as a consistency check.

The held-out test partition is not scored, and `reports/step4_quantization.csv`
is left unchanged.
"""
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fixed_point import make_fixed_model, quantize, quantized_linear_predict

FEATURE_COUNTS = (8, 16, 32)
BIT_WIDTHS = (8, 12, 16)
FROZEN = (16, 12)
REPORTS = ROOT / 'reports'
SYNTH_DIR = ROOT / 'hardware' / 'synth'
OUT_CSV = REPORTS / 'step7_sequential_mac.csv'
OUT_MD = REPORTS / 'step7_sequential_mac.md'


def load_module(name, path):
    """Load a numbered pipeline script as a module to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_quality(export, features, bits):
    """Export one (features, bits) validation point and compute its fixed-point metrics."""
    import joblib
    model_file = ROOT / 'artifacts' / f'step3_logistic_k{features:02d}.joblib'
    if not model_file.is_file():
        raise SystemExit(f'{model_file} is missing. Run scripts/06_freeze_classifier.py first.')
    loaded = joblib.load(model_file)  # locally generated, trusted artifact only
    X, y, _timestamps = validation_quality.dataset
    split = export.load_indices(ROOT / 'reports' / 'split_assignments.csv', len(X))

    # Quantization scale is fitted from train rows; evaluation uses validation rows only.
    train = loaded['preprocessor'].transform(X.iloc[split['train']])
    val = loaded['preprocessor'].transform(X.iloc[split['validation']])
    y_val = y.iloc[split['validation']].to_numpy()
    spec = make_fixed_model(loaded['model'], train, bits, float(loaded['threshold']))

    folder = ROOT / 'hardware' / 'generated' / f'k{features:02d}_b{bits:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    x_int = quantize(val, bits, spec['xfrac'])
    fixed_pred = quantized_linear_predict(x_int, spec['weights'], spec['offset'])
    export.write_rtl_constants(folder, spec)
    export.write_vectors(folder, x_int, fixed_pred, bits)

    float_pred = loaded['model'].predict_proba(val)[:, 1] >= float(loaded['threshold'])
    tp = int((fixed_pred & (y_val == 1)).sum())
    fn = int((~fixed_pred & (y_val == 1)).sum())
    fp = int((fixed_pred & (y_val == 0)).sum())
    tn = int((~fixed_pred & (y_val == 0)).sum())
    return folder, {
        'features': features,
        'bits': bits,
        'samples': len(val),
        'recall': tp / (tp + fn),
        'false_positive_rate': fp / (fp + tn),
        'screening_cost': 10 * fn + fp,
        'agreement_with_float': float((float_pred == fixed_pred).mean()),
        'bundle': loaded,
        'spec': spec,
    }


def simulate_sequential(folder, features, bits, samples):
    """Run Icarus Verilog on classifier_seq_tb and verify latency and interval equal N."""
    binary = folder / 'classifier_seq_sim'
    cmd = [
        'iverilog', '-g2012', '-s', 'classifier_seq_tb',
        f'-DBITS={bits}', f'-DN={features}', f'-DSAMPLES={samples}',
        '-I', str(folder), '-o', str(binary),
        str(ROOT / 'hardware' / 'linear_classifier_seq.sv'),
        str(ROOT / 'hardware' / 'classifier_seq_tb.sv'),
    ]
    subprocess.run(cmd, cwd=folder, check=True)
    run = subprocess.run(['vvp', str(binary)], cwd=folder, check=True, capture_output=True, text=True)
    print(run.stdout.strip())
    match = re.search(
        r'PASS: (\d+) Python/RTL vector comparisons, latency_cycles=(\d+), interval_cycles=(\d+)',
        run.stdout,
    )
    if not match or int(match.group(1)) != samples:
        raise SystemExit(f'Sequential simulation failed for k={features} bits={bits}')
    latency = int(match.group(2))
    interval = int(match.group(3))
    if latency != features or interval != features:
        raise SystemExit(f'Expected latency and interval {features}, got {latency}, {interval}')
    return latency, interval


def synthesize(step5, folder, features, bits, lib, clk_to_q, setup, top, verilog_files):
    """Map one RTL module to Nangate45 with Yosys/ABC and parse cell area and fmax."""
    out = SYNTH_DIR / f'{top}_k{features:02d}_b{bits:02d}'
    out.mkdir(parents=True, exist_ok=True)

    # Match Step 5 constraints: BUF_X1 driver, DFF_X1 D-pin load, and direct ABC `map`.
    constr = out / 'nangate.constr'
    constr.write_text(f'set_driving_cell BUF_X1\nset_load {step5.DFF_D_LOAD_FF}\n', encoding='ascii')
    abc_script = out / 'map.abc'
    abc_script.write_text('strash\nmap\ntopo\nstime -p\n', encoding='ascii')

    script = out / 'synth.ys'
    lines = [f'read_liberty -lib {lib}', f'verilog_defaults -add -I{folder}']
    for path in verilog_files:
        lines.append(f'read_verilog -sv {path}')
    lines.extend([
        f'hierarchy -check -top {top} -chparam BITS {bits} -chparam N {features}',
        f'synth -top {top} -noabc',
        'flatten',
        f'dfflibmap -liberty {lib}',
        f'abc -liberty {lib} -constr {constr} -script {abc_script}',
        f'stat -liberty {lib}',
        f'write_verilog -noattr {out / "netlist.v"}',
    ])
    script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log_path = out / 'yosys.log'
    subprocess.run(['yosys', '-q', '-l', str(log_path), '-s', str(script)], check=True)

    # Extract worst combinational path delay (ps) and cell/flop/area totals from the Yosys log.
    log = log_path.read_text(encoding='utf-8', errors='replace')
    delays = [float(value) for value in re.findall(r'Delay\s*=\s*([0-9.]+)\s*ps', log)]
    if not delays:
        raise SystemExit(f'ABC did not report a path delay for {top} k={features} bits={bits}')
    comb_ns = max(delays) / 1000.0
    stat = log.split('Printing statistics.')[-1]
    cells_match = re.search(r'^\s+(\d+)\s+\S+\s+cells\s*$', stat, re.M)
    area_match = re.search(
        rf"Chip area for module '\\{top}':\s+([0-9.]+)",
        stat,
    )
    if not cells_match or not area_match:
        raise SystemExit(f'Could not parse Yosys stat for {top} k={features} bits={bits}')

    # Exclude Yosys's synthetic $scopeinfo metadata cell from physical cell count.
    scope = 0
    scope_match = re.search(r'^\s+(\d+)\s+\S+\s+\$scopeinfo\s*$', stat, re.M)
    if scope_match:
        scope = int(scope_match.group(1))
    flops = sum(int(count) for count, _name in re.findall(r'^\s+(\d+)\s+\S+\s+(DFF\w+)\s*$', stat, re.M))

    # Full register-to-register clock period = combinational delay + DFF clk-to-Q + DFF setup.
    critical_ns = comb_ns + clk_to_q + setup
    return {
        'cell_count': int(cells_match.group(1)) - scope,
        'flip_flop_count': flops,
        'area_um2': float(area_match.group(1)),
        'comb_delay_ns': comb_ns,
        'critical_path_ns': critical_ns,
        'fmax_mhz': 1000.0 / critical_ns,
    }


def published_parallel():
    """Load the Step 5 parallel hardware sweep and attach its 1-cycle throughput."""
    with (REPORTS / 'step5_hardware_cost.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        fmax = float(row['fmax_mhz'])
        parsed.append({
            'architecture': 'parallel',
            'source': 'reports/step5_hardware_cost.csv',
            'features': int(row['features']),
            'bits': int(row['bits']),
            'recall': float(row['recall']),
            'false_positive_rate': float(row['false_positive_rate']),
            'screening_cost': int(float(row['screening_cost'])),
            'agreement_with_float': float(row['agreement_with_float']),
            'latency_cycles': int(row['latency_cycles']),
            'interval_cycles': 1,
            'throughput_msps': fmax / 1.0,
            'cell_count': int(row['cell_count']),
            'flip_flop_count': int(row['flip_flop_count']),
            'area_um2': float(row['area_um2']),
            'critical_path_ns': float(row['critical_path_ns']),
            'fmax_mhz': fmax,
            'simulation': row['wrapper_simulation'],
        })
    return parsed


def write_reports(rows, yosys_version, resynth):
    """Write the combined parallel + sequential CSV and Markdown comparison table."""
    fields = [
        'architecture', 'source', 'features', 'bits', 'recall', 'false_positive_rate',
        'screening_cost', 'agreement_with_float', 'latency_cycles', 'interval_cycles',
        'throughput_msps', 'cell_count', 'flip_flop_count', 'area_um2',
        'critical_path_ns', 'fmax_mhz', 'simulation',
    ]
    with OUT_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    frozen_parallel = next(row for row in rows if row['architecture'] == 'parallel' and (row['features'], row['bits']) == FROZEN)
    frozen_seq = next(row for row in rows if row['architecture'] == 'sequential' and (row['features'], row['bits']) == FROZEN)
    area_ratio = frozen_parallel['area_um2'] / frozen_seq['area_um2']
    throughput_ratio = frozen_parallel['throughput_msps'] / frozen_seq['throughput_msps']
    lines = [
        '# Sequential MAC comparison',
        '',
        'Same logistic coefficients and the same integer decision as the parallel',
        'classifier. The sequential unit keeps one multiplier. A sample load',
        'places the constant weights in a shift register beside the features,',
        'and each later cycle multiplies the lane at the least-significant bits.',
        'The decision is registered N cycles after the sample is accepted, and',
        'the next sample can be accepted on that same edge, so the initiation',
        'interval is N cycles.',
        '',
        f'Synthesis: {yosys_version}. Library and path method match Step 5:',
        'Nangate45 typical, ABC `map`, DFF_X1 clock-to-Q and setup, no wire-load',
        'model and no clock tree. Throughput is fmax / initiation interval.',
        'These are estimates, not a tapeout. The held-out test set was not used',
        'to choose the architecture.',
        '',
        'The sequential critical path grows with the shift-register width. This',
        'mapper does not rebuffer the select that steers those registers, so part',
        'of that delay is control-net fanout. The parallel numbers use the same',
        'unbuffered `map` script. A re-synthesis of the parallel 16×12 wrapper',
        'matches the published Step 5 area and delay.',
        '',
        f"Frozen point, 16 features and 12 bits. Parallel area {frozen_parallel['area_um2']:.1f} µm²,",
        f"critical path {frozen_parallel['critical_path_ns']:.3f} ns, latency {frozen_parallel['latency_cycles']} cycle,",
        f"throughput {frozen_parallel['throughput_msps']:.1f} million samples/s.",
        f"Sequential area {frozen_seq['area_um2']:.1f} µm², critical path {frozen_seq['critical_path_ns']:.3f} ns,",
        f"latency {frozen_seq['latency_cycles']} cycles, throughput {frozen_seq['throughput_msps']:.1f} million samples/s.",
        f"The parallel design is {area_ratio:.2f}× the sequential area and {throughput_ratio:.2f}× the sequential throughput.",
        '',
        f"Re-synthesis of the parallel 16×12 wrapper in this environment: "
        f"{resynth['area_um2']:.1f} µm², critical path {resynth['critical_path_ns']:.3f} ns, "
        f"fmax {resynth['fmax_mhz']:.1f} MHz. Published Step 5 area is {frozen_parallel['area_um2']:.1f} µm².",
        '',
        '| Architecture | Features | Bits | Recall | Screening cost | Latency (cycles) | Interval (cycles) | Throughput (Msamples/s) | Area (µm²) | Critical path (ns) | fmax (MHz) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    ordered = sorted(rows, key=lambda row: (row['features'], row['bits'], row['architecture']))
    for row in ordered:
        lines.append(
            '| {architecture} | {features} | {bits} | {recall:.4f} | {screening_cost} | '
            '{latency_cycles} | {interval_cycles} | {throughput_msps:.1f} | {area_um2:.1f} | '
            '{critical_path_ns:.3f} | {fmax_mhz:.1f} |'.format(**row)
        )
    lines.extend([
        '',
        'Validation recall and screening cost are properties of the quantized logistic',
        'decision, so they match across the two architectures when the RTL comparison passes.',
        'Equivalence was checked on every validation vector at each features/bits point.',
        '',
    ])
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')


def main():
    export = load_module('export_fixed_point', ROOT / 'scripts' / '04_export_fixed_point.py')
    step5 = load_module('step5_hardware_cost', ROOT / 'scripts' / '05_hardware_cost.py')
    step5.require_tool('yosys')
    step5.require_tool('iverilog')
    step5.require_tool('vvp')

    # Load SECOM once and cache on the helper to avoid re-reading the zip 9 times.
    from src.secom_io import load_secom
    validation_quality.dataset = load_secom(ROOT / 'data' / 'raw' / 'secom.zip')
    lib = step5.ensure_liberty()
    clk_to_q, setup = step5.dff_timing(lib.read_text(encoding='utf-8', errors='replace'))
    version = subprocess.run(['yosys', '-V'], check=True, capture_output=True, text=True).stdout.splitlines()[0].strip()
    version = version.split(' (', 1)[0]

    rows = published_parallel()
    sequential_verilog = [ROOT / 'hardware' / 'linear_classifier_seq.sv']
    parallel_verilog = [
        ROOT / 'hardware' / 'linear_classifier.sv',
        ROOT / 'hardware' / 'linear_classifier_top.sv',
    ]

    # Sweep all 9 (features, bits) combinations for the sequential MAC.
    for features in FEATURE_COUNTS:
        for bits in BIT_WIDTHS:
            folder, quality = validation_quality(export, features, bits)
            latency, interval = simulate_sequential(folder, features, bits, quality['samples'])
            hardware = synthesize(
                step5, folder, features, bits, lib, clk_to_q, setup,
                'linear_classifier_seq', sequential_verilog,
            )
            row = {
                'architecture': 'sequential',
                'source': 'yosys this run',
                'features': features,
                'bits': bits,
                'recall': quality['recall'],
                'false_positive_rate': quality['false_positive_rate'],
                'screening_cost': quality['screening_cost'],
                'agreement_with_float': quality['agreement_with_float'],
                'latency_cycles': latency,
                'interval_cycles': interval,
                'throughput_msps': hardware['fmax_mhz'] / interval,
                'simulation': 'passed',
                **hardware,
            }
            rows.append(row)
            print(
                f"sequential k={features:02d} bits={bits:02d} "
                f"area={row['area_um2']:.1f} path={row['critical_path_ns']:.3f} ns "
                f"fmax={row['fmax_mhz']:.1f} throughput={row['throughput_msps']:.1f} "
                f"latency={latency}"
            )

    # Re-synthesize the frozen parallel 16x12 wrapper to confirm identical tool behavior.
    frozen_folder = ROOT / 'hardware' / 'generated' / 'k16_b12'
    resynth = synthesize(
        step5, frozen_folder, 16, 12, lib, clk_to_q, setup,
        'linear_classifier_top', parallel_verilog,
    )
    print(
        f"parallel resynth k=16 bits=12 area={resynth['area_um2']:.1f} "
        f"path={resynth['critical_path_ns']:.3f} ns fmax={resynth['fmax_mhz']:.1f}"
    )
    write_reports(rows, version, resynth)
    print(f'Saved: {OUT_CSV}')
    print(f'Saved: {OUT_MD}')
    print('TEST was not evaluated.')


if __name__ == '__main__':
    main()
