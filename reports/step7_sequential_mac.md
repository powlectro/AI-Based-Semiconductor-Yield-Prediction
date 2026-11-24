# Sequential MAC comparison

Same logistic coefficients and the same integer decision as the parallel
classifier. The sequential unit keeps one multiplier. A sample load
places the constant weights in a shift register beside the features,
and each later cycle multiplies the lane at the least-significant bits.
The decision is registered N cycles after the sample is accepted, and
the next sample can be accepted on that same edge, so the initiation
interval is N cycles.

Synthesis: Yosys 0.71+. Library and path method match Step 5:
Nangate45 typical, ABC `map`, DFF_X1 clock-to-Q and setup, no wire-load
model and no clock tree. Throughput is fmax / initiation interval.
These are estimates, not a tapeout. The held-out test set was not used
to choose the architecture.

The sequential critical path grows with the shift-register width. This
mapper does not rebuffer the select that steers those registers, so part
of that delay is control-net fanout. The parallel numbers use the same
unbuffered `map` script. A re-synthesis of the parallel 16×12 wrapper
matches the published Step 5 area and delay.

Frozen point, 16 features and 12 bits. Parallel area 11627.7 µm²,
critical path 2.391 ns, latency 1 cycle,
throughput 418.2 million samples/s.
Sequential area 9550.7 µm², critical path 11.095 ns,
latency 16 cycles, throughput 5.6 million samples/s.
The parallel design is 1.22× the sequential area and 74.25× the sequential throughput.

Re-synthesis of the parallel 16×12 wrapper in this environment: 11627.7 µm², critical path 2.391 ns, fmax 418.2 MHz. Published Step 5 area is 11627.7 µm².

| Architecture | Features | Bits | Recall | Screening cost | Latency (cycles) | Interval (cycles) | Throughput (Msamples/s) | Area (µm²) | Critical path (ns) | fmax (MHz) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parallel | 8 | 8 | 0.4286 | 181 | 1 | 1 | 628.8 | 2398.0 | 1.590 | 628.8 |
| sequential | 8 | 8 | 0.4286 | 181 | 8 | 8 | 43.0 | 5716.3 | 2.908 | 343.9 |
| parallel | 8 | 12 | 0.4286 | 178 | 1 | 1 | 503.8 | 5735.8 | 1.985 | 503.8 |
| sequential | 8 | 12 | 0.4286 | 178 | 8 | 8 | 31.6 | 7467.2 | 3.958 | 252.6 |
| parallel | 8 | 16 | 0.4286 | 178 | 1 | 1 | 442.1 | 10406.5 | 2.262 | 442.1 |
| sequential | 8 | 16 | 0.4286 | 178 | 8 | 8 | 23.2 | 9245.6 | 5.399 | 185.2 |
| parallel | 16 | 8 | 0.5714 | 176 | 1 | 1 | 545.5 | 5182.5 | 1.833 | 545.5 |
| sequential | 16 | 8 | 0.5714 | 176 | 16 | 16 | 9.0 | 7109.6 | 6.962 | 143.6 |
| parallel | 16 | 12 | 0.5714 | 175 | 1 | 1 | 418.2 | 11627.7 | 2.391 | 418.2 |
| sequential | 16 | 12 | 0.5714 | 175 | 16 | 16 | 5.6 | 9550.7 | 11.095 | 90.1 |
| parallel | 16 | 16 | 0.5714 | 176 | 1 | 1 | 401.9 | 21046.5 | 2.488 | 401.9 |
| sequential | 16 | 16 | 0.5714 | 176 | 16 | 16 | 3.9 | 12021.6 | 16.123 | 62.0 |
| parallel | 32 | 8 | 0.4286 | 191 | 1 | 1 | 334.1 | 14977.4 | 2.993 | 334.1 |
| sequential | 32 | 8 | 0.4286 | 191 | 32 | 32 | 2.2 | 9923.9 | 14.348 | 69.7 |
| parallel | 32 | 12 | 0.4762 | 178 | 1 | 1 | 372.4 | 21352.4 | 2.685 | 372.4 |
| sequential | 32 | 12 | 0.4762 | 178 | 32 | 32 | 1.2 | 13765.2 | 26.415 | 37.9 |
| parallel | 32 | 16 | 0.4762 | 178 | 1 | 1 | 333.9 | 41095.7 | 2.995 | 333.9 |
| sequential | 32 | 16 | 0.4762 | 178 | 32 | 32 | 0.7 | 17636.1 | 42.763 | 23.4 |

Validation recall and screening cost are properties of the quantized logistic
decision, so they match across the two architectures when the RTL comparison passes.
Equivalence was checked on every validation vector at each features/bits point.
