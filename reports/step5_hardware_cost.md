# Step 5 hardware cost

Clocked wrapper `hardware/linear_classifier_top.sv` around the existing
combinational classifier. Each row is one logistic-regression export.
Validation metrics are the same quantities as `reports/step4_quantization.csv`
(failure recall, false-positive rate, illustrative screening cost).
The held-out test set was not scored.

Synthesis: Yosys 0.71+. Library: Nangate45 typical
(`NangateOpenCellLibrary`, 45 nm). Area is the liberty cell area in µm².
The critical path is the ABC longest path from a flip-flop Q to a flip-flop D,
plus the DFF_X1 clock-to-Q and setup taken from the middle of those liberty tables.
ABC load on timing endpoints is the DFF_X1 D-pin capacitance (1.14029 fF).
Mapping uses ABC's direct `map` command rather than Yosys' slower resizer.
A longer optimizer run could shrink the area or the path a bit.
There is no wire-load model and no clock tree. These are estimates, not a tapeout.

| Features | Bits | Recall | False-positive rate | Screening cost | Float agreement | Latency (cycles) | Cells | Flip-flops | Area (µm²) | Critical path (ns) | fmax (MHz) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 0.4286 | 0.2082 | 181 | 0.9904 | 1 | 2264 | 65 | 2398.0 | 1.590 | 628.8 |
| 8 | 12 | 0.4286 | 0.1980 | 178 | 1.0000 | 1 | 5575 | 97 | 5735.8 | 1.985 | 503.8 |
| 8 | 16 | 0.4286 | 0.1980 | 178 | 1.0000 | 1 | 10307 | 129 | 10406.5 | 2.262 | 442.1 |
| 16 | 8 | 0.5714 | 0.2935 | 176 | 0.9841 | 1 | 4929 | 129 | 5182.5 | 1.833 | 545.5 |
| 16 | 12 | 0.5714 | 0.2901 | 175 | 1.0000 | 1 | 10847 | 193 | 11627.7 | 2.391 | 418.2 |
| 16 | 16 | 0.5714 | 0.2935 | 176 | 0.9968 | 1 | 20507 | 257 | 21046.5 | 2.488 | 401.9 |
| 32 | 8 | 0.4286 | 0.2423 | 191 | 0.9682 | 1 | 14317 | 257 | 14977.4 | 2.993 | 334.1 |
| 32 | 12 | 0.4762 | 0.2321 | 178 | 1.0000 | 1 | 19793 | 385 | 21352.4 | 2.685 | 372.4 |
| 32 | 16 | 0.4762 | 0.2321 | 178 | 1.0000 | 1 | 39541 | 513 | 41095.7 | 2.995 | 333.9 |

Best validation/hardware tradeoff under this estimate: **16 features, 12 bits** (screening cost 175, area 11628 µm², about 418 MHz). The rule is: among rows that match every floating-point validation decision, pick the lowest screening cost and then the smallest area.

Latency is 1 clock cycle for every row: the feature word is registered on one rising edge
and the failure bit is registered on the next rising edge.
