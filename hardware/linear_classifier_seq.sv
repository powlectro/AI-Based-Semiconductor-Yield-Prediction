// Sequential signed fixed-point linear classifier (1-multiplier MAC).
//
// Computes the exact same integer decision as the parallel linear_classifier:
//     sum_{i=0}^{N-1}(x_i * w_i) + OFFSET >= 0
//
// Input packing:
//   features_flat packs N signed BITS-wide sensor values, with feature 0 in
//   the least-significant BITS bits. All values are already train-median
//   imputed, standardized, and quantized in Python.
//
// Handshake and timing:
//   - A new sample is accepted on a rising clk edge when `start` is high and
//     the unit is idle (or on the final accumulation cycle of the previous
//     sample, enabling back-to-back streaming).
//   - `done` pulses high and `failure` updates N cycles after acceptance.
//   - Latency = N cycles, initiation interval = N cycles, throughput = fmax / N.
//
// Generated model_constants.svh (OFFSET and weight_at()) must be on the include path.
module linear_classifier_seq #(
    parameter integer BITS = 16,
    parameter integer N = 16
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire [N*BITS-1:0] features_flat,
    output reg failure,
    output reg done
);
`include "model_constants.svh"
    // Bit width needed to count 0 .. N-1 MAC iterations.
    localparam integer IDX_W = (N <= 2) ? 1 : $clog2(N);

    // State registers:
    //   features_q / weights_q : shift registers holding remaining sensor/weight pairs
    //   acc                    : 64-bit signed running sum (matches Python int64)
    //   idx                    : current MAC step (0 .. N-1)
    //   busy                   : 1 while accumulating a sample
    reg [N*BITS-1:0] features_q;
    reg [N*BITS-1:0] weights_q;
    reg signed [63:0] acc;
    reg [IDX_W-1:0] idx;
    reg busy;

    // Pack the constant weights from model_constants.svh into a flat bus at
    // elaboration time so loading weights_q on `start` uses pure constants.
    wire [N*BITS-1:0] weight_word;
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : weight_pack
            assign weight_word[gi*BITS +: BITS] = weight_at(gi);
        end
    endgenerate

    // Combinational MAC datapath: always multiply the LSB lane (feature idx, weight idx).
    // Shifting right by BITS each cycle avoids an N-to-1 mux in front of the multiplier.
    wire signed [BITS-1:0] x_now = $signed(features_q[BITS-1:0]);
    wire signed [BITS-1:0] w_now = $signed(weights_q[BITS-1:0]);

    // Sign-extend operands to 2*BITS so synthesis builds one BITS x BITS signed
    // multiplier rather than a full 64 x 64 multiplier, then sign-extend to 64 bits.
    wire signed [2*BITS-1:0] x_mul = {{BITS{x_now[BITS-1]}}, x_now};
    wire signed [2*BITS-1:0] w_mul = {{BITS{w_now[BITS-1]}}, w_now};
    wire signed [2*BITS-1:0] product_narrow = x_mul * w_mul;
    wire signed [63:0] product = {{(64 - 2*BITS){product_narrow[2*BITS-1]}}, product_narrow};
    wire signed [63:0] sum = acc + product;

    always @(posedge clk) begin
        if (rst) begin
            // Synchronous active-high reset clears state and outputs.
            features_q <= '0;
            weights_q <= '0;
            acc <= 64'sd0;
            idx <= '0;
            busy <= 1'b0;
            done <= 1'b0;
            failure <= 1'b0;
        end else if (busy && idx == N - 1) begin
            // Final MAC step: register the classification decision (`sum >= 0`)
            // and pulse `done`. If `start` is already asserted, accept the next
            // sample on this same clock edge for zero idle cycles between samples.
            failure <= (sum >= 0);
            done <= 1'b1;
            if (start) begin
                features_q <= features_flat;
                weights_q <= weight_word;
                acc <= OFFSET;
                idx <= '0;
                busy <= 1'b1;
            end else begin
                busy <= 1'b0;
            end
        end else if (busy) begin
            // Intermediate MAC step: update accumulator and shift the next
            // feature/weight pair down into bits [BITS-1:0].
            acc <= sum;
            features_q <= features_q >> BITS;
            weights_q <= weights_q >> BITS;
            idx <= idx + 1'b1;
            done <= 1'b0;
        end else if (start) begin
            // Idle -> active: latch incoming feature vector, preload weights,
            // and initialize the accumulator with the quantized bias/threshold OFFSET.
            features_q <= features_flat;
            weights_q <= weight_word;
            acc <= OFFSET;
            idx <= '0;
            busy <= 1'b1;
            done <= 1'b0;
        end else begin
            // Stay idle and deassert the single-cycle `done` pulse.
            done <= 1'b0;
        end
    end
endmodule
