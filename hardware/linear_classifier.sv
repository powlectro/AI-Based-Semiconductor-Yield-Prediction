// Combinational signed fixed-point linear classifier.
// FEATURES input format: feature 0 is the least-significant BITS bits.
// All values are standardized and quantized by the Python preprocessing path.
// Generated model_constants.svh must be on the include path.
module linear_classifier #(
    parameter integer BITS = 16,
    parameter integer N = 16
) (
    input wire [N*BITS-1:0] features_flat,
    output reg failure
);
`include "model_constants.svh"
    integer i;
    reg signed [63:0] accumulator;
    reg signed [BITS-1:0] x;
    always @* begin
        accumulator = OFFSET;
        x = '0;
        for (i = 0; i < N; i = i + 1) begin
            x = $signed(features_flat[i*BITS +: BITS]);
            accumulator = accumulator + ($signed(x) * $signed(weight_at(i)));
        end
        failure = (accumulator >= 0);
    end
endmodule
