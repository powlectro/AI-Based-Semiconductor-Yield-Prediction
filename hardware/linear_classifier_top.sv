// Clocked wrapper around linear_classifier
// Feature words are captured on the rising edge of clk. The existing combinational classifier then evaluates those registered features, and
// the failure bit is captured on the next rising edge. Latency is therefore 1 clock cycle. rst is synchronous and active-high
// FEATURES input format matches linear_classifier: feature 0 is in the least-significant BITS bits. Preprocessing stays outside this module
module linear_classifier_top #(
    parameter integer BITS = 16,
    parameter integer N = 16
) (
    input wire clk,
    input wire rst,
    input wire [N*BITS-1:0] features_flat,
    output reg failure
);
    localparam integer LATENCY_CYCLES = 1;
    reg [N*BITS-1:0] features_q;
    wire failure_c;

    linear_classifier #(.BITS(BITS), .N(N)) u_classifier (
        .features_flat(features_q),
        .failure(failure_c)
    );

    always @(posedge clk) begin
        if (rst) begin
            features_q <= '0;
            failure <= 1'b0;
        end else begin
            features_q <= features_flat;
            // failure_c still reflects the previous features_q at this edge,
            // so the registered decision trails the registered inputs by LATENCY_CYCLES
            failure <= failure_c;
        end
    end
endmodule
