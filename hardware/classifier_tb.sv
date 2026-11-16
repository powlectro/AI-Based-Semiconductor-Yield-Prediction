// Compile with: iverilog -g2012 -s classifier_tb -I <generated_dir>
`ifndef BITS
`define BITS 16
`endif
`ifndef N
`define N 16
`endif
`ifndef SAMPLES
`define SAMPLES 100
`endif
module classifier_tb;
    localparam integer BITS = `BITS;
    localparam integer N = `N;
    localparam integer SAMPLES = `SAMPLES;
    reg [N*BITS-1:0] inputs;
    wire failure;
    reg [N*BITS:0] vectors [0:SAMPLES-1];
    integer i;
    integer errors;
    linear_classifier #(.BITS(BITS), .N(N)) dut (
        .features_flat(inputs), .failure(failure)
    );
    initial begin
        $readmemh("vectors.mem", vectors);
        errors = 0;
        for (i = 0; i < SAMPLES; i = i + 1) begin
            inputs = vectors[i][N*BITS-1:0];
            #1;
            if (failure !== vectors[i][N*BITS]) begin
                $display("MISMATCH sample=%0d got=%b expected=%b", i, failure, vectors[i][N*BITS]);
                errors = errors + 1;
            end
        end
        if (errors != 0) $fatal(1, "FAILED: %0d mismatches", errors);
        $display("PASS: %0d Python/RTL vector comparisons", SAMPLES);
        $finish;
    end
endmodule
