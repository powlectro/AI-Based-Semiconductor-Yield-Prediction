// Compile with: iverilog -g2012 -s classifier_top_tb -I <generated_dir>
// Same vectors.mem packing as classifier_tb.sv (feature 0 at the LSB, expected decision at bit N*BITS). The wrapper latency is 1 cycle:
// the input word is captured on one rising edge and the registered failure bit is checked one rising edge later
`ifndef BITS
`define BITS 16
`endif

`ifndef N
`define N 16
`endif

`ifndef SAMPLES
`define SAMPLES 100
`endif

module classifier_top_tb;
    localparam integer BITS = `BITS;
    localparam integer N = `N;
    localparam integer SAMPLES = `SAMPLES;
    localparam integer LATENCY_CYCLES = 1;

    reg clk;
    reg rst;
    reg [N*BITS-1:0] inputs;
    wire failure;
    reg [N*BITS:0] vectors [0:SAMPLES-1];
    integer i;
    integer errors;

    linear_classifier_top #(.BITS(BITS), .N(N)) dut (
        .clk(clk), .rst(rst), .features_flat(inputs), .failure(failure)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        $readmemh("vectors.mem", vectors);
        errors = 0;
        rst = 1'b1;
        inputs = '0;

        repeat (2) @(posedge clk);
        @(negedge clk);

        rst = 1'b0;

        for (i = 0; i < SAMPLES; i = i + 1) begin
            inputs = vectors[i][N*BITS-1:0];
            @(posedge clk); // capture this feature word
            @(posedge clk); // registered failure is now this word's decision
            #1;

            if (failure !== vectors[i][N*BITS]) begin
                $display("MISMATCH sample=%0d got=%b expected=%b", i, failure, vectors[i][N*BITS]);
                errors = errors + 1;
            end
        end

        if (errors != 0) $fatal(1, "FAILED: %0d mismatches", errors);
        
        $display("PASS: %0d Python/RTL vector comparisons, latency_cycles=%0d", SAMPLES, LATENCY_CYCLES);
        $finish;
    end
endmodule
