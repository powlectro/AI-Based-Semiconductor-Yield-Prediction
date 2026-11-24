// Testbench for linear_classifier_seq (sequential MAC).
// Compile with:
//   iverilog -g2012 -s classifier_seq_tb -DBITS=12 -DN=16 -DSAMPLES=314 \
//            -I <generated_dir> -o classifier_seq_sim \
//            hardware/linear_classifier_seq.sv hardware/classifier_seq_tb.sv
//
// Uses the same vectors.mem format as classifier_tb.sv:
//   bits [N*BITS-1:0] = packed quantized features (feature 0 at LSB)
//   bit  [N*BITS]     = bit-accurate expected failure prediction from Python
//
// Drives samples back-to-back every N cycles (initiation interval = N) and
// verifies both the `done` pulse timing and every predicted `failure` bit.
`ifndef BITS
`define BITS 16
`endif
`ifndef N
`define N 16
`endif
`ifndef SAMPLES
`define SAMPLES 100
`endif
module classifier_seq_tb;
    localparam integer BITS = `BITS;
    localparam integer N = `N;
    localparam integer SAMPLES = `SAMPLES;
    localparam integer LATENCY_CYCLES = N;
    localparam integer INTERVAL_CYCLES = N;

    reg clk;
    reg rst;
    reg start;
    reg [N*BITS-1:0] inputs;
    wire failure;
    wire done;
    reg [N*BITS:0] vectors [0:SAMPLES-1];
    integer i;
    integer errors;

    linear_classifier_seq #(.BITS(BITS), .N(N)) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .features_flat(inputs),
        .failure(failure),
        .done(done)
    );

    // 10-time-unit clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        $readmemh("vectors.mem", vectors);
        errors = 0;

        // Hold synchronous reset for two rising edges, then release on negedge
        // so stimulus changes cleanly away from the sampling edge.
        rst = 1'b1;
        start = 1'b0;
        inputs = '0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        // Stream all samples back-to-back with period N cycles.
        for (i = 0; i < SAMPLES; i = i + 1) begin
            inputs = vectors[i][N*BITS-1:0];
            start = 1'b1;
            @(posedge clk); // Cycle 0 of sample i (and final/output cycle of sample i-1)
            #1;
            if (i > 0) begin
                // Previous sample's decision must appear on this exact cycle.
                if (done !== 1'b1 || failure !== vectors[i-1][N*BITS]) begin
                    $display("MISMATCH sample=%0d got=%b expected=%b done=%b",
                        i - 1, failure, vectors[i-1][N*BITS], done);
                    errors = errors + 1;
                end
            end else if (done !== 1'b0) begin
                $display("MISMATCH done high before the first result");
                errors = errors + 1;
            end
            @(negedge clk);
            start = 1'b0;
            // Advance through the remaining N-1 accumulation cycles of sample i.
            repeat (N - 1) @(posedge clk);
            @(negedge clk);
        end

        // Final rising edge posts the last sample's decision (no new start asserted).
        @(posedge clk);
        #1;
        if (done !== 1'b1 || failure !== vectors[SAMPLES-1][N*BITS]) begin
            $display("MISMATCH sample=%0d got=%b expected=%b done=%b",
                SAMPLES - 1, failure, vectors[SAMPLES-1][N*BITS], done);
            errors = errors + 1;
        end

        // Confirm `done` drops back to 0 on the following idle cycle.
        @(posedge clk);
        #1;
        if (done !== 1'b0) begin
            $display("MISMATCH done stayed high after the last result");
            errors = errors + 1;
        end

        if (errors != 0) $fatal(1, "FAILED: %0d mismatches", errors);
        $display("PASS: %0d Python/RTL vector comparisons, latency_cycles=%0d, interval_cycles=%0d",
            SAMPLES, LATENCY_CYCLES, INTERVAL_CYCLES);
        $finish;
    end
endmodule
