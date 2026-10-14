// MAC PE (paper Fig. 2a): signed 8x8 multiply -> 16-bit reg -> accumulate into 32-bit reg. 1 MAC/clock after 2-clock fill.
module mac_pe (input clk, input signed [7:0] a, b, input valid, clear, output reg signed [31:0] acc);
  reg signed [15:0] prod; reg v1, c1;
  always @(posedge clk) begin
    prod <= a * b; v1 <= valid; c1 <= clear;
    if (v1) acc <= c1 ? {{16{prod[15]}}, prod} : acc + {{16{prod[15]}}, prod};
  end
endmodule
