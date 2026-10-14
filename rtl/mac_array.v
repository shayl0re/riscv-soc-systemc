// N parallel MAC PEs (paper's 1 / 9 / 128 PE configurations)
module mac_array #(parameter N = 9) (input clk, input [8*N-1:0] a, b, input valid, clear, output [32*N-1:0] acc);
  genvar i;
  generate for (i = 0; i < N; i = i + 1) begin : pe
    mac_pe u (.clk(clk), .a(a[8*i +: 8]), .b(b[8*i +: 8]), .valid(valid), .clear(clear), .acc(acc[32*i +: 32]));
  end endgenerate
endmodule
