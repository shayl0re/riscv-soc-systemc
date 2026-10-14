// 4-bit LFSR (x^4 + x^3 + 1) and output-privacy XOR stage
module lfsr_noise (input clk, input privacy, input [3:0] cls, output reg [3:0] out);
  reg [3:0] s;
  initial s = 4'd9;
  always @(posedge clk) begin
    s   <= {s[2:0], s[3] ^ s[2]};
    out <= privacy ? (cls ^ s) : cls;
  end
endmodule
