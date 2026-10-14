// MP PE (paper Fig. 2b): max of four values, combinational
module maxpool_pe (input [7:0] i0, i1, i2, i3, output [7:0] maxout);
  wire [7:0] m01 = (i0 > i1) ? i0 : i1;
  wire [7:0] m23 = (i2 > i3) ? i2 : i3;
  assign maxout = (m01 > m23) ? m01 : m23;
endmodule
