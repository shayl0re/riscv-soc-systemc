// Authentication + privacy block together (what the paper calls the security and privacy module)
module secure_gate (input clk, input req, input [3:0] key, chal, sig, secret_key, input privacy, input [3:0] cls,
                    output unlocked, error, output [3:0] out);
  wire [1:0] st;
  auth_fsm  a (.clk(clk), .req(req), .key(key), .chal(chal), .sig(sig), .secret_key(secret_key), .state(st), .unlocked(unlocked), .error(error));
  lfsr_noise n (.clk(clk), .privacy(privacy), .cls(cls), .out(out));
endmodule
