// Authentication FSM: IDLE(0) VERIFY(1) UNLOCKED(2) ERROR(3); signature = key ^ ror4(challenge)
module auth_fsm (input clk, input req, input [3:0] key, chal, sig, input [3:0] secret_key,
                 output reg [1:0] state, output unlocked, error);
  wire [3:0] ror = {chal[0], chal[3:1]};
  wire ok = (key == secret_key) && (sig == (key ^ ror));
  always @(posedge clk) case (state)
    2'd0: if (req) state <= 2'd1;
    2'd1: state <= ok ? 2'd2 : 2'd3;
    default: if (!req) state <= 2'd0;
  endcase
  assign unlocked = (state == 2'd2);
  assign error    = (state == 2'd3);
  initial state = 0;
endmodule
