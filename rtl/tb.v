module tb; reg clk=0; always #5 clk=~clk;
  reg req=0; reg [3:0] key, chal, sig; wire [1:0] st; wire un, er;
  auth_fsm dut(.clk(clk),.req(req),.key(key),.chal(chal),.sig(sig),.secret_key(4'hA),.state(st),.unlocked(un),.error(er));
  reg [3:0] mn; integer fails=0, k, c, s;
  reg signed [7:0] a,b; reg v,cl; wire signed [31:0] acc; reg signed [31:0] exp; integer i;
  mac_pe m(.clk(clk),.a(a),.b(b),.valid(v),.clear(cl),.acc(acc));
  task try(input [3:0] kk, cc, ss, input expect_ok); begin
    key=kk; chal=cc; sig=ss; req=1; @(posedge clk); @(posedge clk); @(posedge clk); #1;
    if ((un!==expect_ok) || (er!==!expect_ok)) begin fails=fails+1; end
    req=0; @(posedge clk); @(posedge clk); #1; end endtask
  initial begin
    for (k=0;k<16;k=k+1) for (c=0;c<16;c=c+1) for (s=0;s<16;s=s+1)
      try(k[3:0], c[3:0], s[3:0], (k==10) && (s[3:0]==(k[3:0]^{c[0],c[3:1]})));
    $display("auth_fsm exhaustive 4096 payloads: %0d mismatches vs rule (valid iff key==A and sig==key^ror(chal))", fails);
    fails=0; v=0; cl=0; a=0; b=0; @(posedge clk); #1;
    for (i=0;i<9;i=i+1) begin a=i*7-30; b=60-i*11; v=1; cl=(i==0); @(posedge clk); #1; end
    v=0; repeat(3) @(posedge clk); #1;
    exp=0; for (i=0;i<9;i=i+1) exp = exp + (i*7-30)*(60-i*11);
    $display("mac_pe 9-MAC dot product: rtl=%0d expected=%0d -> %s", acc, exp, (acc==exp)?"MATCH":"MISMATCH");
    $finish; end
endmodule
