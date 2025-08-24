`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 07:40:36 PM
// Design Name: 
// Module Name: mlp_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module mlp_tb;
    reg clk, valid, rst;
    localparam signed [D_WIDTH-1:0] inputs [L1_NUM_INPUTS] = '{
        0,0,0,0,0,0,1,25,32,29,19,1,1,11,22,44,45,1,0,0,3,46,14,0,0,0,31,39,1,0,0,3,53,14,0,0
    };
    
    wire signed [31:0] out [9:0];
    wire done;
    
    mlp dut (
        .clk(clk),
        .valid(valid),
        .rst(rst),
        .inputs(inputs),
        .out(out),
        .done(done)
    );
    
    initial begin
        clk = 0;
        forever begin
            #10 clk = ~clk;
        end
    end
    
    initial begin
        rst <= 1;
        valid <= 0;
        @(posedge clk);
        rst <= 0;
        valid <= 1;
        @(posedge done);
        $write("arr = ");
        for (int i = 0; i < 10 ; i++) begin
            $write("%d ", out[i]);
        end
        $display(""); // newline
        $finish;
    end

endmodule