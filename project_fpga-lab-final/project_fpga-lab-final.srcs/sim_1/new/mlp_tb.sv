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
    localparam signed [8-1:0] inputs [36] = '{
        0,0,0,0,0,0,0,25,32,29,18,0,0,11,21,44,44,0,0,0,3,46,14,0,0,0,31,39,0,0,0,2,52,13,0,0
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