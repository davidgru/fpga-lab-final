`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 02:49:21 PM
// Design Name: 
// Module Name: perceptron_tb
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
import weights_pkg::*;

module perceptron_tb;
    parameter DATA_WIDTH = D_WIDTH;
    parameter NUM_INPUTS = L1_NUM_INPUTS ;
    parameter OUTPUT_WIDTH = O_WIDTH ;
    reg clk, valid, rst;
    reg signed [DATA_WIDTH-1:0] inputs [NUM_INPUTS-1:0];
    reg signed [DATA_WIDTH-1:0] weights [NUM_INPUTS-1:0];
    reg signed [OUTPUT_WIDTH-1:0] bias;
    
    wire signed [OUTPUT_WIDTH-1:0] out;
    wire done;
    
    perceptron #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_INPUTS(NUM_INPUTS),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid(valid),
        .inputs(inputs),
        .weights(weights),
        .bias(bias),
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
        inputs <= L1_W[0];
        weights <= L1_W[0];
        bias <= L1_b[0];
        valid <= 0;
        @(posedge clk);
        rst <= 0;
        valid <= 1;
        @(posedge done);
        $display("output = %d", out);
        $finish;
    end

endmodule
