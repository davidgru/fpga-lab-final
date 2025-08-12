`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 06:10:49 PM
// Design Name: 
// Module Name: mlp
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

module mlp #(
    parameter int DATA_WIDTH = 8,
    parameter int OUTPUT_WIDTH = 32,
    parameter int INPUT_SIZE = 784,
    parameter int HIDDEN_SIZE = 300,
    parameter int OUTPUT_SIZE = 10
) (
    input clk, rst, valid,
    input signed [DATA_WIDTH-1:0] inputs  [INPUT_SIZE-1:0],
    input signed [DATA_WIDTH-1:0] l1_weights [INPUT_SIZE-1:0] [HIDDEN_SIZE-1:0],
    input signed [OUTPUT_WIDTH-1:0] l1_bias [HIDDEN_SIZE-1:0],
    input signed [31:0] l1_m [HIDDEN_SIZE-1:0],
    input signed [31:0] l1_s [HIDDEN_SIZE-1:0],
    input signed [DATA_WIDTH-1:0] l2_weights [HIDDEN_SIZE-1:0] [OUTPUT_SIZE-1:0],
    input signed [OUTPUT_WIDTH-1:0] l2_bias [OUTPUT_SIZE-1:0],
    input signed [31:0] l2_m [OUTPUT_SIZE-1:0],
    input signed [31:0] l2_s [OUTPUT_SIZE-1:0],
    output reg signed [DATA_WIDTH-1:0] out [OUTPUT_SIZE-1:0],
    output reg done
);

    wire signed [OUTPUT_WIDTH-1:0] l1_out [HIDDEN_SIZE-1:0];
    wire l1_done;
    
    wire signed [DATA_WIDTH-1:0] l1_requant_out [HIDDEN_SIZE-1:0];
    wire l1_requant_done;
    
    wire signed [DATA_WIDTH-1:0] l1_act_out [HIDDEN_SIZE-1:0];
    wire l1_act_done;
    
    wire signed [OUTPUT_WIDTH-1:0] l2_out [OUTPUT_SIZE-1:0];
    wire l2_done;
    
    wire signed [DATA_WIDTH-1:0] l2_requant_out [OUTPUT_SIZE-1:0];
    wire l2_requant_done;
    
    
    linear_layer #(
        .NUM_INPUTS(INPUT_SIZE),
        .NUM_OUTPUTS(HIDDEN_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) l1 (
        .clk(clk),
        .rst(rst),
        .valid(valid),
        .inputs(inputs),
        .weights(l1_weights),
        .bias(l1_bias),
        .out(l1_out),
        .done(l1_done)
    );
    
    requantize_layer #(
        .SIZE(HIDDEN_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) l1_requant (
        .clk(clk),
        .rst(rst),
        .valid(l1_done),
        .inputs(l1_out),
        .m(l1_m),
        .s(l1_s),
        .out(l1_requant_out),
        .done(l1_requant_done),
    );
    
    relu #(
        .SIZE(HIDDEN_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) act (
        .clk(clk),
        .rst(rst),
        .valid(l1_requant_done),
        .inputs(l1_requant_out),
        .outputs(l1_act_out),
        .done(l1_act_done),
    );
    
    linear_layer #(
        .NUM_INPUTS(HIDDEN_SIZE),
        .NUM_OUTPUTS(OUTPUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) l1 (
        .clk(clk),
        .rst(rst),
        .valid(l1_act_done),
        .inputs(l1_act_out),
        .weights(l2_weights),
        .bias(l2_bias),
        .out(l2_out),
        .done(l2_done)
    );
    
    requantize_layer #(
        .SIZE(OUTPUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) l1_requant (
        .clk(clk),
        .rst(rst),
        .valid(l2_done),
        .inputs(l2_out),
        .m(l2_m),
        .s(l2_s),
        .out(l2_requant_out),
        .done(l2_requant_done),
    );
    
endmodule
