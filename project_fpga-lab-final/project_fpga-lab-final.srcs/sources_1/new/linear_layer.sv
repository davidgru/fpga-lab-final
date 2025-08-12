`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 05:14:41 PM
// Design Name: 
// Module Name: linear_layer
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


module linear_layer #(
    parameter int NUM_INPUTS = 8,
    parameter int NUM_OUTPUTS = 8,
    parameter int DATA_WIDTH = 8,
    parameter int OUTPUT_WIDTH = 32
) (
    input clk, rst, valid,
    input signed [DATA_WIDTH-1:0] inputs  [NUM_INPUTS-1:0],
    input signed [DATA_WIDTH-1:0] weights [NUM_OUTPUTS-1:0] [NUM_INPUTS-1:0],
    input signed [OUTPUT_WIDTH-1:0] bias [NUM_OUTPUTS-1:0],
    output reg signed [OUTPUT_WIDTH-1:0] out [NUM_OUTPUTS-1:0],
    output reg done
);

    wire [NUM_OUTPUTS-1:0] dones;
    assign done = &dones;

    generate
        for (genvar i = 0; i < NUM_OUTPUTS; i = i + 1) begin : GEN_PERCEPTRONS
            perceptron #(
                .NUM_INPUTS(NUM_INPUTS),
                .DATA_WIDTH(DATA_WIDTH),
                .OUTPUT_WIDTH(OUTPUT_WIDTH)
            ) p (
                .clk(clk),
                .rst(rst),
                .valid(valid),
                .inputs(inputs),
                .weights(weights[i]),
                .bias(bias[i]),
                .out(out[i]),
                .done(dones[i])
            );
        end
    endgenerate
endmodule
