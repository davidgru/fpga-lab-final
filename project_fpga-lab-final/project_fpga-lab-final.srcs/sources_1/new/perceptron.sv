`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 01:44:03 PM
// Design Name: 
// Module Name: perceptron
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


module perceptron#(
    parameter int DATA_WIDTH = 8,
    parameter int NUM_INPUTS = 8,
    parameter int SUM_WIDTH = 32
) (
    input clk, rst, valid,
    input signed [DATA_WIDTH-1:0] inputs  [NUM_INPUTS-1:0],
    input signed [DATA_WIDTH-1:0] weights [NUM_INPUTS-1:0],
    input signed [DATA_WIDTH-1:0] bias,
    output reg signed [SUM_WIDTH-1:0] out,
    output reg done
);
    localparam LEVELS = $clog2(NUM_INPUTS);
    localparam P2_NUM_INPUTS = 1<<LEVELS;

    reg signed [SUM_WIDTH-1:0] stage_reg [0:LEVELS] [P2_NUM_INPUTS-1:0];
    reg valid_stage [0:LEVELS];

    // mult stage
    always @(posedge clk) begin
        if (rst == 1) begin
            valid_stage[0] <= 0;
        end else begin
            for (int i = 0; i < P2_NUM_INPUTS; i++) begin : MULT_STAGE
                if (i < NUM_INPUTS) begin
                    stage_reg[0][i] <= $signed(inputs[i]) * $signed(weights[i]);
                end else begin
                    stage_reg[0][i] <= 0;
                end
            end
            valid_stage[0] <= valid;
        end
    end
    
    // generate pipeline stages for the adder tree
    generate
        for (genvar level = 1; level <= LEVELS; level = level + 1) begin: GEN_ADDER_TREE
            int num_adders = P2_NUM_INPUTS / (1 << level);
            always @(posedge clk) begin
                if (rst == 1) begin
                    valid_stage[level] <= 0;
                end else begin
                    for (int n = 0; n < num_adders; n = n + 1) begin
                        stage_reg[level][n] <= $signed(stage_reg[level-1][2*n]) + $signed(stage_reg[level-1][2*n+1]);
                    end
                    valid_stage[level] <= valid_stage[level-1];
                end
            end
        end
    endgenerate
    
    // add bias
    always @(posedge clk) begin
        if (rst == 1) begin
            done <= 0;
        end else begin
            out <= $signed(stage_reg[LEVELS][0]) + $signed(bias);
            done <= valid_stage[LEVELS];
        end
    end

endmodule
