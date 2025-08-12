`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 07:06:38 PM
// Design Name: 
// Module Name: requantize_layer
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


module requantize_layer #(
    parameter int SIZE = 8,
    parameter int DATA_WIDTH = 8,
    parameter int OUTPUT_WIDTH = 32
) (
    input clk, rst, valid,
    input signed [OUTPUT_WIDTH-1:0] inputs  [SIZE-1:0],
    input signed [OUTPUT_WIDTH-1:0] m [SIZE-1:0],
    input signed [OUTPUT_WIDTH-1:0] s [SIZE-1:0],
    output reg signed [DATA_WIDTH-1:0] out [SIZE-1:0],
    output reg done
);

    reg signed [2*OUTPUT_WIDTH-1:0] mult_stage [SIZE-1:0];
    reg mult_done;

    // apply mult
    always @(posedge clk) begin
        if (rst == 1) begin
            mult_done <= 0;
        end else begin
            for (int i = 0; i < SIZE; i++) begin
                mult_stage[i] <= $signed(inputs[i]) * $signed(m[i]);
            end
            mult_done <= valid;
        end
    end
    
    // apply shift
    always @(posedge clk) begin
        if (rst == 1) begin
            done <= 0;
        end else begin
            for (int i = 0; i < SIZE; i++) begin
                out[i] <= $signed(mult_done[i]) >> s[i];
            end
            done <= mult_done;
        end
    end

endmodule
