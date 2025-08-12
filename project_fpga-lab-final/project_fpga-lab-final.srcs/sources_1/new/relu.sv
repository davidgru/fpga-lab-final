`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/12/2025 07:28:05 PM
// Design Name: 
// Module Name: relu
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


module relu #(
    parameter int SIZE = 8,
    parameter int DATA_WIDTH = 8
) (
    input clk, rst, valid,
    input signed [DATA_WIDTH-1:0] inputs [SIZE-1:0],
    output reg [DATA_WIDTH-1:0] outputs [SIZE-1:0],
    output reg done
);
    
    always @(posedge clk) begin
        if (rst == 1) begin
            done <= 0;
        end else begin
            for (int i = 0; i < SIZE; i++) begin
                outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
            end
            done <= valid;
        end
    end

endmodule
