`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/24/2025 10:01:56 PM
// Design Name: 
// Module Name: mlp_gpio_wrapper3
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


module argmax(
    input clk,
    input rst,
    input valid,
    input signed [10*32-1:0] inputs,
    output reg [3:0] max,
    output reg done
    );
    
    reg [3:0] idx;
    
    always @(posedge clk) begin
        if (!rst) begin
            done <= 0;
            idx <= 0;
            max <= 0;
        end else begin
            if (valid == 1) begin
                if (idx == 10) begin
                    done <= 1;
                    idx <= 0;
                end else begin
                    if ($signed(inputs[32*idx +: 32]) >= $signed(inputs[32*max +: 32])) begin
                        max <= idx;
                    end
                    idx <= idx + 1;
                    done <= 0;
                end
            end else begin
                done <= 0;
                idx <= 0;
                max <= 0;
            end
        end
    end
endmodule

module mlp_gpio_wrapper3(
    input clk,
    input rst,
    output reg [15:0] in_read_addr,
    input [15:0] in_write_addr,
    input [7:0] in_data,
    input in_valid,
    output [3:0] result,
    output done,
    output reg [3:0] magic
    );
    
    reg signed [36*8-1:0] mlp_in;
    reg mlp_in_valid;
    wire signed [10*32-1:0] mlp_out;
    wire mlp_out_valid;
    
    // for debug
    mlp_plain_v mlp_inst (
        .clk(clk),
        .valid(mlp_in_valid),
        .rst(~rstn),
        .inputs(mlp_in),
        .out(mlp_out),
        .done(mlp_out_valid)
    );
    
    argmax argmax_inst (
        .clk(clk),
        .rst(rstn),
        .valid(mlp_out_valid),
        .inputs(mlp_out),
        .max(result),
        .done(done)
    );
    
    // 0: wait for input
    // 1: wait for mlp done and argmax
    reg [1:0] state;
    
    always @(posedge clk) begin
        if (!rst) begin
            mlp_in <= 0;
            mlp_in_valid <= 0;
            state <= 0;
            in_read_addr <= 0;
        end else begin
            case (state)
                0: begin
                    if (in_valid && (in_write_addr == in_read_addr)) begin
                        mlp_in[8*(in_write_addr) +: 8] <= in_data;
                        in_read_addr <= in_read_addr + 1;
                    end
                    mlp_in_valid <= in_read_addr == 36;
                    if (mlp_in_valid) begin
                        state <= 1;
                    end
                end
                1: begin
                    if (done) begin
                        state <= 0;
                        in_read_addr <= 0;
                        mlp_in_valid <= 0;
                    end
                end
            endcase
        end
        magic <= 4'b1111;
    end
endmodule
