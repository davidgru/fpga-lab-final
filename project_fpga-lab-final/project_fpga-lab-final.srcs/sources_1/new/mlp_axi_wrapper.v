`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/18/2025 04:35:04 PM
// Design Name: 
// Module Name: mlp_axi_wrapper
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

module mlp_axi_wrapper(
    input [31:0] s_axis_tdata,
    input s_axis_tvalid,
    output reg s_axis_tready,
    input s_axis_tlast,
    output reg [31:0] m_axis_tdata,
    output reg m_axis_tvalid,
    input m_axis_tready,
    output reg m_axis_tlast,
    input clk,
    input rstn
    );
    
    reg signed [36*8-1:0] mlp_in;
    reg mlp_in_valid;
    wire signed [10*32-1:0] mlp_out;
    wire mlp_out_valid;
    
    mlp_plain_v mlp_inst (
        .clk(clk),
        .valid(mlp_in_valid),
        .rst(~rstn),
        .inputs(mlp_in),
        .out(mlp_out),
        .done(mlp_out_valid)
    );
    
    reg [31:0] in_count;
    reg [31:0] out_count;
    reg return_done;
    
    always @(posedge clk) begin
        if (!rstn) begin
            in_count <= 0;
            mlp_in_valid <= 0;
            s_axis_tready <= 0;
        end else begin
            s_axis_tready <= (in_count != 36);
            if (return_done) begin
                in_count <= 0;
            end
            if (s_axis_tvalid) begin
                mlp_in[8*(in_count + 0) +: 8] <= s_axis_tdata[7:0];
                mlp_in[8*(in_count + 1) +: 8] <= s_axis_tdata[15:8];
                mlp_in[8*(in_count + 2) +: 8] <= s_axis_tdata[23:16];
                mlp_in[8*(in_count + 3) +: 8] <= s_axis_tdata[31:24];
                in_count <= in_count + 4;
            end
            mlp_in_valid <= (in_count == 36);
            
        end
    end
    
    always @(posedge clk) begin
        if (!rstn) begin
            out_count <= 0;
            m_axis_tvalid <= 0;
            m_axis_tlast <= 0;
            m_axis_tdata <= 0;
            return_done <= 0;
        end else begin
            m_axis_tvalid <= mlp_out_valid;
            return_done <= 0;
            if (mlp_out_valid) begin
                m_axis_tdata <= mlp_out[32*out_count +: 32];
                m_axis_tlast <= (out_count == 9);
                if (m_axis_tready && out_count < 9) begin
                    out_count <= out_count + 1;
                    m_axis_tlast <= 0;
                end else begin
                    m_axis_tlast <= 1;
                    out_count <= 0;
                    return_done <= 1;
                end
            end
        end
    end

endmodule
