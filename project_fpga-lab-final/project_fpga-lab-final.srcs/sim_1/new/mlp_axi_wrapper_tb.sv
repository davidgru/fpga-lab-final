`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/18/2025 05:13:40 PM
// Design Name: 
// Module Name: mlp_axi_wrapper_tb
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


module mlp_axi_wrapper_tb;

    localparam signed [D_WIDTH-1:0] inputs [36] = '{
        0,0,0,0,0,0,1,25,32,29,19,1,1,11,22,44,45,1,0,0,3,46,14,0,0,0,31,39,1,0,0,3,53,14,0,0
    };
    
    // Clock and reset
    reg clk;
    reg rstn;

    // AXI-Stream input signals
    reg [31:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    reg s_axis_tlast;

    // AXI-Stream output signals
    wire signed [31:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready;
    reg m_axis_tlast;  // not used as input

    reg signed [31:0] out [9:0];
    reg done;
    
    // Instantiate the DUT
    mlp_axi_wrapper dut (
        .clk(clk),
        .rstn(rstn),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );
    
    initial begin
        clk = 0;
        forever begin
            #10 clk = ~clk;
        end
    end
    
    initial begin
        rstn <= 0;
        s_axis_tvalid <= 0;
        s_axis_tdata  <= 0;
        s_axis_tlast  <= 0;
        m_axis_tready <= 0;
        @(posedge clk);
        rstn <= 1;
        
        // Send input over AXI
        for (integer i = 0; i < 36; i = i + 4) begin
            wait (s_axis_tready);
            s_axis_tdata <= (inputs[i+3] << 24) | (inputs[i+2] << 16) | (inputs[i+1] << 8) | inputs[i];
            s_axis_tvalid <= 1;
            s_axis_tlast <= (i+4 == 36) ? 1 : 0;
            @(posedge clk);
            s_axis_tvalid <= 0;
        end
        s_axis_tvalid <= 0;
        s_axis_tlast <= 0;
        
        // wait for done
        
        for (integer i = 0; i < 10;) begin
            m_axis_tready <= 1;
            @(posedge clk);
            if (m_axis_tvalid) begin
                m_axis_tready <= 0;
                $display("Output %0d = %0d", i, m_axis_tdata);
                i = i + 1;
            end
        end
        
        // Send input over AXI
        for (integer i = 0; i < 36; i = i + 4) begin
            wait (s_axis_tready);
            s_axis_tdata <= (inputs[i+3] << 24) | (inputs[i+2] << 16) | (inputs[i+1] << 8) | inputs[i];
            s_axis_tvalid <= 1;
            s_axis_tlast <= (i+4 == 36) ? 1 : 0;
            @(posedge clk);
            s_axis_tvalid <= 0;
        end
        s_axis_tvalid <= 0;
        s_axis_tlast <= 0;
        
        for (integer i = 0; i < 10;) begin
            m_axis_tready <= 1;
            @(posedge clk);
            if (m_axis_tvalid) begin
                m_axis_tready <= 0;
                $display("Output %0d = %0d", i, m_axis_tdata);
                i = i + 1;
            end
        end
        @(posedge clk);
        
        
        $finish;
    end

endmodule
