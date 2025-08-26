`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/24/2025 11:21:26 PM
// Design Name: 
// Module Name: mlp_gpio_wrapper_tb
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


module mlp_gpio_wrapper_tb2();

    localparam signed [7:0] inputs [36] = '{
        0,0,0,0,0,0,1,25,32,29,19,1,1,11,22,44,45,1,0,0,3,46,14,0,0,0,31,39,1,0,0,3,53,14,0,0
    };
    
    // Clock and reset
    reg clk;
    reg rstn;
    
    wire [15:0] in_read_addr;
    reg [15:0] in_write_addr;
    reg [7:0] in_data;
    reg in_valid;
    wire [3:0] result;
    wire done;
    wire [3:0] magic;
    
    // Instantiate the DUT
    mlp_gpio_wrapper4 dut (
        .clk(clk),
        .rst(rstn),
        .in_read_addr(in_read_addr),
        .in_write_addr(in_write_addr),
        .in_data(in_data),
        .in_valid(in_valid),
        .result(result),
        .done(done),
        .magic(magic)
    );
    
    initial begin
        clk = 0;
        forever begin
            #10 clk = ~clk;
        end
    end
    
    initial begin
        rstn <= 0;
        
        in_write_addr <= 0;
        in_data <= 0;
        in_valid <= 0;
        
        @(posedge clk);
        rstn <= 1;
        @(posedge clk);
        @(posedge clk);
        
        // Send input over GPIO
        in_valid <= 1;
        for (integer i = 0; i < 36; i = i + 1) begin
            in_write_addr <= i;
            in_data <= inputs[i];
            @(posedge clk);
            wait(in_read_addr == in_write_addr + 1);
            @(posedge clk);
        end
        in_valid <= 0;
        
        // wait for done
        wait(done);
        $display("Prediction: %0u", result);
        
        in_write_addr <= 0;
        @(posedge clk);
        wait(in_read_addr == 0);
        // Send input over GPIO
        in_valid <= 1;
        for (integer i = 0; i < 36; i = i + 1) begin
            in_write_addr <= i;
            in_data <= inputs[i];
            @(posedge clk);
            wait(in_read_addr == in_write_addr + 1);
            @(posedge clk);
        end
        in_valid <= 0;
        
        // wait for done
        wait(done);
        $display("Prediction: %0u", result);
        $finish;
    end

endmodule
