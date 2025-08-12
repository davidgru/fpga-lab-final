import numpy as np

model = np.load('quantized_model_brevitas.npz')


print(model.files)

W1 = model['W1_q']
b1 = model['b1_q']
M1 = model['M_fc1']
S1 = model['S_fc1']


W2 = model['W2_q']
b2 = model['b2_q']
M2 = model['M_fc2']
S2 = model['S_fc2']

def make_hyperparams_for_layer(f, weights, spec):
    f.write(f"localparam int {spec}_NUM_INPUTS = {weights.shape[1]};\n")
    f.write(f"localparam int {spec}_NUM_OUTPUTS = {weights.shape[0]};\n")

def write_params_for_layer(f, weights, bias, m, s, spec):
    weight_str = f"localparam signed [D_WIDTH-1:0] {spec}_W [{spec}_NUM_OUTPUTS] [{spec}_NUM_INPUTS] = '{{\n"
    weight_strs = []
    for row in weights:
        weight_strs.append('    {' + ','.join(str(x) for x in row) + '}')
    weight_str += ",\n".join(weight_strs) + "\n};\n"
    f.write(weight_str)

    bias_str = f"localparam signed [O_WIDTH-1:0] {spec}_b [{spec}_NUM_OUTPUTS] = '{{\n"
    bias_str += "    " + ','.join(str(x) for x in bias) + '\n};\n'
    f.write(bias_str)

    m_str = f"localparam signed [31:0] {spec}_m [{spec}_NUM_OUTPUTS] = '{{\n"
    m_str += "    " + ','.join(str(x) for x in m) + '\n};\n'
    f.write(m_str)

    s_str = f"localparam signed [31:0] {spec}_s [{spec}_NUM_OUTPUTS] = '{{\n"
    s_str += "    " + ','.join(str(x) for x in s) + '\n};\n'
    f.write(s_str)



with open('parameters.sv', 'w') as f:
    f.write("package weights_pkg;\n\n")
    f.write(f"localparam int D_WIDTH = 8;\n")
    f.write(f"localparam int O_WIDTH = 32;\n")

    make_hyperparams_for_layer(f, W1, "L1")
    make_hyperparams_for_layer(f, W2, "L2")

    f.write("\n")

    write_params_for_layer(f, W1, b1, M1, S1, 'L1')
    write_params_for_layer(f, W2, b2, M2, S2, 'L2')

    f.write("\n")
    f.write("endpackage;\n")

print(np.sum(W1[0].astype(np.int32) * W1[0].astype(np.int32)) + b1[0].astype(np.int32))


print(S1.dtype)