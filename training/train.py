"""
mnist_mlp_int8_qat_brevitas.py

This script is a drop-in replacement of the previous QAT implementation, but now using **Brevitas** for Quantization-Aware Training (QAT).

What it does
- Builds a 2-layer MLP (784 -> 300 -> 10) where:
  - We use `brevitas.nn.QuantLinear` for weight quantization (per-output-channel symmetric int8).
  - We use `brevitas.nn.QuantAct` to fake-quantize activations per-layer (symmetric int8, per-tensor).
- Trains with QAT using Brevitas fake-quant ops (STE based) so the model learns to be robust to 8-bit quantization.
- After training, the script **exports integer-only parameters** required for FPGA inference into `quantized_model_brevitas.npz`:
  - int8 quantized weights (per-output-channel), int32 biases in accumulator domain
  - activation scales (per-layer), weight scales (per-output-channel)
  - fixed-point multipliers and shifts for requantization
- Provides a numpy-only integer-only inference routine to validate exported params (no FP ops at inference time).

Requirements
- Python packages: torch, torchvision, brevitas, numpy
  Install brevitas: `pip install brevitas` (if you don't already have it)

Usage
    python mnist_mlp_int8_qat_brevitas.py

Notes / Choices
- We use symmetric quantization to signed 8-bit with QMAX=127 (leave -128 unused).
- We use per-channel (per-output) weight quantization by extracting float weights post-training and computing per-row scales for export. Brevitas' QuantLinear is used during training for accurate fake quant behavior.
- Activation quantization uses per-layer QuantAct (per-tensor). We calibrate activation ranges during training by reading the QuantAct observed stats (Brevitas keeps tracking or we compute ranges from data). For robustness we also recompute scales from data before export.

"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Brevitas imports
import brevitas
from brevitas.nn import QuantLinear, QuantIdentity, QuantReLU, QuantHardTanh, QuantConv2d
from brevitas.core.quant import QuantType

QMAX = 127

# ----------------- Model using Brevitas quant modules -----------------
class BrevitasMLP(nn.Module):
    def __init__(self, hidden=150, weight_bit_width=8, act_bit_width=8):
        super().__init__()
        self.act1 = QuantIdentity(bit_width=act_bit_width, quant_type=QuantType.INT, return_quant_tensor=False)
        self.fc1 = QuantLinear(6*6, hidden,
                               bias=True,
                               weight_bit_width=weight_bit_width,
                               weight_quant_type=QuantType.INT)
        self.act2 = QuantReLU(bit_width=act_bit_width, quant_type=QuantType.INT, return_quant_tensor=False)
        self.fc2 = QuantLinear(hidden, 10,
                               bias=True,
                               weight_bit_width=weight_bit_width,
                               weight_quant_type=QuantType.INT)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)
        return x

# ----------------- Helpers for export / integer inference -----------------

def compute_channelwise_weight_scale(weight: np.ndarray):
    max_per_channel = np.max(np.abs(weight), axis=1)
    max_per_channel[max_per_channel == 0.0] = 1e-8
    scales = max_per_channel / QMAX
    return scales


def make_fixed_point_multiplier(scale_ratio):
    if scale_ratio == 0:
        return 0, 0
    shift = 31
    M = int(round(scale_ratio * (1 << shift)))
    while M >= (1 << 31) and shift > 0:
        M >>= 1
        shift -= 1
    return M, shift

def int8_matvec(W_q: np.ndarray, a_q: np.ndarray, bias_q: np.ndarray):
    acc = W_q.astype(np.int32).dot(a_q.astype(np.int32))
    if bias_q is not None:
        acc = acc + bias_q.astype(np.int32)
    return acc

def requantize_accumulator(acc: np.ndarray, M: np.ndarray, S: np.ndarray, out_dtype=np.int8):
    acc64 = acc.astype(np.int64) * M.astype(np.int64)
    res = (acc64 >> S.astype(np.int64)).astype(np.int32)
    if out_dtype == np.int8:
        res = np.clip(res, -QMAX, QMAX).astype(np.int8)
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((6,6)),
        transforms.ToTensor()
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    calib_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = BrevitasMLP(hidden=20, weight_bit_width=8, act_bit_width=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    epochs = 8
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss={running_loss/len(train_loader):.4f} acc={correct/total:.4f}")

    model.eval()
    # Extract FP weights and biases
    W1_fp = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # (300, 784)
    b1_fp = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    W2_fp = model.fc2.weight.detach().cpu().numpy().astype(np.float32)  # (10, 300)
    b2_fp = model.fc2.bias.detach().cpu().numpy().astype(np.float32)

    # Compute per-output-channel weight scales and quantize weights to int8
    W1_scales = compute_channelwise_weight_scale(W1_fp)
    W2_scales = compute_channelwise_weight_scale(W2_fp)
    W1_q = np.round(W1_fp / W1_scales[:, None]).astype(np.int8)
    W2_q = np.round(W2_fp / W2_scales[:, None]).astype(np.int8)

    # Calibrate weight and activation scales
    act_max_fc1 = 0.0
    act_max_fc2 = 0.0
    nb_calib_batches = 100
    with torch.no_grad():
        for i, (x, y) in enumerate(calib_loader):
            if i >= nb_calib_batches:
                break
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            act_max_fc1 = max(act_max_fc1, float(x_flat.abs().max().cpu().item()))
            # forward to fc2 input
            # emulate forward but using FP weights to see ranges (dequantized path)
            h = torch.matmul(model.act1(x_flat), model.fc1.weight.t()) + model.fc1.bias
            h = torch.relu(h)
            act_max_fc2 = max(act_max_fc2, float(h.abs().max().cpu().item()))

    scale_fc1_in = max(act_max_fc1, 1e-8) / QMAX
    scale_fc2_in = max(act_max_fc2, 1e-8) / QMAX
    print('Calibrated activation scales: scale_fc1_in=', scale_fc1_in, ' scale_fc2_in=', scale_fc2_in)

    # Convert biases to int32 in accumulator domain: bias_fp / (w_scale * act_scale)
    b1_q = np.round(b1_fp / (W1_scales * scale_fc1_in)).astype(np.int32)
    b2_q = np.round(b2_fp / (W2_scales * scale_fc2_in)).astype(np.int32)

    # Requant multipliers: map acc1 (scale = W1_scales * scale_fc1_in) -> fc2 input quantized domain (scale_fc2_in)
    scale_ratios_fc1 = (W1_scales * scale_fc1_in) / scale_fc2_in
    M_fc1 = np.zeros_like(scale_ratios_fc1, dtype=np.int64)
    S_fc1 = np.zeros_like(scale_ratios_fc1, dtype=np.int32)
    for j in range(scale_ratios_fc1.shape[0]):
        M, S = make_fixed_point_multiplier(scale_ratios_fc1[j])
        M_fc1[j] = M
        S_fc1[j] = S

    # For final outputs, map to shared output domain so argmax is valid
    out_scales = W2_scales * scale_fc2_in
    shared_out_scale = float(np.max(out_scales))
    scale_ratios_fc2 = out_scales / shared_out_scale
    M_fc2 = np.zeros_like(scale_ratios_fc2, dtype=np.int64)
    S_fc2 = np.zeros_like(scale_ratios_fc2, dtype=np.int32)
    for j in range(scale_ratios_fc2.shape[0]):
        M, S = make_fixed_point_multiplier(scale_ratios_fc2[j])
        M_fc2[j] = M
        S_fc2[j] = S

    # Save everything needed for integer-only inference
    np.savez('quantized_model_brevitas.npz',
             W1_q=W1_q, b1_q=b1_q, W1_scales=W1_scales.astype(np.float32),
             W2_q=W2_q, b2_q=b2_q, W2_scales=W2_scales.astype(np.float32),
             scale_fc1_in=np.float32(scale_fc1_in), scale_fc2_in=np.float32(scale_fc2_in),
             shared_out_scale=np.float32(shared_out_scale),
             M_fc1=M_fc1, S_fc1=S_fc1,
             M_fc2=M_fc2, S_fc2=S_fc2)
    print('Saved quantized_model_brevitas.npz')

    # Test inference
    data = np.load('quantized_model_brevitas.npz')
    W1_q = data['W1_q']
    b1_q = data['b1_q']
    W2_q = data['W2_q']
    b2_q = data['b2_q']
    M_fc1 = data['M_fc1']
    S_fc1 = data['S_fc1']
    M_fc2 = data['M_fc2']
    S_fc2 = data['S_fc2']
    scale_fc1_in = float(data['scale_fc1_in'])
    scale_fc2_in = float(data['scale_fc2_in'])
    shared_out_scale = float(data['shared_out_scale'])

    def quantize_input_image(x_float_batch):
        x_flat = (x_float_batch.reshape(x_float_batch.shape[0], -1) / scale_fc1_in)
        x_q = np.round(x_flat).astype(np.int32)
        x_q = np.clip(x_q, -QMAX, QMAX).astype(np.int8)
        return x_q

    def int_inference_batch(x_float_batch):
        x_q = quantize_input_image(x_float_batch)
        preds = []
        for i in range(x_q.shape[0]):
            a0 = x_q[i].astype(np.int32)
            acc1 = int8_matvec(W1_q, a0, b1_q)
            r1 = requantize_accumulator(acc1, M_fc1, S_fc1, out_dtype=np.int8)
            r1 = r1.astype(np.int32)
            r1[r1 < 0] = 0
            acc2 = int8_matvec(W2_q, r1.astype(np.int8), b2_q)
            acc2_shared = requantize_accumulator(acc2, M_fc2, S_fc2, out_dtype=np.int32)
            pred = int(np.argmax(acc2_shared))
            preds.append(pred)
        return np.array(preds, dtype=np.int32)

    # Evaluate integer inference on a test subset
    test_samples = 2000
    test_loader_small = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_preds = []
    all_labels = []
    seen = 0
    for x, y in test_loader_small:
        x_np = x.numpy()
        preds = int_inference_batch(x_np)
        all_preds.append(preds)
        all_labels.append(y.numpy())
        seen += x_np.shape[0]
        # if seen >= test_samples:
        #     break
    all_preds = np.concatenate(all_preds, axis=0)[:test_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:test_samples]
    acc_int = (all_preds == all_labels).mean()
    print(f'Integer-only inference accuracy on {test_samples} test samples: {acc_int:.4f}')

    # Compare to Brevitas QAT float-like model
    float_preds = []
    seen = 0
    with torch.no_grad():
        for x, y in test_loader_small:
            x = x.to(device)
            out = model(x)
            _, p = out.max(1)
            float_preds.append(p.cpu().numpy())
            seen += x.size(0)
            # if seen >= test_samples:
            #     break
    float_preds = np.concatenate(float_preds, axis=0)[:test_samples]
    acc_float = (float_preds == all_labels).mean()
    print(f'Float (Brevitas QAT) model accuracy on same subset: {acc_float:.4f}')

if __name__ == '__main__':
    main()
