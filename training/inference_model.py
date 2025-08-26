
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

QMAX = 127

model = np.load('quantized_model_brevitas.npz')


print(model.files)

W1_q = model['W1_q']
b1_q = model['b1_q']
M_fc1 = model['M_fc1']
S_fc1 = model['S_fc1']
scale_fc1_in = model['scale_fc1_in']

W2_q = model['W2_q']
b2_q = model['b2_q']
M_fc2 = model['M_fc2']
S_fc2 = model['S_fc2']


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


def quantize_input_image(x_float_batch):
    x_flat = (x_float_batch.reshape(x_float_batch.shape[0], -1) / scale_fc1_in)
    x_q = np.round(x_flat).astype(np.int32)
    x_q = np.clip(x_q, -QMAX, QMAX).astype(np.int8)
    return x_q


def print_as_sv_decl(inpt):
    input_str = f"localparam signed [D_WIDTH-1:0] inputs [L1_NUM_INPUTS] = '{{\n"
    input_str += "    " + ','.join(str(x) for x in inpt) + '\n};\n'
    print(input_str)
    
def int_inference_batch(x_float_batch):
    x_q = quantize_input_image(x_float_batch)
    print_as_sv_decl(x_q[0])
    preds = []
    for i in range(x_q.shape[0]):
        a0 = x_q[i].astype(np.int32)
        acc1 = int8_matvec(W1_q, a0, b1_q)
        # print()
        # print(acc1)
        r1 = requantize_accumulator(acc1, M_fc1, S_fc1, out_dtype=np.int8)
        r1 = r1.astype(np.int32)
        # print()
        # print(r1)
        r1[r1 < 0] = 0
        # print()
        # print(r1)
        acc2 = int8_matvec(W2_q, r1.astype(np.int8), b2_q)
        print()
        print(acc2)
        acc2_shared = requantize_accumulator(acc2, M_fc2, S_fc2, out_dtype=np.int32)
        print(acc2_shared)
        pred = int(np.argmax(acc2_shared))
        preds.append(pred)
    return np.array(preds, dtype=np.int32)

# Evaluate integer inference on a test subset
transform = transforms.Compose([
    transforms.Resize((6,6)),
    transforms.ToTensor(),
])
test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)

test_loader_small = DataLoader(test_ds, batch_size=1, shuffle=False)
all_preds = []
all_labels = []
seen = 0
for x, y in test_loader_small:
    x_np = x.numpy()
    preds = int_inference_batch(x_np)
    all_preds.append(preds)
    all_labels.append(y.numpy())
    seen += x_np.shape[0]
    if seen >= 10:
        break
    break
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
acc_int = (all_preds == all_labels).mean()
print(f'Integer-only inference accuracy on: {acc_int:.4f}')
