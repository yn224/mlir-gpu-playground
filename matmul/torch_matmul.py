import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs):
        return torch.matmul(lhs, torch.ones(2, 2))

model = Matmul()
module = torch_mlir.compile(model, torch.ones(2, 2), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
# print(compiled)
jit_module = backend.load(compiled)
print("++++++++++++++++++++++++++++++++++++++")
input_tensor = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
print("Model's output:")
print(model.forward(input_tensor).numpy())
print("Through TorchMLIR:")
print(jit_module.forward(input_tensor.numpy()))
print("++++++++++++++++++++++++++++++++++++++")