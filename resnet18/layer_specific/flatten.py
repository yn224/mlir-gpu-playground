
import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

model = Flatten()
module = torch_mlir.compile(model, torch.ones(1, 512, 1, 1), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)