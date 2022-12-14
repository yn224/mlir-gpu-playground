import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class AvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.avgpool(x)

model = AvgPool()
module = torch_mlir.compile(model, torch.ones(1, 512, 1, 1), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)
