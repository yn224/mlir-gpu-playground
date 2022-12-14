import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class Maxpool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, 2)

    def _pad(self, x, amount: int):
        n, c, h, w = x.shape
        lr_pad = torch.zeros(n, c, h, amount)
        td_pad = torch.zeros(n, c, amount, w+(amount*2))
        padded = torch.cat([lr_pad, x, lr_pad], 3)
        padded = torch.cat([td_pad, padded, td_pad], 2)
        return padded

    def forward(self, x):
        x = self._pad(x, 1)
        return self.maxpool(x)

model = Maxpool()
module = torch_mlir.compile(model, torch.ones(1, 64, 112, 112), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)