import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class Conv1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(512, 512, kernel_size=3, stride=2)

    def _pad(self, x, amount: int):
        n, c, h, w = x.shape
        lr_pad = torch.zeros(n, c, h, amount)
        td_pad = torch.zeros(n, c, amount, w+(amount*2))
        padded = torch.cat([lr_pad, x, lr_pad], 3)
        padded = torch.cat([td_pad, padded, td_pad], 2)
        return padded

    def forward(self, x):
        x = self._pad(x, 1)
        return self.conv2d(x)

model = Conv1()
module = torch_mlir.compile(model, torch.ones(1, 512, 1, 1), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)