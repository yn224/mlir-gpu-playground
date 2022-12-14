import torch

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class BN1(torch.nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        self.eps = torch.full(shape, 0.00001)

    # Reference: https://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=batchnorm2d
    def _batch_norm(self, X, gamma, beta, eps):
        # Use `is_grad_enabled` to determine whether we are in training mode
        # if not torch.is_grad_enabled():
        #     # In prediction mode, use mean and variance obtained by moving average
        #     X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        # else:
        # assert len(X.shape) in (2, 4)

        # if len(X.shape) == 2:
        #     # When using a fully connected layer, calculate the mean and
        #     # variance on the feature dimension
        #     mean = X.mean(dim=0)
        #     var = ((X - mean) ** 2).mean(dim=0)
        # else:
        # When using a two-dimensional convolutional layer, calculate the
        # mean and variance on the channel dimension (axis=1). Here we
        # need to maintain the shape of `X`, so that the broadcasting
        # operation can be carried out later
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        # In training mode, the current mean and variance are used
        
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # X_hat = (X - mean) / torch.sqrt(var)

        # Update the mean and variance using moving average
        # moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        # moving_var = (1.0 - momentum) * moving_var + momentum * var
        Y = gamma * X_hat + beta  # Scale and shift
        return Y # moving_mean.data, moving_var.data

    def forward(self, x):
        # hw = (self.hw + 2 * self.pad - (self.ks - 1) - 1) // self.std + 1
        # hw = self.hw
        # zeros = torch.zeros(1, self.feat, hw, hw)
        # ones = torch.ones(1, self.feat, hw, hw)
        # eps = torch.full((1, self.feat, 1, 1), 0.00001)
        # y = self._batch_norm(x, ones, zeros, eps)
        # return y
        y = self._batch_norm(x, self.gamma, self.beta, self.eps)
        return y

model = BN1(512, 4)
module = torch_mlir.compile(model, torch.ones(1, 512, 1, 1), output_type="linalg-on-tensors", verbose=True)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)