# Examples Relating to Machine Learning

## Resnet18
* Example taken from [torch-mlir](https://github.com/llvm/torch-mlir/blob/main/examples/torchscript_resnet18.py).

* We need to use _all_ passes defined [here](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py#L115-L153).

* `verbose_resnet18_pass.mlir` - file that can be used for GPU execution (already went through all the passes). Compressed in zip due to its size.

* Limitation
  * `torch.sort` currently cannot get lowered to MLIR currently due to the error similar to that described [here](https://github.com/llvm/torch-mlir/issues/1151). Hence, the result is obtained by manually extracting out the prediction values.