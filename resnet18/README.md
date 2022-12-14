# Resnet18
This is an example motivated from [torch-mlir](https://github.com/llvm/torch-mlir/blob/main/examples/torchscript_resnet18.py). In the process of investigating for optimization opportunity, I had to look at each layer specifically. Hence, the manual layout of Resnet18 can be found in `tester_resnet18.py`, with each layer outlined in `layer_specific/` directory.
* Generated from the motivational example itself, `verbose_resnet18_pass.zip` contains an MLIR file that can be used for GPU execution directly (passes already applied). It is compressed in zip due to its size.

As briefly mentioned in the main README, the order of passes required to lower `tensor` dialect to `memref` dialect is included [here](https://github.com/llvm/torch-mlir/blob/f83a90585682c25367565fe8d612dd600e27ee04/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py#L115-L153). Before running on the server, we apply all the passes up to `tm-tensor-to-loops`, excluding `refback*`.
  * _NOTE_: The linked passes are not the most up-to-date version of TorchMLIR. Most up-to-date passes can be found [here](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py#L117-L166)

## Limitations
  1. In `verbose_resnet18_pass.zip`, `torch.sort` currently cannot get lowered to MLIR currently due to the error similar to that described [here](https://github.com/llvm/torch-mlir/issues/1151). Hence, the result is obtained by manually extracting out the prediction values.
  2. Adding `padding` argument in Conv2d cannot be translated
      * _Solution_ - Padding performed in a separate function `_pad` as seen from `layer_specific/conv2d.py`.
  3. BatchNorm2d cannot be translated
      * _Solution_ - Manual implementation of batchnorm in `layer_specific/batchnorm2d.py` 