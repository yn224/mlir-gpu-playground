# MLIR
Learning & experimenting with LLVM/MLIR/TorchMLIR to GPU

## LLVM Version
Currently, LLVM15 is being used for implementing and testing.

## TorchMLIR
Tools & Versions:
* cmake 3.24.0
* python3.9
* gcc 9.4.0
* g++ 9.4.0

Build instruction with python binding (adopted from [here](https://github.com/llvm/torch-mlir/blob/main/build_tools/build_standalone.sh)):
```
cd $TORCH_MLIR_SDIR

mkdir build && cd build

cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_C_COMPILER=`which gcc` \
              -DCMAKE_CXX_COMPILER=`which g++` \
              -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \ 
              -DPython3_FIND_VIRTUALENV=ONLY \ 
              -DLLVM_ENABLE_PROJECTS=mlir \
              -DTORCH_MLIR_ENABLE_MHLO=OFF \
              -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
              -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=$TORCH_MLIR_SDIR \
              -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=$TORCH_MLIR_SDIR/externals/llvm-external-projects/torch-mlir-dialects \
              -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
              -DLLVM_TARGETS_TO_BUILD=host $LLVM_SDIR/llvm

ninja tools/torch-mlir/all check-torch-mlir-all
```

## Examples
* `sample`: Sample program (matmul) to demonstrate TorchMLIR compiler
* `resnet18`: Resnet18 example provided by PyTorch
* `centernet` (private): Centernet model
* `reactnet` (private): Reactnet model
* `skynet` (private): Skynet model
* `squeezenet` (private): Squeezenet model
* `ultranet` (private): Ultranet model
* `yolo` (private): YOLO model

## Execution time
The following are the execution times for different examples (matmul omitted due to its simplicity).
* Resnet18 - using image given already in [here](https://github.com/llvm/torch-mlir/blob/main/examples/torchscript_resnet18.py#L62)
  | Unit: (s) | Pytorch Flow | Torchscript Flow |
  |:----------|:------------:|:----------------:|
  | Inference | 0.0004436969757080078 | 9.140702724456787 |
  |   Test    | 0.04910922050476074   | 23.24636745452881 |

  | Unit: (s) | Full MLIR |
  |:----------|:---------:|
  |    CPU    |  24.309   |
  |    GPU    |  7.533    |

_NOTE_: GPU is being run in `brg-zhang-xcel` server @Conell

## Compile and Run on GPU
To generate GPU code, the script `run.sh` - adopted from [here](https://github.com/zzzDavid/mlir-playground/blob/main/gpu-backend/compile.sh) - would be used.

Given some `program.py` with `verbose=True` in `torch_mlir.compile`,
1. Generate and save the generated IR by running `python3 program.py &> out.mlir`
2. Remove `Torch Backend IR` and keep `Linalg Backend IR`.
3. Apply the following passes to lower `tensor` dialect to `memref` dialect:
```
./build/bin/torch-mlir-opt --scf-bufferize --tm-tensor-bufferize --linalg-init-tensor-to-alloc-tensor --linalg-bufferize --func-bufferize --arith-bufferize --tensor-bufferize --finalizing-bufferize --tm-tensor-to-loops out.mlir &> out_lowered.mlir 
```
4. Further lower and generate assembly by running `./run.sh out_lowered.mlir 1`
    * The numerical option of 2 makes it a CPU-runner.
5. Edit out `"exout.llvm"` contained in the line `.file 1 "$PWD" "exout.llvm"` of `exout.s` that gets generated.
6. Run `./run.sh out_lowered.mlir 3` to run the program on GPU.

## Debugging
General errors encountered are being logged in `error-tracking.md`.