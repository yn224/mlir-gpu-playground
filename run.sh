#!/bin/sh
if [ "$#" -ne 2 ]; then
    echo "Usage: ./jit.sh <filename> <compile-or-run>" >&2
    echo "       <compile-or-run>" >&2
    echo "       * 0 if running file through partial pass chain" >&2
    echo "       * 1 if running file through full pass chain" >&2
    echo "       * 2 if measuring time" >&2
    echo "       * 3 if running the generated assembly" >&2
    exit 1
fi

fname=$1
corr=$2

export LLVM_INSTALL_DIR=/work/shared/common/llvm-project-15.0.0-gpu
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=$LLVM_INSTALL_DIR/build/bin/:$PATH

if [ "$corr" -eq 0 ]; then
    mlir-opt $1 \
          --convert-linalg-to-parallel-loops \
          --convert-parallel-loops-to-gpu \
          --gpu-map-parallel-loops \
          --gpu-kernel-outlining \
          --lower-affine \
          --convert-scf-to-cf \
          --canonicalize \
          --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" \
          --reconcile-unrealized-casts \
          --gpu-to-llvm \
          --mlir-print-ir-after-failure > exout.llvm

    mlir-translate exout.llvm --mlir-to-llvmir > exout.ll
    opt exout.ll -O3 -S | llc -O3 -o exout.s

elif [ "$corr" -eq 1 ]; then
    mlir-opt $1 \
          --convert-linalg-to-parallel-loops \
          --convert-parallel-loops-to-gpu \
          --gpu-map-parallel-loops \
          --gpu-kernel-outlining \
          --canonicalize \
          --lower-affine \
          --convert-scf-to-cf \
          --arith-expand \
          --convert-math-to-llvm \
          --convert-math-to-libm \
          --convert-linalg-to-llvm \
          --convert-memref-to-llvm \
          --convert-arith-to-llvm \
          --convert-func-to-llvm \
          --convert-cf-to-llvm \
          --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" \
          --reconcile-unrealized-casts \
          --gpu-to-llvm > exout.llvm

    mlir-translate exout.llvm --mlir-to-llvmir > exout.ll
    opt exout.ll -O3 -S | llc -O3 -o exout.s

elif [ "$corr" -eq 2 ]; then

    mlir-cpu-runner \
          --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_cuda_runtime.so \
          --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_runner_utils.so \
          --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_c_runner_utils.so \
          --entry-point-result=void exout.llvm

else
    as -o exout.o exout.s
    clang++ exout.o -no-pie -L$LLVM_INSTALL_DIR/build/lib -o exec -lcuda -lmlir_cuda_runtime -lmlir_runner_utils -lmlir_c_runner_utils
    ./exec
fi
