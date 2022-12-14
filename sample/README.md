# Sample TorchMLIR Compilation Process using Matmul
`runnable_linalg.mlir` - result of lowering `linalg_tensor_backend_matmul.mlir` using passes shown above _without_ `refback-munge-calling-conventions`.

## Stages
1. Running `python3 torch_matmul.py` yields `raw_matmul.mlir`, which gets compiled down to `torch_backend_matmul.mlir` and `linalg_tensor_backend_matmul.mlir`.
2. Apply [the pass chain](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py#L115-L153) on the file `linalg_tensor_backend_matmul.mlir`. Below are the IRs after each pass before converting to LLVM.
    <details>
      <summary>IR changes</summary>

      * `refback-generalize-tensor-pad` (No change)
      * `scf-bufferize` (No change)
      * `tm-tensor-bufferize` (No change)
      * `linalg-init-tensor-to-alloc-tensor` (No change)
      * `linalg-bufferize`: DESCRIPTION
        <details>
          <summary>code</summary>
          
          ```
          module attributes {torch.debug_module_name = "Matmul"} {
            func.func @forward(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = bufferization.to_memref %arg0 : memref<2x2xf32>
            %cst = arith.constant 1.000000e+00 : f32
            %cst_0 = arith.constant 0.000000e+00 : f32
            %c2 = arith.constant 2 : index
            %c2_1 = arith.constant 2 : index
            %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
            %2 = bufferization.to_tensor %1 : memref<2x2xf32>
            linalg.fill ins(%cst : f32) outs(%1 : memref<2x2xf32>)
            %3 = bufferization.to_tensor %1 : memref<2x2xf32>
            %c2_2 = arith.constant 2 : index
            %c2_3 = arith.constant 2 : index
            %4 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
            %5 = bufferization.to_tensor %4 : memref<2x2xf32>
            linalg.fill ins(%cst_0 : f32) outs(%4 : memref<2x2xf32>)
            %6 = bufferization.to_tensor %4 : memref<2x2xf32>
            %7 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
            memref.copy %4, %7 : memref<2x2xf32> to memref<2x2xf32>
            %8 = bufferization.to_tensor %7 : memref<2x2xf32>
            linalg.matmul ins(%0, %1 : memref<2x2xf32>, memref<2x2xf32>) outs(%7 : memref<2x2xf32>)
            %9 = bufferization.to_tensor %7 : memref<2x2xf32>
            return %9 : tensor<2x2xf32>
          }
          ```
        </details>
      * `func-bufferize`: DESCRIPTION
        <details>
          <summary>code</summary>

          ```
          module attributes {torch.debug_module_name = "Matmul"} {
            func.func @forward(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
              %0 = bufferization.to_tensor %arg0 : memref<2x2xf32>
              %1 = bufferization.to_memref %0 : memref<2x2xf32>
              %cst = arith.constant 1.000000e+00 : f32
              %cst_0 = arith.constant 0.000000e+00 : f32
              %c2 = arith.constant 2 : index
              %c2_1 = arith.constant 2 : index
              %2 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              %3 = bufferization.to_tensor %2 : memref<2x2xf32>
              linalg.fill ins(%cst : f32) outs(%2 : memref<2x2xf32>)
              %4 = bufferization.to_tensor %2 : memref<2x2xf32>
              %c2_2 = arith.constant 2 : index
              %c2_3 = arith.constant 2 : index
              %5 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              %6 = bufferization.to_tensor %5 : memref<2x2xf32>
              linalg.fill ins(%cst_0 : f32) outs(%5 : memref<2x2xf32>)
              %7 = bufferization.to_tensor %5 : memref<2x2xf32>
              %8 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              memref.copy %5, %8 : memref<2x2xf32> to memref<2x2xf32>
              %9 = bufferization.to_tensor %8 : memref<2x2xf32>
              linalg.matmul ins(%1, %2 : memref<2x2xf32>, memref<2x2xf32>) outs(%8 : memref<2x2xf32>)
              %10 = bufferization.to_tensor %8 : memref<2x2xf32>
              %11 = bufferization.to_memref %10 : memref<2x2xf32>
              return %11 : memref<2x2xf32>
            }
          }
          ```
        </details>
      * `arith-bufferize`: DESCRIPTION
        <details>
          <summary>code</summary>

          ```
          module attributes {torch.debug_module_name = "Matmul"} {
            func.func @forward(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
              %0 = bufferization.to_tensor %arg0 : memref<2x2xf32>
              %cst = arith.constant 1.000000e+00 : f32
              %cst_0 = arith.constant 0.000000e+00 : f32
              %c2 = arith.constant 2 : index
              %c2_1 = arith.constant 2 : index
              %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              %2 = bufferization.to_tensor %1 : memref<2x2xf32>
              linalg.fill ins(%cst : f32) outs(%1 : memref<2x2xf32>)
              %3 = bufferization.to_tensor %1 : memref<2x2xf32>
              %c2_2 = arith.constant 2 : index
              %c2_3 = arith.constant 2 : index
              %4 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              %5 = bufferization.to_tensor %4 : memref<2x2xf32>
              linalg.fill ins(%cst_0 : f32) outs(%4 : memref<2x2xf32>)
              %6 = bufferization.to_tensor %4 : memref<2x2xf32>
              %7 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              memref.copy %4, %7 : memref<2x2xf32> to memref<2x2xf32>
              %8 = bufferization.to_tensor %7 : memref<2x2xf32>
              linalg.matmul ins(%arg0, %1 : memref<2x2xf32>, memref<2x2xf32>) outs(%7 : memref<2x2xf32>)
              %9 = bufferization.to_tensor %7 : memref<2x2xf32>
              return %7 : memref<2x2xf32>
            }
          }
          ```
        </details>
      * `tensor-bufferize` (No change)
      * `finalizing-bufferize`
        <details>
          <summary>code</summary>

          ```
          module attributes {torch.debug_module_name = "Matmul"} {
            func.func @forward(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
              %cst = arith.constant 1.000000e+00 : f32
              %cst_0 = arith.constant 0.000000e+00 : f32
              %c2 = arith.constant 2 : index
              %c2_1 = arith.constant 2 : index
              %0 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              linalg.fill ins(%cst : f32) outs(%0 : memref<2x2xf32>)
              %c2_2 = arith.constant 2 : index
              %c2_3 = arith.constant 2 : index
              %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              linalg.fill ins(%cst_0 : f32) outs(%1 : memref<2x2xf32>)
              %2 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              memref.copy %1, %2 : memref<2x2xf32> to memref<2x2xf32>
              linalg.matmul ins(%arg0, %0 : memref<2x2xf32>, memref<2x2xf32>) outs(%2 : memref<2x2xf32>)
              return %2 : memref<2x2xf32>
            }
          }
          ```
        </details>
      * `refback-munge-calling-conventions`
        <details>
          <summary>code</summary>
          
          ```
          module attributes {torch.debug_module_name = "Matmul"} {
            func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
            func.func @forward(%arg0: memref<*xf32>) attributes {llvm.emit_c_interface} {
              %0 = memref.cast %arg0 : memref<*xf32> to memref<2x2xf32>
              %cst = arith.constant 1.000000e+00 : f32
              %cst_0 = arith.constant 0.000000e+00 : f32
              %c2 = arith.constant 2 : index
              %c2_1 = arith.constant 2 : index
              %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              linalg.fill ins(%cst : f32) outs(%1 : memref<2x2xf32>)
              %c2_2 = arith.constant 2 : index
              %c2_3 = arith.constant 2 : index
              %2 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              linalg.fill ins(%cst_0 : f32) outs(%2 : memref<2x2xf32>)
              %3 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
              memref.copy %2, %3 : memref<2x2xf32> to memref<2x2xf32>
              linalg.matmul ins(%0, %1 : memref<2x2xf32>, memref<2x2xf32>) outs(%3 : memref<2x2xf32>)
              %4 = memref.cast %3 : memref<2x2xf32> to memref<*xf32>
              call @refbackend_consume_func_return_mrf32(%4) : (memref<*xf32>) -> ()
              return
            }
          }
          ```
        </details>
      * `refback-insert-rng-globals` (no change - adds only this RNG line: `memref.global "private" @global_seed : memref<i64> = dense<0>`)
      * `tm-tensor-to-loops` (no change)
      * `refback-munge-memref-copy` (output == `lowered_linalg.mlir`)
      * _Omitted passes_
      * `refback-expand-ops-for-llvm` (no change)
    </details>
