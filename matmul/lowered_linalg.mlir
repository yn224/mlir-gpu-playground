// RUN: torch-mlir-opt --refback-generalize-tensor-pad --scf-bufferize --tm-tensor-bufferize --linalg-init-tensor-to-alloc-tensor --linalg-bufferize --func-bufferize --arith-bufferize --tensor-bufferize --finalizing-bufferize --refback-munge-calling-conventions --refback-insert-rng-globals --tm-tensor-to-loops --refback-munge-memref-copy --tm-tensor-to-loops $1

#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Matmul"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = memref.cast %arg0 : memref<*xf32> to memref<2x2xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst_0 : f32) outs(%1 : memref<2x2xf32>)
    %2 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst : f32) outs(%2 : memref<2x2xf32>)
    %3 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<2x2xf32>) outs(%3 : memref<2x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    }
    linalg.matmul ins(%0, %1 : memref<2x2xf32>, memref<2x2xf32>) outs(%3 : memref<2x2xf32>)
    %4 = memref.cast %3 : memref<2x2xf32> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%4) : (memref<*xf32>) -> ()
    return
  }
}
