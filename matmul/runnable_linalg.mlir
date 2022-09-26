#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Matmul"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<2x2xf32>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst_0 : f32) outs(%1 : memref<2x2xf32>)
    %2 = memref.alloc() {alignment = 128 : i64} : memref<2x2xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<2x2xf32>) outs(%2 : memref<2x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    }
    linalg.matmul ins(%arg0, %0 : memref<2x2xf32>, memref<2x2xf32>) outs(%2 : memref<2x2xf32>)
    return %2 : memref<2x2xf32>
  }
  func.func @main() {
    %A = memref.alloc() : memref<2x2xf32>

    %cst2 = arith.constant 2.0 : f32
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cst5 = arith.constant 5.0 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    memref.store %cst2, %A[%c0, %c0] : memref<2x2xf32>
    memref.store %cst3, %A[%c0, %c1] : memref<2x2xf32>
    memref.store %cst4, %A[%c1, %c0] : memref<2x2xf32>
    memref.store %cst5, %A[%c1, %c1] : memref<2x2xf32>

    %cast_A = memref.cast %A : memref<2x2xf32> to memref<*xf32>
    gpu.host_register %cast_A : memref<*xf32>

    %B = call @forward(%A) : (memref<2x2xf32>) -> (memref<2x2xf32>)
    %cast_B = memref.cast %B : memref<2x2xf32> to memref<*xf32>

    call @printMemrefF32(%cast_B) : (memref<*xf32>) -> ()

    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}