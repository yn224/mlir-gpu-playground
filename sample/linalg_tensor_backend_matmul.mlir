module attributes {torch.debug_module_name = "Matmul"} {
  func.func @forward(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = linalg.matmul ins(%arg0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
  }
}