module attributes {torch.debug_module_name = "Matmul"} {
  func.func @forward(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %int2 = torch.constant.int 2
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,2],f32>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
    return %2 : !torch.vtensor<[2,2],f32>
  }
}
