module attributes {torch.debug_module_name = "Matmul"} {
  func.func private @__torch__.Matmul.forward(%arg0: !torch.nn.Module<"__torch__.Matmul">, %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[2,2],f32>}) -> !torch.tensor {
    %none_0 = torch.constant.none
    %int2 = torch.constant.int 2
    %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.aten.ones %1, %none_0, %none_0, %none_0, %none_0 : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
    %3 = torch.aten.matmul %arg1, %2 : !torch.tensor, !torch.tensor -> !torch.tensor
    return %3 : !torch.tensor
  }
  torch.class_type @__torch__.Matmul {
    torch.attr private "training" : !torch.bool
    torch.attr private "_is_full_backward_hook" : !torch.optional<bool>
    torch.method "forward", @__torch__.Matmul.forward
  }
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.nn_module {
    torch.slot "training", %true : !torch.bool
    torch.slot "_is_full_backward_hook", %none : !torch.none
  } : !torch.nn.Module<"__torch__.Matmul">
}
