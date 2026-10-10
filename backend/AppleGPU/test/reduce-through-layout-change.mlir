// RUN: apple_opt.py %s reduce_through_layout_change | %filecheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: tt.func @max_to_scalar
// CHECK-NOT: ttg.convert_layout
// CHECK: }) : (tensor<64xi32, #ttg.slice<{dim = 1, {{.*}}>>) -> i32
tt.func @max_to_scalar(%x: tensor<64x32xi32, #blocked>) -> i32 {
  %rows = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %m = arith.maxsi %a, %b : i32
    tt.reduce.return %m : i32
  }) : (tensor<64x32xi32, #blocked>) -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %moved = ttg.convert_layout %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi32, #blocked1>
  %all = "tt.reduce"(%moved) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %m = arith.maxsi %a, %b : i32
    tt.reduce.return %m : i32
  }) : (tensor<64xi32, #blocked1>) -> i32
  tt.return %all : i32
}

// CHECK-LABEL: tt.func @argmax_to_scalar
// CHECK-NOT: ttg.convert_layout
// CHECK: }) : (tensor<64xf32, #ttg.slice<{dim = 1, {{.*}}>>, tensor<64xi32, #ttg.slice<{dim = 1, {{.*}}>>) -> (f32, i32)
tt.func @argmax_to_scalar(%v: tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, %i: tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> (f32, i32) {
  %mv = ttg.convert_layout %v : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
  %mi = ttg.convert_layout %i : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xi32, #blocked1>
  %r:2 = "tt.reduce"(%mv, %mi) <{axis = 0 : i32}> ({
  ^bb0(%va: f32, %ia: i32, %vb: f32, %ib: i32):
    %gt = arith.cmpf ogt, %va, %vb : f32
    %bv = arith.select %gt, %va, %vb : f32
    %bi = arith.select %gt, %ia, %ib : i32
    tt.reduce.return %bv, %bi : f32, i32
  }) : (tensor<64xf32, #blocked1>, tensor<64xi32, #blocked1>) -> (f32, i32)
  tt.return %r#0, %r#1 : f32, i32
}

// CHECK-LABEL: tt.func @tensor_result
// CHECK: %[[MOVED:[0-9a-z_]+]] = ttg.convert_layout
// CHECK: "tt.reduce"(%[[MOVED]])
tt.func @tensor_result(%x: tensor<64x32xi32, #blocked>) -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> {
  %moved = ttg.convert_layout %x : tensor<64x32xi32, #blocked> -> tensor<64x32xi32, #blocked2>
  %cols = "tt.reduce"(%moved) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %m = arith.maxsi %a, %b : i32
    tt.reduce.return %m : i32
  }) : (tensor<64x32xi32, #blocked2>) -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
  tt.return %cols : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
}

// CHECK-LABEL: tt.func @sources_disagree
// CHECK: %[[MV:[0-9a-z_]+]] = ttg.convert_layout
// CHECK: %[[MI:[0-9a-z_]+]] = ttg.convert_layout
// CHECK: "tt.reduce"(%[[MV]], %[[MI]])
tt.func @sources_disagree(%v: tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, %i: tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>) -> (f32, i32) {
  %mv = ttg.convert_layout %v : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
  %mi = ttg.convert_layout %i : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<64xi32, #blocked1>
  %r:2 = "tt.reduce"(%mv, %mi) <{axis = 0 : i32}> ({
  ^bb0(%va: f32, %ia: i32, %vb: f32, %ib: i32):
    %gt = arith.cmpf ogt, %va, %vb : f32
    %bv = arith.select %gt, %va, %vb : f32
    %bi = arith.select %gt, %ia, %ib : i32
    tt.reduce.return %bv, %bi : f32, i32
  }) : (tensor<64xf32, #blocked1>, tensor<64xi32, #blocked1>) -> (f32, i32)
  tt.return %r#0, %r#1 : f32, i32
}

// CHECK-LABEL: tt.func @one_operand_in_place
// CHECK: %[[MV:[0-9a-z_]+]] = ttg.convert_layout
// CHECK: "tt.reduce"(%[[MV]], %arg1)
tt.func @one_operand_in_place(%v: tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, %i: tensor<64xi32, #blocked1>) -> (f32, i32) {
  %mv = ttg.convert_layout %v : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
  %r:2 = "tt.reduce"(%mv, %i) <{axis = 0 : i32}> ({
  ^bb0(%va: f32, %ia: i32, %vb: f32, %ib: i32):
    %gt = arith.cmpf ogt, %va, %vb : f32
    %bv = arith.select %gt, %va, %vb : f32
    %bi = arith.select %gt, %ia, %ib : i32
    tt.reduce.return %bv, %bi : f32, i32
  }) : (tensor<64xf32, #blocked1>, tensor<64xi32, #blocked1>) -> (f32, i32)
  tt.return %r#0, %r#1 : f32, i32
}

}
