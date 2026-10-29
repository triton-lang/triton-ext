// expand_dims and broadcast re-describe which register holds which
// coordinate, so a 2D tile's values are never copied: the store reads the
// names the load bound, with no threadgroup round trip and no shuffle.
//
// RUN: msl_driver.py %s | %filecheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_rt(%x_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
    %stride = arith.constant dense<32> : tensor<4x1xi32, #blocked>
    %m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_0 = tt.expand_dims %m {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
    %n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %n_1 = tt.expand_dims %n {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %row = arith.muli %m_0, %stride : tensor<4x1xi32, #blocked>
    %row_b = tt.broadcast %row : tensor<4x1xi32, #blocked> -> tensor<4x32xi32, #blocked>
    %col_b = tt.broadcast %n_1 : tensor<1x32xi32, #blocked> -> tensor<4x32xi32, #blocked>
    %off = arith.addi %row_b, %col_b : tensor<4x32xi32, #blocked>
    %po = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #blocked>
    %qo = tt.addptr %po, %off : tensor<4x32x!tt.ptr<f32>, #blocked>, tensor<4x32xi32, #blocked>
    %px = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #blocked>
    %qx = tt.addptr %px, %off : tensor<4x32x!tt.ptr<f32>, #blocked>, tensor<4x32xi32, #blocked>
    %v = tt.load %qx : tensor<4x32x!tt.ptr<f32>, #blocked>
    tt.store %qo, %v : tensor<4x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: kernel void triton_rt
// CHECK-NOT: threadgroup float
// CHECK-NOT: simd_shuffle
