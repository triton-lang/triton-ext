// RUN: loop_split.py %s | %filecheck %s

tt.func @split_kernel_sgt(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is split at mid+1, kept within the loop bounds; first loop
  // with subf, and second with addf
  // CHECK-LABEL: split_kernel_sgt
  // CHECK: %[[NEXT:.*]] = arith.addi %arg2, %c1_i32 : i32
  // CHECK: %[[PAST:.*]] = arith.cmpi sge, %arg2, %arg1 : i32
  // CHECK: %[[SEL:.*]] = arith.select %[[PAST]], %arg1, %[[NEXT]] : i32
  // CHECK: %[[MAX:.*]] = arith.maxsi %[[SEL]], %c0_i32 : i32
  // CHECK: %[[MIDP:.*]] = arith.minsi %[[MAX]], %arg1 : i32
  // CHECK: scf.for {{.*}} = %c0_i32 to %[[MIDP]]
  // CHECK-NOT: scf.if
  // CHECK:   arith.subf
  // CHECK: scf.for {{.*}} = %[[MIDP]] to %arg1
  // CHECK-NOT: scf.if
  // CHECK:   arith.addf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sgt, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_sge(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is split at mid, kept within the loop bounds
  // CHECK-LABEL: split_kernel_sge
  // CHECK-NOT: arith.addi
  // CHECK: %[[MAX:.*]] = arith.maxsi %arg2, %c0_i32 : i32
  // CHECK: %[[MIDP:.*]] = arith.minsi %[[MAX]], %arg1 : i32
  // CHECK: scf.for {{.*}} = %c0_i32 to %[[MIDP]]
  // CHECK-NOT: scf.if
  // CHECK:   arith.subf
  // CHECK: scf.for {{.*}} = %[[MIDP]] to %arg1
  // CHECK-NOT: scf.if
  // CHECK:   arith.addf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sge, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_slt(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the 2 loops; first with addf, second with subf
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK:   arith.addf
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi slt, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_sle(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is split at mid+1
  // CHECK-LABEL: split_kernel_sle
  // CHECK: %[[PLUS1:.*]] = arith.constant 1 : i32
  // CHECK: arith.addi {{.*}}, %[[PLUS1]] : i32
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK:   arith.addf
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sle, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_step10(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %arg2: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the mid-point is kept within the loop bounds, then rounded up to the
  // next iteration
  // CHECK-LABEL: split_kernel_step10
  // CHECK: %[[MAX:.*]] = arith.maxsi %arg2, %c0_i32 : i32
  // CHECK: %[[MIN:.*]] = arith.minsi %[[MAX]], %arg1 : i32
  // CHECK: %[[ITERS:.*]] = arith.ceildivsi %[[MIN]], %c10_i32 : i32
  // CHECK: %[[ROUND:.*]] = arith.muli %[[ITERS]], %c10_i32 : i32
  // CHECK: %[[MIDP:.*]] = arith.minsi %[[ROUND]], %arg1 : i32

  // CHECK: scf.for {{.*}} = %c0_i32 to %[[MIDP]] step %c10_i32
  // CHECK:   arith.addf
  // CHECK: scf.for {{.*}} = %[[MIDP]] to %arg1 step %c10_i32
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c10_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi slt, %arg3, %arg2 : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

// CHECK-LABEL: split_and_remove_load
// CHECK: scf.for
// CHECK:   tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
tt.func @split_and_remove_load(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>

  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %cmp = arith.cmpi slt, %arg3, %mid : i32
    %mask = tt.splat %cmp : i1 -> tensor<256xi1>

    // Load using the splatted mask
    %3 = tt.load %arg5, %mask, %1 : tensor<256x!tt.ptr<f32>>

    %add = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %add, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

// `mid < iv` is `iv > mid`, so the loop is split at mid+1, with constants of
// the induction variable's type.
// CHECK-LABEL: split_kernel_swapped_i64
// CHECK: %[[NEXT:.*]] = arith.addi %arg2, %c1_i64 : i64
// CHECK: %[[PAST:.*]] = arith.cmpi sge, %arg2, %arg1 : i64
// CHECK: %[[SEL:.*]] = arith.select %[[PAST]], %arg1, %[[NEXT]] : i64
// CHECK: %[[MAX:.*]] = arith.maxsi %[[SEL]], %c0_i64 : i64
// CHECK: %[[MIDP:.*]] = arith.minsi %[[MAX]], %arg1 : i64
// CHECK: scf.for {{.*}} = %c0_i64 to %[[MIDP]]
// CHECK-NOT: scf.if
// CHECK:   arith.subf
// CHECK: scf.for {{.*}} = %[[MIDP]] to %arg1
// CHECK-NOT: scf.if
// CHECK:   arith.addf
tt.func @split_kernel_swapped_i64(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i64, %mid: i64) -> tensor<256xf32> {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  %2:2 = scf.for %arg3 = %c0_i64 to %arg1 step %c1_i64 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i64 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi slt, %mid, %arg3 : i64
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}
