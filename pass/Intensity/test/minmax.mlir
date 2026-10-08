// RUN: intensity.py %s | %filecheck %s

// Trip count is max(min(args[2], 4), 0). Per-iteration traffic is 256 * 4 =
// 1024 bytes; multiplication by that positive constant distributes.
// CHECK-LABEL: tt.func @min_trip(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tint.op_count = "0", tint.store_bytes = "max(min(args[2] * 1024, 4096), 0)"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(min(args[2] * 1024, 4096), 0)"}
tt.func @min_trip(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %bound = arith.minsi %N, %c4 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// Same shape with arith.maxsi. max(args[2], 4) * 1024 = max(args[2] * 1024, 4096).
// CHECK-LABEL: tt.func @max_trip(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tint.op_count = "0", tint.store_bytes = "max(args[2] * 1024, 4096)"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 4096)"}
tt.func @max_trip(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %bound = arith.maxsi %N, %c4 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// The loop bound is an scf.if result. The pass upper-bounds it with
// max(then, else) = max(args[2], 8), then distributes * 1024.
// CHECK-LABEL: tt.func @if_bound(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 8192)"}
tt.func @if_bound(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cond = arith.cmpi slt, %N, %c8 : i32
  %bound = scf.if %cond -> (i32) {
    scf.yield %N : i32
  } else {
    scf.yield %c8 : i32
  }
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// Both yields are the same value, so max(e, e) collapses and the trip count
// stays the argument, then the trip count is clamped at zero.
// CHECK-LABEL: tt.func @if_same(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 0)"}
tt.func @if_same(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cond = arith.constant true
  %bound = scf.if %cond -> (i32) {
    scf.yield %N : i32
  } else {
    scf.yield %N : i32
  }
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// max(N + 4, N) differs by the positive constant 4, so the bound is N + 4.
// CHECK-LABEL: tt.func @max_affine_pos(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024 + 4096, 0)"}
tt.func @max_affine_pos(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %n4 = arith.addi %N, %c4 : i32
  %bound = arith.maxsi %n4, %N : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// max(N, N + 4) differs by the negative constant -4, so the bound is N + 4.
// CHECK-LABEL: tt.func @max_affine_neg(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024 + 4096, 0)"}
tt.func @max_affine_neg(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %n4 = arith.addi %N, %c4 : i32
  %bound = arith.maxsi %N, %n4 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// min(N + 4, N) differs by the positive constant 4, so the bound is N.
// CHECK-LABEL: tt.func @min_affine_pos(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 0)"}
tt.func @min_affine_pos(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %n4 = arith.addi %N, %c4 : i32
  %bound = arith.minsi %n4, %N : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// min(N, N + 4) differs by the negative constant -4, so the bound is N.
// CHECK-LABEL: tt.func @min_affine_neg(
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 0)"}
tt.func @min_affine_neg(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %n4 = arith.addi %N, %c4 : i32
  %bound = arith.minsi %N, %n4 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %bound step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
