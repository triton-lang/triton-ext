// RUN: intensity.py %s | %filecheck %s

// Block-matmul kernel with `!tt.tensordesc` parameters. The kernel signature
// still carries explicit shape (M, N, K) and stride parameters next to the
// descriptors, mirroring kernels that take both forms. The compute-density pass
// treats `tt.descriptor_load` / `tt.descriptor_store` like their pointer
// counterparts, attributing load/store bytes and compute to the descriptor arg
// (`isPointerLikeFuncArgType` accepts `!tt.tensordesc<>`).
// CHECK-LABEL: tt.func @matmul_tensordesc(
// CHECK-SAME:  %arg0: !tt.tensordesc<64x64xf16> {tint.load_bytes = "max(ceildiv(args[5], 64) * 8192, 0)"}
// CHECK-SAME:  %arg1: !tt.tensordesc<64x64xf16> {tint.load_bytes = "max(ceildiv(args[5], 64) * 8192, 0)"}
// CHECK-SAME:  %arg2: !tt.tensordesc<64x64xf16> {tint.op_count = "max(ceildiv(args[5], 64) * 524288 + 4096, 4096)", tint.store_bytes = "8192"}
tt.func @matmul_tensordesc(%A: !tt.tensordesc<64x64xf16>,
                           %B: !tt.tensordesc<64x64xf16>,
                           %C: !tt.tensordesc<64x64xf16>,
                           %M: i32, %N: i32, %K: i32,
                           %sam: i32, %sak: i32,
                           %sbk: i32, %sbn: i32,
                           %scm: i32, %scn: i32) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %zero = arith.constant 0.0 : f32
  %acc0 = tt.splat %zero : f32 -> tensor<64x64xf32>
  %final = scf.for %k = %c0 to %K step %c64
      iter_args(%acc = %acc0) -> (tensor<64x64xf32>) : i32 {
    %a = tt.descriptor_load %A[%c0, %k] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
    %b = tt.descriptor_load %B[%k, %c0] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
    %d = tt.dot %a, %b, %acc :
        tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    scf.yield %d : tensor<64x64xf32>
  }
  %final_f16 = arith.truncf %final : tensor<64x64xf32> to tensor<64x64xf16>
  tt.descriptor_store %C[%c0, %c0], %final_f16 :
      !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
  tt.return
}

// Epilogue sub-tiling: the accumulator is written back by two stores of one
// half each (`tt.reshape` + `tt.trans` + `tt.split`, as `EPILOGUE_SUBTILE`
// does in the persistent-matmul tutorial). The dot FLOPs feed both stores but
// must be counted once. `tt.reshape`, `tt.trans` and `tt.split` contribute
// nothing; only the per-half `truncf` (2 * 2048) is added on top.
// CHECK-LABEL: tt.func @matmul_tensordesc_subtile(
// CHECK-SAME:  %arg0: !tt.tensordesc<64x64xf16> {tint.load_bytes = "max(ceildiv(args[3], 64) * 8192, 0)"}
// CHECK-SAME:  %arg1: !tt.tensordesc<64x64xf16> {tint.load_bytes = "max(ceildiv(args[3], 64) * 8192, 0)"}
// CHECK-SAME:  %arg2: !tt.tensordesc<64x32xf16> {tint.op_count = "max(ceildiv(args[3], 64) * 524288 + 4096, 4096)", tint.store_bytes = "8192"}
tt.func @matmul_tensordesc_subtile(%A: !tt.tensordesc<64x64xf16>,
                                   %B: !tt.tensordesc<64x64xf16>,
                                   %C: !tt.tensordesc<64x32xf16>,
                                   %K: i32) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %zero = arith.constant 0.0 : f32
  %acc0 = tt.splat %zero : f32 -> tensor<64x64xf32>
  %final = scf.for %k = %c0 to %K step %c64
      iter_args(%acc = %acc0) -> (tensor<64x64xf32>) : i32 {
    %a = tt.descriptor_load %A[%c0, %k] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
    %b = tt.descriptor_load %B[%k, %c0] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
    %d = tt.dot %a, %b, %acc :
        tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    scf.yield %d : tensor<64x64xf32>
  }
  %r = tt.reshape %final : tensor<64x64xf32> -> tensor<64x2x32xf32>
  %t = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32> -> tensor<64x32x2xf32>
  %lo, %hi = tt.split %t : tensor<64x32x2xf32> -> tensor<64x32xf32>
  %lo_f16 = arith.truncf %lo : tensor<64x32xf32> to tensor<64x32xf16>
  tt.descriptor_store %C[%c0, %c0], %lo_f16 :
      !tt.tensordesc<64x32xf16>, tensor<64x32xf16>
  %hi_f16 = arith.truncf %hi : tensor<64x32xf32> to tensor<64x32xf16>
  tt.descriptor_store %C[%c0, %c32], %hi_f16 :
      !tt.tensordesc<64x32xf16>, tensor<64x32xf16>
  tt.return
}
