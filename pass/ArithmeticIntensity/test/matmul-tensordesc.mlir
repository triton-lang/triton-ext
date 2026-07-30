// RUN: arithmetic_intensity.py %s | %filecheck %s

// Block-matmul kernel with `!tt.tensordesc` parameters. The kernel signature
// still carries explicit shape (M, N, K) and stride parameters next to the
// descriptors, mirroring kernels that take both forms. The compute-density pass
// treats `tt.descriptor_load` / `tt.descriptor_store` like their pointer
// counterparts, attributing bandwidth/compute to the descriptor arg
// (`isPointerLikeFuncArgType` accepts `!tt.tensordesc<>`).
// CHECK-LABEL: tt.func @matmul_tensordesc(
// CHECK-SAME:  %arg0: !tt.tensordesc<64x64xf16> {tt.bandwidth = "(args[5] / 64) * 8192"}
// CHECK-SAME:  %arg1: !tt.tensordesc<64x64xf16> {tt.bandwidth = "(args[5] / 64) * 8192"}
// CHECK-SAME:  %arg2: !tt.tensordesc<64x64xf16> {tt.bandwidth = "8192", tt.compute = "(args[5] / 64) * 524288 + 4096"}
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
