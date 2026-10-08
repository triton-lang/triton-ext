// RUN: intensity.py %s | %filecheck %s

// The dynamic loop range becomes a symbol on the function arg. Uses the typical
// kernel idiom: scalar `!tt.ptr<f32>` parameters with the address tensor built
// from `tt.make_range` + `tt.splat` + `tt.addptr`.
//
// Per-iter bytes = 256 * 4 = 1024. Trip count is max(args[2], 0); multiplying
// by the positive byte count distributes to max(args[2] * 1024, 0).
// CHECK-LABEL: tt.func @param_loop(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tint.op_count = "0", tint.store_bytes = "max(args[2] * 1024, 0)"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(args[2] * 1024, 0)"}
tt.func @param_loop(%out: !tt.ptr<f32>,
                    %in: !tt.ptr<f32>,
                    %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %i = %c0 to %N step %c1 : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}

// A persistent loop `for tile in range(pid, num_tiles, NUM_SMS)` runs
// ceil((num_tiles - pid) / NUM_SMS) times, not floor: with num_tiles <= NUM_SMS
// every program still processes one tile.
//
// Per-iter bytes = 256 * 4 = 1024. Trip count is
// max(ceildiv(args[2] - pid, 170), 0), which distributes over * 1024.
// CHECK-LABEL: tt.func @persistent_loop(
// CHECK-SAME:  %arg0: !tt.ptr<f32> {tint.op_count = "0", tint.store_bytes = "max(ceildiv(args[2] - program_id[0], 170) * 1024, 0)"}
// CHECK-SAME:  %arg1: !tt.ptr<f32> {tint.load_bytes = "max(ceildiv(args[2] - program_id[0], 170) * 1024, 0)"}
tt.func @persistent_loop(%out: !tt.ptr<f32>,
                         %in: !tt.ptr<f32>,
                         %num_tiles: i32) {
  %pid = tt.get_program_id x : i32
  %num_sms = arith.constant 170 : i32
  %off = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %inb = tt.splat %in : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %outb = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %inp = tt.addptr %inb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %outp = tt.addptr %outb, %off : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %tile = %pid to %num_tiles step %num_sms : i32 {
    %v = tt.load %inp : tensor<256x!tt.ptr<f32>>
    tt.store %outp, %v : tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
