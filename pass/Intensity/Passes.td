#ifndef TRITON_EXT_PASS_INTENSITY
#define TRITON_EXT_PASS_INTENSITY

include "mlir/Pass/PassBase.td"

def TritonIntensity : Pass</*cli-arg*/"triton-intensity", /*Op*/"mlir::ModuleOp"> {
  let summary = "Calculate the algorithmic intensity";
  let description = [{
    This pass will calculate the algorithmic intensity of the kernel and set an attribute
    for the equation for each parameter that returns a value. This includes an equation for
    `tint.op_count`, `tint.load_bytes` and `tint.store_bytes` per CTA.
  }];
}

#endif // TRITON_EXT_PASS_INTENSITY
