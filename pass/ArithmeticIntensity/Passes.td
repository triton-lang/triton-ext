#ifndef TRITON_EXT_PASS_ARITHMETICINTENSITY
#define TRITON_EXT_PASS_ARITHMETICINTENSITY

include "mlir/Pass/PassBase.td"

def TritonArithmeticIntensity : Pass</*cli-arg*/"triton-arithmetic-intensity", /*Op*/"mlir::ModuleOp"> {
  let summary = "Calculate the algorithmic arithmetic intensity";
  let description = [{
    This pass will calculate the algorithmic arithmetic intensity of the kernel and set an attribute
    for the equation for each parameter that returns a value. This includes an equation for `compute`
    and `bandwidth` per CTA.
  }];
}

#endif // TRITON_EXT_PASS_ARITHMETICINTENSITY
