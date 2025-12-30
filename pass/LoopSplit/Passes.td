#ifndef TRITON_EXT_PASS_LOOPSPLIT
#define TRITON_EXT_PASS_LOOPSPLIT

include "mlir/Pass/PassBase.td"

def TritonLoopSplit : Pass<"triton-loop-split", "mlir::ModuleOp"> {
  let summary = "Loop splitter";
  let description = [{
    This pass uses a basic analysis that considers induction variable that is compared to loop invariant values.
    When a comparison is encountered, the loop will be split into two consecutive loops with this value as the mid-point
    such that loop1 range -> [lo..midpoint) and loop2 range -> [midpoint..hi).
  }];
  let description = [{
    Bisect a loop when beneficial.
  }];

  let constructor = "mlir::triton_plugin::passes::createLoopSplitPass()";
}

#endif
