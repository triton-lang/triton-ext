// AgpuLog - the walk's trace files.
//
// compiler.py captures the pass's stderr to scan it for an out-of-budget
// message and would swallow a diagnostic written there.
#ifndef AGPU_BRIDGE_LOG_H
#define AGPU_BRIDGE_LOG_H

#include "agpu/core/Gates.h"

#include <cstdlib>
#include <fstream>
#include <string>

namespace mlir::triton::applegpu::bridge {

inline void appendLog(agpu::Gate g, const std::string &text) {
  if (const char *path = ::getenv(std::string(agpu::gateSpec(g).env).c_str())) {
    std::ofstream f(path, std::ios::app);
    f << text;
  }
}

} // namespace mlir::triton::applegpu::bridge

#endif // AGPU_BRIDGE_LOG_H
