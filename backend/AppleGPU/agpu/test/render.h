// The printer bridge: MSL AST to the string a case asserts on.
#ifndef AGPU_TEST_RENDER_H
#define AGPU_TEST_RENDER_H

#include "agpu/msl/Printer.h"

#include <sstream>
#include <string>

namespace agpu_test {

inline std::string render(const agpu::msl::Block &b) {
  std::ostringstream os;
  agpu::msl::Printer p(os);
  p.printBlock(b);
  return os.str();
}

inline std::string render(agpu::msl::Expr *e) {
  std::ostringstream os;
  agpu::msl::Printer p(os);
  p.printExpr(e);
  return os.str();
}

inline std::string render(agpu::msl::Stmt *s) {
  agpu::msl::Block b;
  b.push_back(s);
  return render(b);
}

inline std::string renderType(const agpu::msl::Type &t) {
  std::ostringstream os;
  agpu::msl::Printer p(os);
  p.printType(t);
  return os.str();
}

} // namespace agpu_test

#endif // AGPU_TEST_RENDER_H
