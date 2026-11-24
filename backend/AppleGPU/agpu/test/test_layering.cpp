// The layering, enforced.
//
// Five layers and the dependencies run one way:
//
//   core/  pure arithmetic and the containers -- no AST, no planning
//   msl/   the AST, printer and analyses; may use core, nothing above it
//   plan/  decides; may use core and msl names, must not emit
//   emit/  consumes a plan, produces AST
//   bind/  turns an IR into the facts emit/ consumes -- the top and the
//          only layer that may include any of the others
#include "harness.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> includesOf(const std::string &path) {
  std::vector<std::string> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t hash = line.find("#include \"");
    if (hash == std::string::npos)
      continue;
    const std::size_t start = hash + 10;
    const std::size_t end = line.find('"', start);
    if (end != std::string::npos)
      out.push_back(line.substr(start, end - start));
  }
  return out;
}

bool mentions(const std::vector<std::string> &incs, const std::string &needle) {
  for (const std::string &i : incs)
    if (i.find(needle) != std::string::npos)
      return true;
  return false;
}

const char *kRoot = AGPU_INCLUDE_DIR;

// Every rule below reads a header as text, and a reader that cannot open its
// file returns nothing, which passes every "does not mention" check. So a
// renamed header must fail here, or it silently retires its rule.
std::string hdr(const std::string &rel) {
  std::string path = std::string(kRoot) + "/agpu/" + rel;
  CHECK(std::filesystem::exists(path));
  return path;
}

std::vector<std::string> headersUnder(const std::string &rel) {
  std::vector<std::string> out;
  for (const auto &e : std::filesystem::recursive_directory_iterator(hdr(rel)))
    if (e.is_regular_file() && e.path().extension() == ".h")
      out.push_back(e.path().string());
  return out;
}

std::vector<std::string> linesOf(const std::string &path) {
  std::vector<std::string> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    out.push_back(line);
  return out;
}

// Lines mentioning `needle` outside a comment.
//
// Comments discuss the constants they replaced -- "a second `warp * 32`
// elsewhere is how..." -- and flagging those would make the rule
// unstateable.
std::vector<std::string> codeLinesWith(const std::string &path,
                                       const std::string &needle) {
  std::vector<std::string> out;
  for (const std::string &l : linesOf(path)) {
    const std::size_t comment = l.find("//");
    const std::size_t at = l.find(needle);
    if (at == std::string::npos)
      continue;
    if (comment != std::string::npos && comment < at)
      continue;
    out.push_back(l);
  }
  return out;
}

} // namespace

int main() {
  // ── core/ depends on nothing above it ──────────────────────────────────

  CASE("core headers include nothing from a layer above them");
  {
    // Checked by walking every header actually under core/.
    // The containers used to live in msl/, which put msl/ underneath core/
    // and made the order above a fiction. Reintroducing any upward include
    // fails here.
    for (const std::string &f : headersUnder("core")) {
      auto incs = includesOf(f);
      CHECK(!mentions(incs, "msl/"));
      CHECK(!mentions(incs, "plan/"));
      CHECK(!mentions(incs, "emit/"));
      CHECK(!mentions(incs, "bind/"));
    }
  }

  CASE("msl/ may use core/ and nothing else in agpu");
  {
    for (const std::string &f : headersUnder("msl")) {
      auto incs = includesOf(f);
      CHECK(!mentions(incs, "plan/"));
      CHECK(!mentions(incs, "emit/"));
      CHECK(!mentions(incs, "bind/"));
    }
  }

  // ── plan/ decides and does not emit ───────────────────────────────────

  CASE("planners never build AST");
  {
    // Checked over every header under plan/: a rule enforced against a list
    // is enforced against whoever remembered to extend the list.
    for (const std::string &f : headersUnder("plan")) {
      auto incs = includesOf(f);
      CHECK(!mentions(incs, "msl/Context.h"));
      CHECK(!mentions(incs, "msl/Printer.h"));
      CHECK(!mentions(incs, "emit/"));
    }
  }

  CASE("a planner naming an MSL type is the one allowed exception");
  {
    // A name is not a node: neither includes Context.h, so neither can build
    // an expression.
    for (const char *f : {"plan/CanonicalFragment.h", "plan/Elementwise.h"}) {
      auto incs = includesOf(hdr(f));
      CHECK(!mentions(incs, "msl/Context.h"));
      CHECK(!mentions(incs, "emit/"));
    }
  }

  CASE("a planner may name a builtin, because a name is not a node");
  {
    // msl/Builtins.h is strings and nothing else -- it pulls in no Context
    // and no Ast, so a planner naming a builtin still cannot build a call.
    CHECK(!mentions(includesOf(hdr("msl/Builtins.h")), "msl/Ast.h"));
    CHECK(!mentions(includesOf(hdr("msl/Builtins.h")), "msl/Context.h"));
    CHECK(!mentions(includesOf(hdr("msl/Builtins.h")), "core/"));
    CHECK(!mentions(includesOf(hdr("msl/Builtins.h")), "plan/"));

    auto incs = includesOf(hdr("plan/AtomicPlan.h"));
    CHECK(mentions(incs, "msl/Builtins.h"));
    CHECK(!mentions(incs, "msl/Context.h"));
  }

  // ── emit/ is the only layer that builds nodes ──────────────────────────

  CASE("every emitter consumes the plan header for its subject");
  {
    struct Pair {
      const char *emitter;
      const char *plan;
    };
    const Pair pairs[] = {
        {"emit/EmitDot.h", "plan/DotPlan.h"},
        {"emit/EmitScalar.h", "plan/DotPlan.h"},
        {"emit/EmitScan.h", "plan/ScanPlan.h"},
        {"emit/EmitAtomic.h", "plan/AtomicPlan.h"},
        {"emit/EmitMove.h", "plan/AccessPlan.h"},
        {"emit/EmitBand.h", "plan/BandPlan.h"},
        {"emit/EmitShuffle.h", "plan/ShufflePlan.h"},
        {"emit/EmitPrint.h", "plan/PrintPlan.h"},
        {"emit/EmitAssert.h", "plan/AssertPlan.h"},
    };
    for (const Pair &p : pairs)
      CHECK(mentions(includesOf(hdr(p.emitter)), p.plan));
  }

  CASE("the AST proper depends on nothing in agpu");
  {
    // Stronger than the msl/ sweep above: these six take not even core/.
    for (const char *f : {"msl/Ast.h", "msl/Context.h", "msl/AstWalk.h",
                          "msl/Printer.h", "msl/Analysis.h", "msl/Equal.h"}) {
      auto incs = includesOf(hdr(f));
      CHECK(!mentions(incs, "core/"));
      CHECK(!mentions(incs, "plan/"));
      CHECK(!mentions(incs, "emit/"));
    }
  }

  // ── one hardware fact, one owner ───────────────────────────────────────

  CASE("the warp size is declared once");
  {
    // Checked as TEXT: nothing about a second `int64_t warpSize = 32;` fails
    // to compile.
    const char *headers[] = {
        "core/Units.h",         "plan/ShufflePlan.h",       "plan/ScanPlan.h",
        "plan/ReductionPlan.h", "plan/CanonicalFragment.h", "emit/KernelAbi.h",
        "emit/EmitHistogram.h", "plan/DotPlan.h",           "plan/WarpSlots.h",
        "emit/EmitKernel.h",
    };
    int declarations = 0;
    for (const char *f : headers)
      declarations += (int)codeLinesWith(hdr(f), "kWarpSize = ").size();
    CHECK_EQ(declarations, 1);

    CHECK_EQ((int)codeLinesWith(hdr("core/Units.h"), "kWarpSize = ").size(), 1);

    for (const char *f : headers) {
      if (std::string(f) == "core/Units.h")
        continue;
      CHECK(codeLinesWith(hdr(f), "* 32").empty());
      CHECK(codeLinesWith(hdr(f), "= 32;").empty());
    }
  }

  CASE("the fragment dimension is declared once");
  {
    CHECK_EQ((int)codeLinesWith(hdr("core/Units.h"), "kSgFragDim = ").size(),
             1);
    CHECK(codeLinesWith(hdr("plan/CanonicalFragment.h"), "return 8;").empty());
    CHECK(codeLinesWith(hdr("msl/Builtins.h"), "\"8x8\"").empty());
  }

  CASE("bytes-per-element is spelled once");
  {
    // `byteWidthOf` rounds UP; a hand-written `bits / 8` rounds down. Text,
    // because a second spelling compiles perfectly.
    for (const std::string &f : headersUnder("")) {
      if (f == hdr("plan/ElemType.h"))
        continue;
      CHECK(codeLinesWith(f, "bits / 8").empty());
      CHECK(codeLinesWith(f, "bits/8").empty());
    }
    CHECK_EQ((int)codeLinesWith(hdr("plan/ElemType.h"), "byteWidthOf").size(),
             1);
  }

  CASE("the threadgroup budgets are declared once");
  {
    for (const std::string &f : headersUnder("")) {
      if (f == hdr("core/Units.h"))
        continue;
      CHECK(codeLinesWith(f, "65536").empty());
      CHECK(codeLinesWith(f, "32768").empty());
    }
  }

  // ── bind/ is the top and nothing reaches up into it ───────────────────

  CASE("no layer below bind/ includes it");
  {
    // Keeps `emit/` usable from more than one front end: an emitter that
    // includes `bind/` would know how one particular IR names its values.
    for (const char *f :
         {"emit/EmitDot.h", "emit/EmitScan.h", "emit/EmitMove.h",
          "emit/EmitKernel.h", "emit/EmitRebind.h", "emit/EmitMemDesc.h",
          "emit/EmitPrint.h", "plan/PrintPlan.h", "emit/EmitAssert.h",
          "plan/AssertPlan.h", "plan/DotPlan.h", "plan/RebindPlan.h",
          "core/TileView.h", "core/Decline.h", "msl/Ast.h", "msl/Analysis.h"})
      CHECK(!mentions(includesOf(hdr(f)), "bind/"));
  }

  CASE("bind/ holds no IR types of its own");
  {
    // Checked as text because there is no MLIR here to fail the build --
    // that is exactly the property being defended.
    for (const char *f : {"bind/SymbolTable.h", "bind/Dispatch.h",
                          "bind/LayoutBind.h", "bind/PointerBind.h"}) {
      auto incs = includesOf(hdr(f));
      CHECK(!mentions(incs, "mlir/"));
      CHECK(!mentions(incs, "triton/"));
      CHECK(!mentions(incs, "llvm/"));
      CHECK(codeLinesWith(hdr(f), "mlir::").empty());
      CHECK(codeLinesWith(hdr(f), "Operation *").empty());
    }
  }

  CASE("bind/ may use the layers below it");
  {
    // Stated so the rule above is not mistaken for isolation.
    const auto incs = includesOf(hdr("bind/SymbolTable.h"));
    CHECK(!incs.empty());
  }

  CASE("the bridge's op tables stay readable without MLIR");
  {
    // test_optables.cpp is an agpu-suite test that includes this bridge
    // header, which links only because the header pulls in no MLIR. The
    // decoders that need MLIR types live in AgpuEnums.h instead.
    const std::string f = std::string(AGPU_BRIDGE_DIR) + "/AgpuOpTables.h";
    CHECK(std::filesystem::exists(f));
    const auto incs = includesOf(f);
    CHECK(!mentions(incs, "mlir/"));
    CHECK(!mentions(incs, "triton/"));
    CHECK(!mentions(incs, "llvm/"));
    CHECK(codeLinesWith(f, "Operation *").empty());
    CHECK(codeLinesWith(f, "mlir::Type").empty());
    CHECK(codeLinesWith(f, "mlir::Value").empty());
  }

  return ::agpu_test::report("Layering");
}
