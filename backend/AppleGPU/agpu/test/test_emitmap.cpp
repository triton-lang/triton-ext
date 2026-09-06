// map_elementwise, emitted.
#include "agpu/emit/EmitMap.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

namespace {

RegisterNames regs(const char *prefix, int n) {
  RegisterNames r;
  for (int i = 0; i < n; ++i)
    r.push_back(msl::Str(prefix) + std::to_string(i));
  return r;
}

} // namespace

int main() {
  CASE("each inlining is handed its own group's registers");
  {
    // A source-major read of a group-major list hands inlining 1 a1 a2 b1 b2.
    // The correct registers are a2 a3 b2 b3.
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    const MapPlan plan{{2, 1, 6, 2, false}};
    const std::vector<RegisterNames> sources = {regs("a", 6), regs("b", 6)};
    std::vector<RegisterNames> results;
    std::vector<std::vector<msl::Str>> handed;

    Decision d = emitMap(c, body, plan, sources, {f32()}, results, nm,
                         [&](const MapBody &in, msl::Block &) {
                           handed.push_back(in.arguments);
                           return std::vector<msl::Str>{
                               "r" + std::to_string(handed.size()) + "a",
                               "r" + std::to_string(handed.size()) + "b"};
                         });
    CHECK(d.ok());
    CHECK_EQ((int)handed.size(), 3);
    CHECK_EQ(handed[0], (std::vector<msl::Str>{"a0", "a1", "b0", "b1"}));
    CHECK_EQ(handed[1], (std::vector<msl::Str>{"a2", "a3", "b2", "b3"}));
    CHECK_EQ(handed[2], (std::vector<msl::Str>{"a4", "a5", "b4", "b5"}));
  }

  CASE("results come back in register order across groups");
  {
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    const MapPlan plan{{1, 2, 4, 2, false}};
    std::vector<RegisterNames> results;
    int g = 0;

    Decision d =
        emitMap(c, body, plan, {regs("a", 4)}, {f32(), i32()}, results, nm,
                [&](const MapBody &, msl::Block &) {
                  const std::string t = std::to_string(g++);
                  return std::vector<msl::Str>{"x" + t + "0", "x" + t + "1",
                                               "y" + t + "0", "y" + t + "1"};
                });
    CHECK(d.ok());
    CHECK_EQ((int)results.size(), 2);
    CHECK_EQ(results[0], (RegisterNames{"x00", "x01", "x10", "x11"}));
    CHECK_EQ(results[1], (RegisterNames{"y00", "y01", "y10", "y11"}));
  }

  CASE("a single-block region declares nothing");
  {
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    std::vector<RegisterNames> results;
    Decision d =
        emitMap(c, body, MapPlan{{1, 1, 2, 1, false}}, {regs("a", 2)}, {f32()},
                results, nm, [&](const MapBody &in, msl::Block &b) {
                  b.push_back(c.exprStmt(c.var(in.arguments[0])));
                  return std::vector<msl::Str>{in.arguments[0]};
                });
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK(out.find("a0") != std::string::npos);
    CHECK(out.find("mp") == std::string::npos);
  }

  CASE("a multi-block region declares a capture per result element, typed");
  {
    // Declarations must precede the body: a multi-block region assigns inside
    // branches whose scope ends before anything can read them.
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    std::vector<RegisterNames> results;
    const MapPlan plan{{1, 2, 2, 2, true}};
    std::vector<msl::Str> captured;

    Decision d = emitMap(c, body, plan, {regs("a", 2)}, {f32(), i32()}, results,
                         nm, [&](const MapBody &in, msl::Block &) {
                           captured = in.captures;
                           return std::vector<msl::Str>{};
                         });
    CHECK(d.ok());
    CHECK_EQ((int)captured.size(), 4);
    CHECK_EQ(results[0], (RegisterNames{captured[0], captured[1]}));
    CHECK_EQ(results[1], (RegisterNames{captured[2], captured[3]}));

    const std::string out = render(body);
    CHECK(out.find("float " + captured[0]) != std::string::npos);
    CHECK(out.find("int " + captured[2]) != std::string::npos);
  }

  CASE("a body returning the wrong count declines");
  {
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    std::vector<RegisterNames> results;
    Decision d =
        emitMap(c, body, MapPlan{{1, 1, 2, 2, false}}, {regs("a", 2)}, {f32()},
                results, nm, [](const MapBody &, msl::Block &) {
                  return std::vector<msl::Str>{"only-one"};
                });
    CHECK(d.isDecline());
  }

  CASE("a source list that disagrees with the plan declines");
  {
    msl::Context c;
    msl::Block body;
    const MapNames nm;
    std::vector<RegisterNames> results;
    auto never = [](const MapBody &, msl::Block &) {
      return std::vector<msl::Str>{};
    };
    CHECK((emitMap(c, body, MapPlan{{2, 1, 2, 1, false}}, {regs("a", 2)},
                   {f32()}, results, nm, never)
               .isDecline()));
    CHECK((emitMap(c, body, MapPlan{{2, 1, 2, 1, false}},
                   {regs("a", 2), regs("b", 3)}, {f32()}, results, nm, never)
               .isDecline()));
    CHECK((emitMap(c, body, MapPlan{{1, 2, 2, 1, false}}, {regs("a", 2)},
                   {f32()}, results, nm, never)
               .isDecline()));
  }

  return ::agpu_test::report("EmitMap");
}
