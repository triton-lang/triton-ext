// Accessing through a memdesc handle.
#include "agpu/emit/EmitMemDesc.h"
#include "agpu/msl/Printer.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

int main() {
  CASE("a compile-time coordinate folds to one literal subscript");
  {
    msl::Context c;
    const MemDesc m = allocMemDesc("pool", {8, 16});
    CHECK_EQ(render(memDescElem(c, m, {2, 3})), std::string("pool[35]"));
    CHECK_EQ(render(memDescElem(c, m, {0, 0})), std::string("pool[0]"));
  }

  CASE("the runtime expression agrees with the compile-time offset");
  {
    msl::Context c;
    const MemDesc m = allocMemDesc("pool", {4, 6});
    for (int64_t i = 0; i < 4; ++i)
      for (int64_t j = 0; j < 6; ++j) {
        const std::string runtime =
            render(memDescElemAt(c, m, {c.lit(i), c.lit(j)}));
        const std::string compile = render(memDescElem(c, m, {i, j}));
        CHECK_EQ(runtime, compile);
      }
  }

  CASE("a genuinely runtime index emits the view's own stride");
  {
    msl::Context c;
    const MemDesc m = allocMemDesc("pool", {4, 6});
    CHECK_EQ(render(memDescElemAt(c, m, {c.var("r"), c.var("k")})),
             std::string("pool[r * 6 + k]"));
  }

  CASE("a null component contributes no term at all");
  {
    msl::Context c;
    const MemDesc m = allocMemDesc("pool", {4, 6});
    CHECK_EQ(render(memDescElemAt(c, m, {c.var("r"), nullptr})),
             std::string("pool[r * 6]"));
    CHECK_EQ(render(memDescElemAt(c, m, {nullptr, c.var("k")})),
             std::string("pool[k]"));
    CHECK_EQ(render(memDescElemAt(c, m, {nullptr, nullptr})),
             std::string("pool[0]"));
  }

  CASE("a stride of one emits no multiply");
  {
    msl::Context c;
    const MemDesc m = allocMemDesc("pool", {4, 6});
    const std::string s = render(memDescElemAt(c, m, {nullptr, c.var("k")}));
    CHECK(s.find("*") == std::string::npos);
  }

  // ── the handle operations ──────────────────────────────────────────────

  CASE("a slice's corner is carried by the view");
  {
    msl::Context c;
    const MemDesc all = allocMultiBuffered("pool", 3, {4, 6});
    const MemDesc one = all.index(2);

    CHECK_EQ(render(memDescElem(c, one, {0, 0})), std::string("pool[48]"));
    CHECK_EQ(render(memDescElemAt(c, one, {nullptr, nullptr})),
             std::string("pool[48]"));
    CHECK_EQ(render(memDescElemAt(c, one, {c.var("r"), c.var("k")})),
             std::string("pool[48 + r * 6 + k]"));
  }

  CASE("a subslice's corner is carried the same way");
  {
    msl::Context c;
    const MemDesc full = allocMemDesc("pool", {8, 8});
    const MemDesc corner = full.subslice({2, 3}, {4, 4});

    CHECK_EQ(corner.offsetOf({0, 0}), 2 * 8 + 3);
    CHECK_EQ(render(memDescElem(c, corner, {0, 0})), std::string("pool[19]"));
    CHECK_EQ(render(memDescElemAt(c, corner, {c.var("r"), nullptr})),
             std::string("pool[19 + r * 8]"));
  }

  CASE("the runtime and compile-time forms agree on a subslice too");
  {
    msl::Context c;
    const MemDesc corner = allocMemDesc("pool", {8, 8}).subslice({2, 3});
    for (int64_t i = 0; i < 3; ++i)
      for (int64_t j = 0; j < 3; ++j)
        CHECK_EQ(render(memDescElemAt(c, corner, {c.lit(i), c.lit(j)})),
                 render(memDescElem(c, corner, {i, j})));
  }

  // ── the declaration ────────────────────────────────────────────────────

  CASE("the buffer is sized from its own handle");
  {
    msl::Context c;
    const MemDesc m = allocMultiBuffered("pool", 3, {4, 6});
    const std::string s =
        render(memDescDecl(c, m, msl::Type::scalar(msl::Scalar::F32)));
    CHECK(s.find("threadgroup") != std::string::npos);
    CHECK(s.find("pool[72]") != std::string::npos);
    CHECK_EQ(m.cosizeElems(), 72);
  }

  CASE("the last addressable element is inside the declared buffer");
  {
    const MemDesc m = allocMultiBuffered("pool", 3, {4, 6});
    int64_t peak = 0;
    for (int64_t s = 0; s < 3; ++s)
      for (int64_t i = 0; i < 4; ++i)
        for (int64_t j = 0; j < 6; ++j) {
          const int64_t off = m.offsetOf({s, i, j});
          CHECK(off < m.cosizeElems());
          if (off > peak)
            peak = off;
        }
    CHECK_EQ(peak, m.cosizeElems() - 1);
  }

  return ::agpu_test::report("EmitMemDesc");
}
