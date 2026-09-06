// Emitting a rebinding, which usually means emitting nothing.
#include "agpu/emit/EmitRebind.h"
#include "agpu/msl/Printer.h"
#include "fixtures.h"
#include "harness.h"
#include "render.h"

#include <sstream>

using namespace agpu;
using agpu_test::render;

namespace {

std::vector<msl::Str> names(const char *prefix, int n) {
  std::vector<msl::Str> v;
  for (int i = 0; i < n; ++i)
    v.push_back(msl::Str(prefix) + std::to_string(i));
  return v;
}

using agpu_test::coordsOfShape;

Rebind splat(std::size_t regs) {
  Rebind out;
  out.from.assign(regs, 0);
  return out;
}

Rebind transposed(const std::vector<RegCoord> &res,
                  const std::vector<RegCoord> &src) {
  return rebind(res, indexByCoord(src), [](const RegCoord &rc, RegCoord &want) {
    want = {rc[1], rc[0]};
    return true;
  });
}

std::vector<Rebind> joined(const std::vector<RegCoord> &res,
                           const std::vector<RegCoord> &src) {
  std::vector<Rebind> out;
  for (int which = 0; which < 2; ++which) {
    Rebind r = rebind(res, indexByCoord(src),
                      [which](const RegCoord &rc, RegCoord &want) {
                        if (rc[1] != which)
                          return false;
                        want = {rc[0]};
                        return true;
                      });
    r.sourceIndex = which;
    out.push_back(std::move(r));
  }
  return out;
}

} // namespace

int main() {
  CASE("an aliased rebinding emits no text at all");
  {
    const std::vector<RegCoord> src = coordsOfShape({2, 3});
    const std::vector<RegCoord> res = coordsOfShape({3, 2});
    const Rebind r = transposed(res, src);
    CHECK(rebindDecision(r).ok());

    const std::vector<msl::Str> out = aliasRebind(r, names("s", 6));
    CHECK(allNamed(out));
    CHECK_EQ((int)out.size(), 6);
    CHECK_EQ(out[1], msl::Str("s3"));
  }

  CASE("splat aliases every result to the one source name");
  {
    const std::vector<msl::Str> out = aliasRebind(splat(4), {msl::Str("v")});
    CHECK(allNamed(out));
    for (const msl::Str &n : out)
      CHECK_EQ(n, msl::Str("v"));
  }

  CASE("a copy declares each result and reads its source");
  {
    msl::Context c;
    msl::Block body;
    const Decision d =
        copyRebind(c, body, splat(2), names("s", 1), names("d", 2), f32());
    CHECK(d.ok());
    const std::string out = render(body);
    CHECK(out.find("float d0 = s0") != std::string::npos);
    CHECK(out.find("float d1 = s0") != std::string::npos);
  }

  CASE("a copy with the wrong number of destinations declines");
  {
    msl::Context c;
    msl::Block body;
    CHECK(copyRebind(c, body, splat(4), names("s", 1), names("d", 2), f32())
              .isDecline());
    CHECK(render(body).empty());
  }

  CASE("an incomplete plan declines, emitting no partial copy");
  {
    // Otherwise the mapped registers are declared and the rest read as
    // undeclared names.
    msl::Context c;
    msl::Block body;
    Rebind partial;
    partial.from = {0, -1};
    CHECK(copyRebind(c, body, partial, names("s", 1), names("d", 2), f32())
              .isDecline());
    CHECK(render(body).empty());
  }

  CASE("a join takes each register from whichever source claimed it");
  {
    const std::vector<RegCoord> src = coordsOfShape({3});
    const std::vector<RegCoord> res = coordsOfShape({3, 2});
    const std::vector<Rebind> rs = joined(res, src);

    const std::vector<msl::Str> out =
        aliasJoin(rs, {names("a", 3), names("b", 3)});
    CHECK(allNamed(out));
    CHECK_EQ((int)out.size(), 6);
    CHECK_EQ(out[0], msl::Str("a0"));
    CHECK_EQ(out[1], msl::Str("b0"));
    CHECK_EQ(out[2], msl::Str("a1"));
    CHECK_EQ(out[5], msl::Str("b2"));
  }

  CASE("an unnamed register is reported by the emitter itself");
  {
    Rebind partial;
    partial.from = {0, -1};
    const std::vector<msl::Str> out = aliasRebind(partial, names("s", 1));
    CHECK(!allNamed(out));
    CHECK(out[1].empty());

    CHECK(!allNamed({}));
  }

  return ::agpu_test::report("EmitRebind");
}
