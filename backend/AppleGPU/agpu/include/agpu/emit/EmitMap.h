// map_elementwise - inline a scalar body once per group of registers.
//
// A single-block region yields: each inlining binds arguments, emits the body,
// and reads names off the terminator. A multi-block region has more than one
// terminator, so each result element gets a declared variable first and
// whichever block runs assigns to it.
#ifndef AGPU_EMIT_MAP_H
#define AGPU_EMIT_MAP_H

#include "agpu/core/Names.h"
#include "agpu/msl/Context.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MapPlan.h"

#include <string>
#include <vector>

namespace agpu {

// The names of one tensor's registers.
using RegisterNames = std::vector<msl::Str>;

struct MapNames {
  msl::Str capture = "mp"; // one per result element of a multi-block region

  msl::Str captureAt(int group, int slot) const {
    return capture + std::to_string(group) + "_" + std::to_string(slot);
  }
};

// `arguments[a]` is the name bound to block argument `a`; `captures` is empty
// for a yielding region and holds one name per result element otherwise.
struct MapBody {
  std::vector<msl::Str> arguments;
  std::vector<msl::Str> captures;
};

// `sources[s][r]` is source tensor `s`'s register `r`. The result is
// `results[k][r]`, in register order.
//
// `emitBody` is called once per group; it appends to `body` and returns one
// name per result element, ordered by `resultOperand(k, e)`. For a
// multi-block region it assigns to the captures instead and its return is
// not read.
template <typename EmitBodyFn>
inline Decision emitMap(msl::Context &c, msl::Block &body, const MapPlan &plan,
                        const std::vector<RegisterNames> &sources,
                        const std::vector<ElemType> &resultTypes,
                        std::vector<RegisterNames> &results, const MapNames &nm,
                        EmitBodyFn emitBody) {
  if (Decision d = mapDecision(plan); !d.ok())
    return d;
  if ((int)resultTypes.size() != plan.f.numResults)
    return Decision::declined("map_elementwise",
                              "result type count does not match the plan");
  if ((int)sources.size() != plan.f.numSources)
    return Decision::declined("map_elementwise",
                              "source count does not match the plan");
  for (const RegisterNames &s : sources)
    if ((int)s.size() != plan.f.numRegisters)
      return Decision::declined("map_elementwise",
                                "sources disagree on register count");

  results.assign(plan.f.numResults, {});

  for (int g = 0; g < plan.groups(); ++g) {
    MapBody in;
    in.arguments.resize(plan.numBlockArguments());
    for (int s = 0; s < plan.f.numSources; ++s)
      for (int e = 0; e < plan.f.pack; ++e)
        in.arguments[plan.blockArgument(s, e)] =
            sources[s][plan.sourceRegister(g, e)];

    in.captures.resize(plan.numCaptures());
    for (int k = 0; k < plan.f.numResults && plan.needsCaptures(); ++k)
      for (int e = 0; e < plan.f.pack; ++e) {
        const int slot = plan.resultOperand(k, e);
        const msl::Str v = nm.captureAt(g, slot);
        body.push_back(c.declStmt(mslTypeOf(resultTypes[k]), v, nullptr));
        in.captures[slot] = v;
      }

    std::vector<msl::Str> out = emitBody(in, body);
    const std::vector<msl::Str> &yielded =
        plan.needsCaptures() ? in.captures : out;
    if ((int)yielded.size() != plan.numResultOperands())
      return Decision::declined("map_elementwise",
                                "body produced the wrong number of results");
    for (int k = 0; k < plan.f.numResults; ++k)
      for (int e = 0; e < plan.f.pack; ++e)
        results[k].push_back(yielded[plan.resultOperand(k, e)]);
  }
  return Decision::emitted();
}

} // namespace agpu

#endif // AGPU_EMIT_MAP_H
