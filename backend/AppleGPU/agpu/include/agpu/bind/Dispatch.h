// Dispatch - which handler lowers an operation.
//
// Insertion order: first match wins.
#ifndef AGPU_DISPATCH_H
#define AGPU_DISPATCH_H

#include "agpu/core/Decline.h"
#include "agpu/core/ValueId.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace agpu {

// One operation, flattened out of whatever IR produced it.
struct OpView {
  std::string_view name; // "arith.addf", "tt.load", ...
  std::vector<ValueId> operands;
  std::vector<ValueId> results;

  // The op's own parameters, flattened. Anything richer (layouts, types) is
  // extracted into a facts struct before dispatch runs.
  std::vector<int64_t> ints;

  // A buffer or callee name, when there is one. Borrowed from the walk, which
  // outlives every handler.
  std::string_view text;

  int64_t intAt(std::size_t i, int64_t fallback = 0) const {
    return i < ints.size() ? ints[i] : fallback;
  }
};

// A handler must return `notMine()` for an op it does not recognise.
// `failed()` stops the search.
using OpHandler = std::function<Decision(const OpView &)>;

// A body plus the op names it claims. An empty `names` means the body decides
// for itself and must be offered every op.
struct OpFamily {
  std::vector<std::string_view> names;
  OpHandler body;

  operator OpHandler() const {
    OpHandler b = body;
    if (names.empty())
      return b;
    return [names = names, body = std::move(b)](const OpView &op) -> Decision {
      for (std::string_view n : names)
        if (n == op.name)
          return body(op);
      return Decision::notMine();
    };
  }
};

// The families, in the order they are tried.
class DispatchTable {
public:
  // `who` is for diagnostics: a decline names the handler that produced it.
  void add(std::string who, OpFamily f) {
    const std::size_t at = entries_.size();
    entries_.push_back({std::move(who), std::move(f.body)});
    if (f.names.empty()) {
      unnamed_.push_back(at);
      return;
    }
    for (std::string_view n : f.names)
      byName_[n].push_back(at);
  }

  void add(std::string who, OpHandler h) {
    add(std::move(who), OpFamily{{}, std::move(h)});
  }

  Decision run(const OpView &op) const {
    std::string ignored;
    return runNamed(op, ignored);
  }

  // Run and report which handler answered. One traversal: a second pass would
  // re-run a handler that already emitted.
  Decision runNamed(const OpView &op, std::string &who) const {
    const auto it = byName_.find(op.name);
    const std::vector<std::size_t> *claim =
        it == byName_.end() ? nullptr : &it->second;
    std::size_t ci = 0, ui = 0;
    while (true) {
      const std::size_t c =
          claim && ci < claim->size() ? (*claim)[ci] : entries_.size();
      const std::size_t u =
          ui < unnamed_.size() ? unnamed_[ui] : entries_.size();
      if (c == entries_.size() && u == entries_.size())
        break;
      const std::size_t at = c < u ? (++ci, c) : (++ui, u);
      const Entry &e = entries_[at];
      const Decision d = e.handler(op);
      if (d.keepLooking())
        continue;
      who = e.who;
      return d;
    }
    who.clear();
    return Decision::declined(std::string(op.name),
                              "no handler for this operation");
  }

  std::size_t size() const { return entries_.size(); }

  std::vector<std::string> order() const {
    std::vector<std::string> out;
    for (const Entry &e : entries_)
      out.push_back(e.who);
    return out;
  }

private:
  struct Entry {
    std::string who;
    OpHandler handler;
  };
  std::vector<Entry> entries_;
  std::unordered_map<std::string_view, std::vector<std::size_t>> byName_;
  std::vector<std::size_t> unnamed_;
};

// A handler that claims exactly the ops in a list.
inline OpFamily forOps(std::vector<std::string_view> names,
                       std::function<Decision(const OpView &)> body) {
  return OpFamily{std::move(names), std::move(body)};
}

} // namespace agpu

#endif // AGPU_DISPATCH_H
