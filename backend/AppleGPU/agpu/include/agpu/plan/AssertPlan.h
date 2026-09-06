// AssertPlan.h - what a failed `tt.assert` records.
//
// Metal has no trap or abort. The failing thread records the failure and
// returns; the launcher turns the record into an exception after the dispatch.
// Other threads keep running. A failed assert leaves the kernel's outputs
// undefined.
#ifndef AGPU_ASSERT_PLAN_H
#define AGPU_ASSERT_PLAN_H

#include "agpu/msl/Containers.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agpu {

// A record's fields, in order, as 32-bit words; the buffer is
// `device atomic_uint *`.
enum class AssertField : std::int32_t {
  Site = 0, // which tt.assert failed; the host holds messages/locations
  Pid = 1,  // threadgroup_position_in_grid.x
  Tid = 2,  // flat thread index within the threadgroup

  Count = 3,
};

// The spelling each field takes in the layout description.
inline constexpr struct {
  const char *name;
  AssertField field;
} kAssertFieldNames[] = {
    {"site", AssertField::Site},
    {"pid", AssertField::Pid},
    {"tid", AssertField::Tid},
};

inline constexpr std::int32_t kAssertRecordWords =
    static_cast<std::int32_t>(AssertField::Count);

inline constexpr std::int32_t assertFieldWord(AssertField f) {
  return static_cast<std::int32_t>(f);
}

// One word, the number of failures the kernel tried to record. Bumped before
// the bounds test, so an overrun still reports the true count.
enum class AssertHeader : std::int32_t {
  Head = 0,

  Count = 1,
};

inline constexpr std::int32_t kAssertHeaderWords =
    static_cast<std::int32_t>(AssertHeader::Count);

inline constexpr std::int32_t assertHeaderWord(AssertHeader h) {
  return static_cast<std::int32_t>(h);
}

// Small: only the first failure is the diagnostic. Its own buffer, so a
// kernel printing in a hot loop cannot push the assert record out.
inline constexpr std::int32_t kAssertBufferRecords = 64;

class AssertCapacity {
public:
  AssertCapacity() = default;
  explicit AssertCapacity(std::int32_t records) : records_(records) {}

  std::int32_t records() const { return records_; }

  std::int64_t words() const {
    return (std::int64_t)kAssertHeaderWords +
           (std::int64_t)records_ * kAssertRecordWords;
  }
  std::int64_t bytes() const { return words() * 4; }

  std::int64_t wordOfRecord(std::int32_t slot) const {
    return (std::int64_t)kAssertHeaderWords +
           (std::int64_t)slot * kAssertRecordWords;
  }

private:
  std::int32_t records_ = 0;
};

inline AssertCapacity assertCapacity() {
  return AssertCapacity(kAssertBufferRecords);
}

enum class AssertHalt {
  // Record and return. The thread abandons its remaining work.
  Return,
  // Record and carry on, where returning would desynchronise a barrier the
  // rest of the threadgroup still reaches.
  Continue,
};

struct AssertContext {
  // Whether anything after this point synchronises across threads: a barrier,
  // or a reduction or scan containing one.
  bool barrierFollows = false;
};

// A thread that returns early never reaches a later barrier and a barrier
// some threads do not reach is undefined in Metal.
inline AssertHalt assertHaltFor(const AssertContext &ctx) {
  if (ctx.barrierFollows)
    return AssertHalt::Continue;
  return AssertHalt::Return;
}

// The message and source location live on the host: the kernel writes an
// index and the launcher resolves it.
struct AssertSite {
  std::int32_t site = 0;
  msl::Str message;
  msl::Str file;
  std::int32_t line = 0;
  AssertHalt halt = AssertHalt::Return;
};

// The kernel's ABI depends on whether this is empty: no sites means no
// binding and no allocation.
class AssertPlan {
public:
  std::int32_t add(AssertSite site) {
    site.site = (std::int32_t)sites_.size();
    sites_.push_back(std::move(site));
    return sites_.back().site;
  }

  void clear() { sites_.clear(); }

  // Not `clear()`: earlier kernels' sites must survive a rebuild, since the
  // module numbers messages across all of them.
  void truncate(std::size_t n) {
    if (n < sites_.size())
      sites_.resize(n);
  }

  bool asserts() const { return !sites_.empty(); }
  std::size_t siteCount() const { return sites_.size(); }
  const std::vector<AssertSite> &sites() const { return sites_; }

  AssertCapacity capacity() const { return assertCapacity(); }

private:
  std::vector<AssertSite> sites_;
};

inline constexpr const char *kAssertLayoutTag = "AGPU-ASSERT-LAYOUT";

// Rendered for the host to parse, so the launcher does not restate the record
// layout.
inline msl::Str assertLayoutText(const AssertPlan &plan) {
  const AssertCapacity cap = plan.capacity();
  msl::Str out;
  out += msl::Str(kAssertLayoutTag) +
         " headerWords=" + std::to_string(kAssertHeaderWords) + "\n";
  out += msl::Str(kAssertLayoutTag) +
         " headWord=" + std::to_string(assertHeaderWord(AssertHeader::Head)) +
         "\n";
  out += msl::Str(kAssertLayoutTag) +
         " recordWords=" + std::to_string(kAssertRecordWords) + "\n";
  out += msl::Str(kAssertLayoutTag) +
         " records=" + std::to_string(cap.records()) + "\n";
  out += msl::Str(kAssertLayoutTag) + " bytes=" + std::to_string(cap.bytes()) +
         "\n";

  for (const auto &f : kAssertFieldNames)
    out += msl::Str(kAssertLayoutTag) + " field." + f.name + "=" +
           std::to_string(assertFieldWord(f.field)) + "\n";

  out += msl::Str(kAssertLayoutTag) +
         " sites=" + std::to_string(plan.siteCount()) + "\n";
  for (const AssertSite &s : plan.sites()) {
    // The description is line-oriented, so escape newlines.
    msl::Str safe;
    for (const char ch : s.message) {
      if (ch == '\n')
        safe += "\\n";
      else if (ch == '\\')
        safe += "\\\\";
      else
        safe += ch;
    }
    const msl::Str at = std::to_string(s.site);
    out += msl::Str(kAssertLayoutTag) + " where." + at + "=" + s.file + ":" +
           std::to_string(s.line) + "\n";
    out += msl::Str(kAssertLayoutTag) + " msg." + at + "=" + safe + "\n";
  }
  return out;
}

} // namespace agpu

#endif // AGPU_ASSERT_PLAN_H
