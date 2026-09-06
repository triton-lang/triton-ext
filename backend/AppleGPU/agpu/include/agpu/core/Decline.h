// Decline.h - a refusal to lower an op, carrying the reason for the log.
#ifndef AGPU_DECLINE_H
#define AGPU_DECLINE_H

#include "agpu/core/Containers.h"

#include <ostream>

#include <cstddef>
#include <vector>

namespace agpu {

enum class Outcome {
  Emitted,  // lowered
  NotMine,  // another emitter's op; keep looking
  Declined, // mine, but this shape is not supported
  Failed,   // mine and something went wrong that should not have
};

// A decline is a routing decision (the caller may have a fallback); a
// failure is a bug.
class Decision {
public:
  static Decision emitted() { return Decision(Outcome::Emitted, {}, {}); }
  static Decision notMine() { return Decision(Outcome::NotMine, {}, {}); }
  static Decision failed() { return Decision(Outcome::Failed, {}, {}); }

  static Decision declined(msl::Str where, msl::Str why) {
    return Decision(Outcome::Declined, std::move(where), std::move(why));
  }

  Outcome outcome() const { return kind_; }
  bool ok() const { return kind_ == Outcome::Emitted; }
  bool isDecline() const { return kind_ == Outcome::Declined; }
  bool keepLooking() const { return kind_ == Outcome::NotMine; }
  bool isBug() const { return kind_ == Outcome::Failed; }

  const msl::Str &where() const { return where_; }
  const msl::Str &why() const { return why_; }

  msl::Str message() const {
    if (kind_ != Outcome::Declined)
      return {};
    return where_ + ": " + why_;
  }

private:
  Decision(Outcome k, msl::Str where, msl::Str why)
      : kind_(k), where_(std::move(where)), why_(std::move(why)) {}

  Outcome kind_;
  msl::Str where_, why_;
};

// `site` is where in the source; `config` is the compilation (warp count,
struct DeclineSite {
  msl::Str site;
};

class DeclineLog {
public:
  void record(const Decision &d, DeclineSite at = {}) {
    if (!d.isDecline())
      return;
    entries_.push_back(d);
    sites_.push_back(std::move(at));
  }

  void note(const Decision &d, DeclineSite at = {}) {
    record(d, std::move(at));
  }

  bool empty() const { return entries_.empty(); }
  std::size_t size() const { return entries_.size(); }

  // Truncates back to where a rebuilt kernel started. Entries below the mark
  // survive.
  void truncate(std::size_t mark) {
    if (mark > entries_.size())
      return;
    entries_.erase(entries_.begin() + (std::ptrdiff_t)mark, entries_.end());
    sites_.erase(sites_.begin() + (std::ptrdiff_t)mark, sites_.end());
  }

  void printSummary(std::ostream &os) const {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      os << "MSL-REJECT\t" << entries_[i].where() << "\t" << entries_[i].why();
      if (!sites_[i].site.empty())
        os << "\tat " << sites_[i].site;
      os << "\n";
    }
  }

private:
  std::vector<Decision> entries_;
  std::vector<DeclineSite> sites_;
};

} // namespace agpu

#endif // AGPU_DECLINE_H
