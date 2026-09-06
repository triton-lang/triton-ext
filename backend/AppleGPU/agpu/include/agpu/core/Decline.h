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
// tile shape, ...). One kernel autotuned under N configs stays a single site.
struct DeclineSite {
  msl::Str site;
  msl::Str config;

  bool operator<(const DeclineSite &o) const {
    if (site != o.site)
      return site < o.site;
    return config < o.config;
  }
};

enum class DeclineTag {
  Reject, // a shape not covered; the user may need to change something
  Plan,   // a choice among things all emittable; informational
  Count,
};

struct DeclineTagSpec {
  DeclineTag tag;
  const char *name;
};

inline constexpr DeclineTagSpec kDeclineTags[] = {
    {DeclineTag::Reject, "MSL-REJECT"},
    {DeclineTag::Plan, "MSL-PLAN"},
};

static_assert(sizeof(kDeclineTags) / sizeof(kDeclineTags[0]) ==
                  std::size_t(DeclineTag::Count),
              "every DeclineTag needs a row");

inline const char *tagName(DeclineTag t) {
  for (const DeclineTagSpec &s : kDeclineTags)
    if (s.tag == t)
      return s.name;
  return "";
}

struct DeclineTally {
  msl::Str where;
  msl::Str why;
  DeclineTag tag = DeclineTag::Reject;
  std::size_t occurrences = 0;
  std::vector<msl::Str> sites;
  std::vector<msl::Str> configs;

  std::size_t distinctSites() const { return sites.size(); }
  std::size_t distinctConfigs() const { return configs.size(); }
};

class DeclineLog {
public:
  void record(const Decision &d) { record(d, {}, DeclineTag::Reject); }

  void record(const Decision &d, DeclineSite at) {
    record(d, std::move(at), DeclineTag::Reject);
  }

  void record(const Decision &d, DeclineSite at, DeclineTag tag) {
    if (!d.isDecline())
      return;
    entries_.push_back(d);
    sites_.push_back(std::move(at));
    tags_.push_back(tag);
  }

  void note(const Decision &d, DeclineSite at = {}) {
    record(d, std::move(at), DeclineTag::Plan);
  }

  bool empty() const { return entries_.empty(); }
  std::size_t size() const { return entries_.size(); }
  const std::vector<Decision> &entries() const { return entries_; }

  // Truncates back to where a rebuilt kernel started. Entries below the mark
  // survive.
  void truncate(std::size_t mark) {
    if (mark > entries_.size())
      return;
    entries_.erase(entries_.begin() + (std::ptrdiff_t)mark, entries_.end());
    sites_.erase(sites_.begin() + (std::ptrdiff_t)mark, sites_.end());
    tags_.erase(tags_.begin() + (std::ptrdiff_t)mark, tags_.end());
  }

  bool declined(const msl::Str &why) const {
    for (const Decision &d : entries_)
      if (d.why() == why)
        return true;
    return false;
  }

  // One row per (gate, reason, tag), with occurrences and distinct sites
  // counted separately from distinct configs.
  std::vector<DeclineTally> summary() const {
    std::vector<DeclineTally> out;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const Decision &d = entries_[i];
      std::size_t row = out.size();
      for (std::size_t t = 0; t < out.size(); ++t)
        if (out[t].where == d.where() && out[t].why == d.why() &&
            out[t].tag == tags_[i])
          row = t;
      if (row == out.size())
        out.push_back(DeclineTally{d.where(), d.why(), tags_[i], 0, {}, {}});

      ++out[row].occurrences;
      addDistinct(out[row].sites, sites_[i].site);
      addDistinct(out[row].configs, sites_[i].config);
    }
    return out;
  }

  void printSummary(std::ostream &os) const {
    const std::vector<DeclineTally> rows = summary();
    if (rows.empty())
      return;

    std::size_t rejects = 0;
    for (const DeclineTally &r : rows)
      if (r.tag == DeclineTag::Reject)
        ++rejects;

    os << "MSL-REJECT-SUMMARY\tdistinct rejects: " << rejects
       << "\tplan notes: " << (rows.size() - rejects) << "\n";

    for (DeclineTag tag : {DeclineTag::Reject, DeclineTag::Plan})
      for (const DeclineTally &r : rows) {
        if (r.tag != tag)
          continue;
        os << tagName(r.tag) << "-SITE\t" << r.where << "\t" << r.why
           << "\tsites=" << r.distinctSites()
           << "\tconfigs=" << r.distinctConfigs()
           << "\toccurrences=" << r.occurrences;

        constexpr std::size_t kShow = 3;
        for (std::size_t i = 0; i < r.sites.size() && i < kShow; ++i)
          os << (i ? " " : "\tat ") << r.sites[i];
        if (r.sites.size() > kShow)
          os << " (+" << (r.sites.size() - kShow) << " more)";
        os << "\n";
      }
  }

private:
  // Empty means "not supplied", so it is not counted.
  static void addDistinct(std::vector<msl::Str> &into, const msl::Str &v) {
    if (v.empty())
      return;
    for (const msl::Str &known : into)
      if (known == v)
        return;
    into.push_back(v);
  }

  std::vector<Decision> entries_;
  std::vector<DeclineSite> sites_;
  std::vector<DeclineTag> tags_;
};

} // namespace agpu

#endif // AGPU_DECLINE_H
