// RecordBuffer.h - a device buffer of fixed-width records behind a word
// header, and the layout text the host parses to read it.
//
// `tt.print` and `tt.assert` both append records the launcher decodes after
// the dispatch; this is the protocol they share.
#ifndef AGPU_RECORD_BUFFER_H
#define AGPU_RECORD_BUFFER_H

#include "agpu/msl/Containers.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agpu {

template <std::int32_t HeaderWords, std::int32_t RecordWords>
class RecordCapacity {
public:
  RecordCapacity() = default;
  explicit RecordCapacity(std::int32_t records) : records_(records) {}

  std::int32_t records() const { return records_; }

  std::int64_t words() const {
    return (std::int64_t)HeaderWords + (std::int64_t)records_ * RecordWords;
  }
  std::int64_t bytes() const { return words() * 4; }

  std::int64_t wordOfRecord(std::int32_t slot) const {
    return (std::int64_t)HeaderWords + (std::int64_t)slot * RecordWords;
  }

private:
  std::int32_t records_ = 0;
};

// Sites number in add order and a body may be built twice, so a caller that
// re-walks must clear() first.
template <class Site> class SiteList {
public:
  std::int32_t add(Site site) {
    site.site = (std::int32_t)sites_.size();
    sites_.push_back(std::move(site));
    return sites_.back().site;
  }

  void clear() { sites_.clear(); }

  // Earlier kernels' sites survive this kernel's rebuild: the module numbers
  // sites across all of them.
  void truncate(std::size_t n) {
    if (n < sites_.size())
      sites_.resize(n);
  }

  bool empty() const { return sites_.empty(); }
  std::size_t siteCount() const { return sites_.size(); }
  const std::vector<Site> &sites() const { return sites_; }

private:
  std::vector<Site> sites_;
};

// The layout text is line-oriented.
inline msl::Str escapeLayoutLine(const msl::Str &s) {
  msl::Str safe;
  for (const char ch : s) {
    if (ch == '\n')
      safe += "\\n";
    else if (ch == '\\')
      safe += "\\\\";
    else
      safe += ch;
  }
  return safe;
}

// One `key=value` per line, all decimal, so the parser needs no schema of
// its own.
template <std::int32_t HeaderWords, std::int32_t RecordWords>
msl::Str recordLayoutHeader(const char *tag, std::int32_t headWord,
                            RecordCapacity<HeaderWords, RecordWords> cap) {
  msl::Str out;
  out += msl::Str(tag) + " headerWords=" + std::to_string(HeaderWords) + "\n";
  out += msl::Str(tag) + " headWord=" + std::to_string(headWord) + "\n";
  out += msl::Str(tag) + " recordWords=" + std::to_string(RecordWords) + "\n";
  out += msl::Str(tag) + " records=" + std::to_string(cap.records()) + "\n";
  out += msl::Str(tag) + " bytes=" + std::to_string(cap.bytes()) + "\n";
  return out;
}

} // namespace agpu

#endif // AGPU_RECORD_BUFFER_H
