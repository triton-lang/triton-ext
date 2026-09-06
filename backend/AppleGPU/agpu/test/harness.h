// Minimal test harness.
#ifndef AGPU_TEST_HARNESS_H
#define AGPU_TEST_HARNESS_H

#include "agpu/core/Containers.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace agpu_test {

inline int &failures() {
  static int n = 0;
  return n;
}
inline int &checks() {
  static int n = 0;
  return n;
}
inline const char *&currentCase() {
  static const char *s = "";
  return s;
}

inline void fail(const char *file, int line, const std::string &what) {
  ++failures();
  std::printf("  FAIL  %s\n        %s:%d\n        %s\n", currentCase(), file,
              line, what.c_str());
}

template <class T> std::string show(const T &v) { return std::to_string(v); }
inline std::string show(const std::string &s) { return "\"" + s + "\""; }
inline std::string show(const char *s) { return std::string("\"") + s + "\""; }
inline std::string show(bool b) { return b ? "true" : "false"; }
template <class A, class B> std::string show(const std::pair<A, B> &p);
template <class T> std::string show(const std::vector<T> &v) {
  std::string s = "{";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i)
      s += ", ";
    s += show(v[i]);
  }
  return s + "}";
}
template <class A, class B> std::string show(const std::pair<A, B> &p) {
  return "(" + show(p.first) + ", " + show(p.second) + ")";
}
template <class T, unsigned N>
std::string show(const agpu::msl::SmallVector<T, N> &v) {
  std::string s = "{";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i)
      s += ", ";
    s += show(v[i]);
  }
  return s + "}";
}

template <class A, class B>
void checkEq(const A &a, const B &b, const char *ea, const char *eb,
             const char *file, int line) {
  ++checks();
  if (!(a == b))
    fail(file, line,
         std::string(ea) + " == " + eb + "\n        got: " + show(a) +
             "\n        want: " + show(b));
}

inline void checkTrue(bool c, const char *e, const char *file, int line) {
  ++checks();
  if (!c)
    fail(file, line, std::string("expected true: ") + e);
}

inline int report(const char *suite) {
  if (failures() == 0)
    std::printf("PASS  %s (%d checks)\n", suite, checks());
  else
    std::printf("FAIL  %s (%d of %d checks failed)\n", suite, failures(),
                checks());
  return failures() == 0 ? 0 : 1;
}

} // namespace agpu_test

#define CASE(name) ::agpu_test::currentCase() = (name)
#define CHECK_EQ(a, b)                                                         \
  ::agpu_test::checkEq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK(c) ::agpu_test::checkTrue((c), #c, __FILE__, __LINE__)

#endif // AGPU_TEST_HARNESS_H
