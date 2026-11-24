// EnumBitset.h - a set of enumerators, one bit each.
#ifndef AGPU_ENUM_BITSET_H
#define AGPU_ENUM_BITSET_H

#include <cstdint>

namespace agpu {

// `E` needs a `Count` enumerator, which bounds the width.
template <class E, class Word = std::uint32_t> class EnumBitset {
public:
  void add(E e) { bits_ |= mask(e); }
  bool has(E e) const { return (bits_ & mask(e)) != 0; }
  bool any() const { return bits_ != 0; }

  void set(E e, bool value) {
    if (value)
      bits_ |= mask(e);
    else
      bits_ &= ~mask(e);
  }

private:
  static_assert(unsigned(E::Count) <= sizeof(Word) * 8,
                "the enum has outgrown the bitset word");

  static Word mask(E e) { return Word(1) << unsigned(e); }

  Word bits_ = 0;
};

} // namespace agpu

#endif // AGPU_ENUM_BITSET_H
