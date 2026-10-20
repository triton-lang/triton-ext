// Containers.h - the container choices, in one place. std, so the library
// carries no external dependency; swapping them is editing this file.
#ifndef AGPU_CORE_CONTAINERS_H
#define AGPU_CORE_CONTAINERS_H

#include "agpu/core/SmallVector.h"

#include <map>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace agpu::core {

// `N` elements live inline: an AST vector is short and there are many.
template <class T, unsigned N = 8> using SmallVec = SmallVector<T, N>;

// Membership only: nothing iterates a `PtrSet`, so it need not be ordered.

using Str = std::string;
using StrRef = std::string_view;

template <class K, class V> using Map = std::map<K, V>;
template <class T> using PtrSet = std::unordered_set<T>;

} // namespace agpu::core

namespace agpu::msl {

using core::SmallVector;

template <class T, unsigned N = 8> using SmallVec = core::SmallVec<T, N>;

using Str = core::Str;
using StrRef = core::StrRef;

template <class K, class V> using Map = core::Map<K, V>;
template <class T> using PtrSet = core::PtrSet<T>;

} // namespace agpu::msl

#endif // AGPU_CORE_CONTAINERS_H
