// SmallVector.h - a vector holding its first `N` elements inline.
#ifndef AGPU_CORE_SMALL_VECTOR_H
#define AGPU_CORE_SMALL_VECTOR_H

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <utility>

namespace agpu::core {

template <class T, unsigned N> class SmallVector {
public:
  using value_type = T;
  using size_type = std::size_t;
  using iterator = T *;
  using const_iterator = const T *;

  SmallVector() = default;

  SmallVector(std::initializer_list<T> init) {
    reserve(init.size());
    for (const T &v : init)
      push_back(v);
  }

  explicit SmallVector(size_type n) { resize(n); }

  SmallVector(size_type n, const T &v) {
    reserve(n);
    for (size_type i = 0; i < n; ++i)
      pushRaw(v);
  }

  template <class It> SmallVector(It first, It last) {
    for (; first != last; ++first)
      push_back(*first);
  }

  SmallVector(const SmallVector &o) {
    reserve(o.size_);
    for (size_type i = 0; i < o.size_; ++i)
      pushRaw(o.data_[i]);
  }

  SmallVector(SmallVector &&o) noexcept {
    if (o.heap()) {
      data_ = o.data_;
      cap_ = o.cap_;
      size_ = o.size_;
      o.data_ = o.inline_();
      o.cap_ = N;
      o.size_ = 0;
      return;
    }
    reserve(o.size_);
    for (size_type i = 0; i < o.size_; ++i)
      pushRaw(std::move(o.data_[i]));
    o.clear();
  }

  SmallVector &operator=(const SmallVector &o) {
    if (this == &o)
      return *this;
    clear();
    reserve(o.size_);
    for (size_type i = 0; i < o.size_; ++i)
      pushRaw(o.data_[i]);
    return *this;
  }

  SmallVector &operator=(SmallVector &&o) noexcept {
    if (this == &o)
      return *this;
    clear();
    if (o.heap()) {
      release();
      data_ = o.data_;
      cap_ = o.cap_;
      size_ = o.size_;
      o.data_ = o.inline_();
      o.cap_ = N;
      o.size_ = 0;
      return *this;
    }
    reserve(o.size_);
    for (size_type i = 0; i < o.size_; ++i)
      pushRaw(std::move(o.data_[i]));
    o.clear();
    return *this;
  }

  ~SmallVector() {
    clear();
    release();
  }

  bool empty() const { return size_ == 0; }
  size_type size() const { return size_; }
  size_type capacity() const { return cap_; }

  T *data() { return data_; }
  const T *data() const { return data_; }

  iterator begin() { return data_; }
  iterator end() { return data_ + size_; }
  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }

  T &operator[](size_type i) { return data_[i]; }
  const T &operator[](size_type i) const { return data_[i]; }

  T &front() { return data_[0]; }
  const T &front() const { return data_[0]; }
  T &back() { return data_[size_ - 1]; }
  const T &back() const { return data_[size_ - 1]; }

  void push_back(const T &v) {
    grow(size_ + 1);
    pushRaw(v);
  }

  void push_back(T &&v) {
    grow(size_ + 1);
    pushRaw(std::move(v));
  }

  template <class... Args> T &emplace_back(Args &&...args) {
    grow(size_ + 1);
    ::new (static_cast<void *>(data_ + size_)) T(std::forward<Args>(args)...);
    return data_[size_++];
  }

  void pop_back() { data_[--size_].~T(); }

  void clear() {
    for (size_type i = size_; i > 0; --i)
      data_[i - 1].~T();
    size_ = 0;
  }

  void reserve(size_type want) {
    if (want > cap_)
      reallocate(want);
  }

  template <class It> void assign(It first, It last) {
    clear();
    for (; first != last; ++first)
      push_back(*first);
  }

  void swap(SmallVector &o) {
    SmallVector tmp(std::move(o));
    o = std::move(*this);
    *this = std::move(tmp);
  }

  void resize(size_type want) {
    if (want < size_) {
      for (size_type i = size_; i > want; --i)
        data_[i - 1].~T();
      size_ = want;
      return;
    }
    grow(want);
    while (size_ < want)
      ::new (static_cast<void *>(data_ + size_++)) T();
  }

  iterator insert(const_iterator at, const T &v) {
    const size_type i = size_type(at - data_);
    grow(size_ + 1);
    if (i == size_) {
      pushRaw(v);
      return data_ + i;
    }
    ::new (static_cast<void *>(data_ + size_)) T(std::move(data_[size_ - 1]));
    ++size_;
    for (size_type j = size_ - 1; j > i + 1; --j)
      data_[j - 1] = std::move(data_[j - 2]);
    data_[i] = v;
    return data_ + i;
  }

  iterator erase(const_iterator at) {
    const size_type i = size_type(at - data_);
    for (size_type j = i; j + 1 < size_; ++j)
      data_[j] = std::move(data_[j + 1]);
    pop_back();
    return data_ + i;
  }

  iterator erase(const_iterator first, const_iterator last) {
    const size_type i = size_type(first - data_);
    const size_type n = size_type(last - first);
    for (size_type j = i; j + n < size_; ++j)
      data_[j] = std::move(data_[j + n]);
    for (size_type k = 0; k < n; ++k)
      pop_back();
    return data_ + i;
  }

private:
  T *inline_() { return reinterpret_cast<T *>(buf_); }
  bool heap() const { return data_ != reinterpret_cast<const T *>(buf_); }
  void release() {
    if (heap())
      ::operator delete(static_cast<void *>(data_));
  }

  void pushRaw(const T &v) {
    ::new (static_cast<void *>(data_ + size_)) T(v);
    ++size_;
  }
  void pushRaw(T &&v) {
    ::new (static_cast<void *>(data_ + size_)) T(std::move(v));
    ++size_;
  }

  void grow(size_type want) {
    if (want <= cap_)
      return;
    reallocate(std::max(want, cap_ * 2));
  }

  void reallocate(size_type want) {
    T *fresh = static_cast<T *>(::operator new(want * sizeof(T)));
    for (size_type i = 0; i < size_; ++i) {
      ::new (static_cast<void *>(fresh + i)) T(std::move(data_[i]));
      data_[i].~T();
    }
    release();
    data_ = fresh;
    cap_ = want;
  }

  alignas(T) unsigned char buf_[N * sizeof(T)];
  T *data_ = inline_();
  size_type size_ = 0;
  size_type cap_ = N;
};

template <class T, unsigned N>
bool operator==(const SmallVector<T, N> &a, const SmallVector<T, N> &b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <class T, unsigned N>
bool operator!=(const SmallVector<T, N> &a, const SmallVector<T, N> &b) {
  return !(a == b);
}

} // namespace agpu::core

#endif // AGPU_CORE_SMALL_VECTOR_H
