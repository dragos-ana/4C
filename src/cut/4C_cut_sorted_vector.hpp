// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_SORTED_VECTOR_HPP
#define FOUR_C_CUT_SORTED_VECTOR_HPP

#include "4C_config.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  /// sorted vector that emulates a set (and is supposed to be much more efficient)
  template <class K, bool b_no_duplicates = true, class Pr = std::less<K>,
      class A = std::allocator<K>>
  class SortedVector
  {
   public:
    using Myt_ = SortedVector<K, b_no_duplicates, Pr, A>;
    using Cont = std::vector<K, A>;
    using allocator_type = typename Cont::allocator_type;
    using size_type = typename Cont::size_type;
    using difference_type = typename Cont::difference_type;
    using reference = typename Cont::reference;
    using const_reference = typename Cont::const_reference;
    using value_type = typename Cont::value_type;
    using key_type = K;
    using iterator = typename Cont::iterator;
    using const_iterator = typename Cont::const_iterator;
    using key_compare = Pr;
    using value_compare = Pr;

    using const_reverse_iterator = typename Cont::const_reverse_iterator;
    using reverse_iterator = typename Cont::reverse_iterator;

    using Pairii_ = std::pair<iterator, iterator>;
    using Paircc_ = std::pair<const_iterator, const_iterator>;
    using Pairib_ = std::pair<iterator, bool>;

    explicit SortedVector(const Pr& pred = Pr(), const A& al = A()) : less_(pred), vec_(al) {}

    template <class It>
    SortedVector(It first, It beyond, const Pr& pred = Pr(), const A& al = A())
        : less_(pred), vec_(first, beyond, al)
    {
      stable_sort();
    }

    SortedVector(const Myt_& x) : less_(x.less_), vec_(x.vec_) {}

    Myt_& operator=(const Myt_& x)
    {
      vec_.operator=(x.vec_);
      less_ = x.less_;
      return *this;
    }

    Myt_& operator=(const Cont& x)
    {
      vec_.operator=(x);
      sort();
      return *this;
    }

    void reserve(size_type n) { vec_.reserve(n); }
    iterator begin() { return vec_.begin(); }
    const_iterator begin() const { return vec_.begin(); }
    iterator end() { return vec_.end(); }
    const_iterator end() const { return vec_.end(); }
    reverse_iterator rbegin() { return vec_.rbegin(); }
    const_reverse_iterator rbegin() const { return vec_.rbegin(); }

    reverse_iterator rend() { return vec_.rend(); }
    const_reverse_iterator rend() const { return vec_.rend(); }

    size_type size() const { return vec_.size(); }
    size_type max_size() const { return vec_.max_size(); }
    bool empty() const { return vec_.empty(); }
    A get_allocator() const { return vec_.get_allocator(); }
    const_reference at(size_type p) const { return vec_.at(p); }
    reference at(size_type p) { return vec_.at(p); }
    const_reference operator[](size_type p) const { return vec_.operator[](p); }

    reference operator[](size_type p) { return vec_.operator[](p); }
    reference front() { return vec_.front(); }
    const_reference front() const { return vec_.front(); }
    reference back() { return vec_.back(); }
    const_reference back() const { return vec_.back(); }
    void pop_back() { vec_.pop_back(); }

    void assign(const_iterator first, const_iterator beyond) { vec_.assign(first, beyond); }

    void assign(size_type n, const K& x = K()) { vec_.assign(n, x); }

    /// insert members
    Pairib_ insert(const value_type& x)
    {
      if (b_no_duplicates)
      {
        iterator p = lower_bound(x);
        if (p == end() or less_(x, *p))
        {
          return Pairib_(insert_impl(p, x), true);
        }
        else
        {
          return Pairib_(p, false);
        }
      }
      else
      {
        iterator p = upper_bound(x);
        return Pairib_(insert_impl(p, x), true);
      }
    }

    iterator insert(iterator it, const value_type& x)  // it is the hint
    {
      return insert(x).first;
    }

    template <class It>
    void insert(It first, It beyond)
    {
      size_type n = std::distance(first, beyond);
      reserve(size() + n);
      for (; first != beyond; ++first)
      {
        insert(*first);
      }
    }

    iterator ierase(iterator p) { return vec_.erase(p); }

    iterator erase(iterator first, iterator beyond) { return vec_.erase(first, beyond); }

    size_type erase(const K& key)
    {
      Pairii_ begEnd = equal_range(key);
      size_type n = std::distance(begEnd.first, begEnd.second);
      erase(begEnd.first, begEnd.second);
      return n;
    }

    void clear() { vec_.clear(); }

    void swap(Myt_& x)
    {
      vec_.swap(x.vec_);
      std::swap(less_, x.less_);
    }

    friend void swap(Myt_& x, Myt_& Y_) { x.swap(Y_); }

    key_compare key_comp() const { return less_; }
    value_compare value_comp() const { return (key_comp()); }

    iterator find(const K& k)
    {
      iterator p = lower_bound(k);
      return (p == end() or less_(k, *p)) ? end() : p;
    }

    const_iterator find(const K& k) const
    {
      const_iterator p = lower_bound(k);
      return (p == end() or less_(k, *p)) ? end() : p;
    }

    size_type count(const K& k) const
    {
      Paircc_ Ans_ = equal_range(k);
      size_type n = std::distance(Ans_.first, Ans_.second);
      return n;
    }

    iterator lower_bound(const K& k) { return std::lower_bound(begin(), end(), k, less_); }

    const_iterator lower_bound(const K& k) const
    {
      return std::lower_bound(begin(), end(), k, less_);
    }

    iterator upper_bound(const K& k) { return std::upper_bound(begin(), end(), k, less_); }

    const_iterator upper_bound(const K& k) const
    {
      return std::upper_bound(begin(), end(), k, less_);
    }

    Pairii_ equal_range(const K& k) { return std::equal_range(begin(), end(), k, less_); }

    Paircc_ equal_range(const K& k) const { return std::equal_range(begin(), end(), k, less_); }

    /// functions for use with direct std::vector-access
    Cont& get_container() { return vec_; }

    /// restore sorted order after low level access
    void sort()
    {
      std::sort(vec_.begin(), vec_.end(), less_);
      if (b_no_duplicates)
      {
        vec_.erase(unique(), vec_.end());
      }
    }

    /// restore sorted order after low level access
    void stable_sort()
    {
      std::stable_sort(vec_.begin(), vec_.end(), less_);
      if (b_no_duplicates)
      {
        erase(unique(), end());
      }
    }

    bool operator==(const SortedVector<K, b_no_duplicates, Pr, A>& other) const
    {
      return eq(other);
    }

    bool operator<(const SortedVector<K, b_no_duplicates, Pr, A>& other) const { return lt(other); }

   protected:
    iterator unique()
    {
      iterator front_ = vec_.begin(), out_ = vec_.end(), end_ = vec_.end();
      bool bCopy_ = false;
      for (iterator prev_; (prev_ = front_) != end_ and ++front_ != end_;)
      {
        if (less_(*prev_, *front_))
        {
          if (bCopy_)
          {
            *out_ = *front_;
            ++out_;
          }
        }
        else
        {
          if (not bCopy_)
          {
            out_ = front_;
            bCopy_ = true;
          }
        }
      }
      return out_;
    }

    bool eq(const Myt_& x) const
    {
      return (size() == x.size() and std::equal(begin(), end(), x.begin()));
    }

    bool lt(const Myt_& x) const
    {
      return (std::lexicographical_compare(begin(), end(), x.begin(), x.end()));
    }

    iterator insert_impl(iterator p, const value_type& x)
    {
      iterator i = vec_.insert(p, x);
      return i;
    }

    bool leq(const K& ty0, const K& ty1) { return not less_(ty1, ty0); }

    bool geq(const K& ty0, const K& ty1) { return not less_(ty0, ty1); }

    bool lt(const K& ty0, const K& ty1) { return less_(ty0, ty1); }

    bool gt(const K& ty0, const K& ty1) { return less_(ty1, ty0); }

    key_compare less_;
    Cont vec_;
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
