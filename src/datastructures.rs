use bitvec::prelude::*;
use core::mem;
use num::{Integer, Num, NumCast, ToPrimitive};
use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::ops::{AddAssign, Div, Index, IndexMut};
use std::{fmt, iter};
use fnv::FnvHashMap;

#[derive(Clone, Default)]
pub struct BitSet {
    cardinality: usize,
    bit_vec: BitVec,
}

impl Ord for BitSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bit_vec.cmp(&other.bit_vec)
    }
}

impl PartialOrd for BitSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bit_vec.partial_cmp(&other.bit_vec)
    }
}

impl Debug for BitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let values: Vec<_> = self.iter().map(|i| i.to_string()).collect();
        write!(
            f,
            "BitSet {{ cardinality: {}, bit_vec: [{}]}}",
            self.cardinality,
            values.join(", "),
        )
    }
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.cardinality == other.cardinality && self.bit_vec == other.bit_vec
    }
}
impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bit_vec.hash(state)
    }
}

#[inline]
fn subset_helper(a: &[usize], b: &[usize]) -> bool {
    if a.len() > b.len() {
        a.iter()
            .zip(b.iter().chain(iter::repeat(&0usize)))
            .find(|(a, b)| (**a | **b) != **b)
            .is_none()
    } else {
        a.iter()
            .chain(iter::repeat(&0usize))
            .zip(b.iter())
            .find(|(a, b)| (**a | **b) != **b)
            .is_none()
    }
}

const fn block_size() -> usize {
    mem::size_of::<usize>() * 8
}

impl BitSet {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            cardinality: 0,
            bit_vec: bitvec![0; size],
        }
    }

    pub fn from_bitvec(bit_vec: BitVec) -> Self {
        let cardinality = bit_vec.iter().filter(|b| **b).count();
        Self {
            cardinality,
            bit_vec,
        }
    }

    pub fn from_slice<T: Div<Output = T> + ToPrimitive + AddAssign + Default + Copy + Display>(
        size: usize,
        slice: &[T],
    ) -> Self {
        let mut bit_vec: BitVec = bitvec![0; size];
        slice.iter().for_each(|i| {
            bit_vec.set(NumCast::from(*i).unwrap(), true);
        });
        let cardinality = slice.len();
        Self {
            cardinality,
            bit_vec,
        }
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.cardinality == 0
    }

    #[inline]
    pub fn full(&self) -> bool {
        self.cardinality == self.bit_vec.len()
    }

    #[inline]
    pub(crate) fn new_all_set(size: usize) -> Self {
        Self {
            cardinality: size,
            bit_vec: bitvec![1; size],
        }
    }

    #[inline]
    pub fn is_disjoint_with(&self, other: &BitSet) -> bool {
        self.bit_vec
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .find(|(x, y)| **x ^ **y != **x | **y)
            .is_none()
    }

    #[inline]
    pub fn intersects_with(&self, other: &BitSet) -> bool {
        !self.is_disjoint_with(other)
    }

    #[inline]
    pub fn is_subset_of(&self, other: &BitSet) -> bool {
        self.cardinality <= other.cardinality
            && subset_helper(self.bit_vec.as_slice(), other.as_slice())
    }

    #[inline]
    pub fn is_superset_of(&self, other: &BitSet) -> bool {
        other.is_subset_of(&self)
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.bit_vec.as_slice()
    }

    #[inline]
    pub fn as_bitslice(&self) -> &BitSlice {
        self.bit_vec.as_bitslice()
    }

    #[inline]
    pub fn as_bit_vec(&self) -> &BitVec {
        &self.bit_vec
    }

    #[inline]
    pub fn set_bit(&mut self, idx: usize) -> bool {
        if !*self.bit_vec.get(idx).unwrap() {
            self.bit_vec.set(idx, true);
            self.cardinality += 1;
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn unset_bit(&mut self, idx: usize) -> bool {
        if *self.bit_vec.get(idx).unwrap() {
            self.bit_vec.set(idx, false);
            self.cardinality -= 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    #[inline]
    pub fn or(&mut self, other: &BitSet) {
        if other.len() > self.bit_vec.len() {
            self.bit_vec.resize(other.len(), false);
        }
        for (x, y) in self
            .bit_vec
            .as_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x = *x | y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn resize(&mut self, size: usize) {
        let old_size = self.bit_vec.len();
        self.bit_vec.resize(size, false);
        if size < old_size {
            self.cardinality = self.bit_vec.count_ones();
        }
    }

    #[inline]
    pub fn and(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x = *x & y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn and_not(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x = *x & !y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn not(&mut self) {
        self.bit_vec
            .as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = !*x);
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn unset_all(&mut self) {
        self.bit_vec.as_mut_slice().iter_mut().for_each(|x| *x = 0);
        self.cardinality = 0;
    }

    #[inline]
    pub fn set_all(&mut self) {
        self.bit_vec
            .as_mut_slice()
            .iter_mut()
            .for_each(|x| *x = std::usize::MAX);
        self.cardinality = self.bit_vec.len();
    }

    #[inline]
    pub fn has_smaller(&mut self, other: &BitSet) -> Option<bool> {
        let self_idx = self.get_first_set();
        if self_idx.is_none() {
            return None;
        }
        let other_idx = other.get_first_set();
        if other_idx.is_none() {
            return None;
        }
        Some(self_idx < other_idx)
    }

    #[inline]
    pub fn get_first_set(&self) -> Option<usize> {
        if self.cardinality != 0 {
            return self.get_next_set(0);
            /*let mut self_idx: usize = 0;
            for x in self.bit_vec.as_slice() {
                if x == &0 {
                    self_idx += block_size();
                } else {
                    break;
                }
            }
            for i in self_idx..(self_idx + block_size()) {
                if self.bit_vec[i] {
                    return Some(i);
                }
            }*/
        }
        None
    }

    #[inline]
    pub fn get_next_set(&self, mut idx: usize) -> Option<usize> {
        if idx >= self.bit_vec.len() {
            return None;
        }
        let mut block_idx = idx / block_size();
        let mut word_idx = idx % block_size();
        let mut block = self.bit_vec.as_slice()[block_idx];
        block &= usize::MAX << word_idx;
        while block == 0usize {
            block_idx += 1;
            block = self.bit_vec.as_slice()[block_idx];
        }
        let v = block_idx * block_size() + block.trailing_zeros() as usize;
        return if v >= self.bit_vec.len() {
            None
        } else {
            Some(v)
        };

        /*while idx % block_size() != 0 && idx < self.bit_vec.len() {
            if self.bit_vec[idx] {
                return Some(idx);
            }
            idx += 1;
        }

        let block = idx / block_size();
        let slice = self.bit_vec.as_slice();
        if block >= slice.len() {
            return None;
        }
        for x in slice {
            if x == &0 {
                idx += block_size();
            } else {
                break;
            }
        }*/
        /*while idx < self.bit_vec.len() {
            if self.bit_vec[idx] {
                return Some(idx);
            }
            idx += 1;
        }
        None*/
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<u32> {
        let mut tmp = Vec::with_capacity(self.cardinality);
        for (i, _) in self
            .bit_vec
            .as_bitslice()
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
        {
            tmp.push(i as u32);
        }
        tmp
    }

    #[inline]
    pub fn at(&self, idx: usize) -> bool {
        self.bit_vec[idx]
    }

    #[inline]
    pub fn iter(&self) -> BitSetIterator {
        BitSetIterator {
            iter: self.bit_vec.as_slice().iter(),
            block: 0,
            idx: 0,
            size: self.bit_vec.len(),
        }
    }
}

pub struct BitSetIterator<'a> {
    iter: ::std::slice::Iter<'a, usize>,
    block: usize,
    idx: usize,
    size: usize,
}

impl<'a> Iterator for BitSetIterator<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.size {
            return None;
        }
        while self.block == 0 {
            self.block = if let Some(&i) = self.iter.next() {
                if i == 0 {
                    self.idx += block_size();
                    continue;
                } else {
                    self.idx = ((self.idx + block_size() - 1) / block_size()) * block_size();
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.block.trailing_zeros() as usize;
        self.block >>= offset;
        self.block >>= 1;
        self.idx += offset + 1;
        return Some(self.idx - 1);
    }
}

impl Index<usize> for BitSet {
    type Output = bool;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.bit_vec.index(index)
    }
}

pub struct PQ {
    heap: Vec<usize>,
    values: FnvHashMap<usize, i64>,
    indices: FnvHashMap<usize, usize>,
}

enum ChildType {
    First,
    Second,
}

impl PQ {
    pub fn new() -> Self {
        Self {
            heap: Vec::default(),
            values: FnvHashMap::default(),
            indices: FnvHashMap::default(),
        }
    }

    pub fn remove(&mut self, k: usize) {
        assert!(self.values.contains_key(&k));
        if !k == 0 {
            let v = *self.values.get(&self.heap[0]).unwrap() - 1;
            self.insert(k, v);
        }
        self.pop_min();
    }

    pub fn insert(&mut self, k: usize, v: i64) {
        if self.values.contains_key(&k) {
            self.update(k, v);
        } else {
            self.values.insert(k, v);
            self.indices.insert(k, self.heap.len());
            self.heap.push(k);
            if self.heap.len() > 1 {
                self.up(self.heap.len() - 1);
            }
        }
    }

    fn update(&mut self, k: usize, v: i64) {
        *self.values.get_mut(&k).unwrap() = v;
        self.up(*self.indices.get(&k).unwrap());
        self.down(*self.indices.get(&k).unwrap());
    }

    pub fn pop_min(&mut self) -> Option<(usize, i64)> {
        if !self.heap.is_empty() {
            let k = self.heap[0];
            let v = *self.values.get(&k).unwrap();
            self.heap[0] = *self.heap.last().unwrap();
            *self.indices.get_mut(&0).unwrap() = 0;
            self.heap.pop();
            if self.heap.len() > 1 {
                self.down(0);
            }
            return Some((k, v));
        }
        None
    }

    fn up(&mut self, idx: usize) {
        let mut current = idx;
        let value = self.heap[current];
        let mut parent = self.parent(current);

        while current > 0 && self.values.get(&value) < self.values.get(&self.heap[parent]) {
            let i = self.heap[parent];
            self.heap[current] = i;
            *self.indices.get_mut(&i).unwrap() = current;
            current = parent;
            parent = self.parent(parent);
        }
    }

    fn down(&mut self, idx: usize) {
        let mut current = idx;
        let value = self.heap[current];

        while self.heap.len() > self.child(current, ChildType::First) {
            let mut first = self.child(current, ChildType::First);
            let mut second = self.child(current, ChildType::Second);
            if second < self.heap.len() {
                let v1 = self.values.get(&self.heap[second]).unwrap();
                let v2 = self.values.get(&self.heap[first]).unwrap();
                if v1 < v2 {
                    first = second;
                }
            }
            if self.values.get(&self.heap[first]) < self.values.get(&value) {
                self.heap[current] = self.heap[first];
                *self.indices.get_mut(&self.heap[current]).unwrap() = current;
                current = first
            } else {
                break;
            }
        }
        self.heap[current] = value;
        *self.indices.get_mut(&value).unwrap() = current
    }

    fn parent(&self, idx: usize) -> usize {
        (idx-1)/2
    }

    fn child(&self, idx: usize, child_type: ChildType) -> usize {
        let off = match child_type {
            ChildType::First => 1,
            ChildType::Second => 1,
        };
        idx*2 + off
    }
}

#[cfg(test)]
mod tests {
    use crate::datastructures::BitSet;

    #[test]
    fn iter() {
        let mut bs = BitSet::new(129);

        let a: Vec<usize> = vec![12, 63, 128];
        for i in &a {
            bs.set_bit(*i);
        }

        let mut iter = bs.iter();
        let b: Vec<usize> = bs.iter().collect();
        assert_eq!(a, b);
        let mut c = Vec::new();
        let mut v = bs.get_next_set(0);
        while v.is_some() {
            c.push(v.unwrap());
            v = bs.get_next_set(v.unwrap() + 1);
        }
        assert_eq!(a, c);
    }
}
