use crate::datastructures::BitSet;
use crate::graph::graph::Graph;
use crate::graph::mutable_graph::MutableGraph;
use crate::graph::tree_decomposition::TreeDecomposition;
use crate::heuristic_elimination_order::{heuristic_elimination_decompose, MinFillSelector};
use crate::util::EliminationOrder;
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use rand::prelude::{SliceRandom, ThreadRng};
use rand::Rng;
use std::cmp::{max, min, Ordering};
use std::collections::{HashSet, VecDeque};
use std::convert::TryFrom;
use std::io::BufRead;
use std::iter::FromIterator;

#[derive(Clone, Debug)]
pub struct HashMapGraph {
    data: FnvHashMap<usize, FnvHashSet<usize>>,
}

#[derive(Clone, Debug, Copy)]
struct MissingEdge {
    u: usize,
    v: usize,
}

#[derive(Clone, Debug)]
pub struct MinorSafeResult {
    pub separator: FnvHashSet<usize>,
    pub belongs_to: (usize, usize),
    pub tree_decomposition: TreeDecomposition,
}

impl HashMapGraph {
    pub fn has_vertex(&self, u: usize) -> bool {
        self.data.contains_key(&u)
    }
    pub fn neighborhood_set(&self, u: usize) -> &FnvHashSet<usize> {
        &self.data.get(&u).unwrap()
    }
    pub fn dfs(&self, u: usize) -> HashMapGraphDfs {
        assert!(self.data.contains_key(&u));
        let mut visited = BitSet::new(self.data.len());
        visited.set_bit(u);
        HashMapGraphDfs {
            graph: self,
            stack: vec![u],
            visited,
        }
    }

    pub fn find_cut_vertex(&self) -> Option<FnvHashSet<usize>> {
        if self.data.is_empty() {
            return None;
        }
        let u = self.data.keys().copied().next().unwrap();
        let mut c = 0;
        let mut l: FnvHashMap<usize, usize> = FnvHashMap::default();
        let mut d: FnvHashMap<usize, usize> = FnvHashMap::default();
        let ignore: FnvHashSet<usize> = FnvHashSet::default();
        if let Some(v) = self.articulation_point_helper(u, u, &mut c, &mut l, &mut d, &ignore) {
            let mut set: FnvHashSet<usize> = FnvHashSet::default();
            set.insert(v);
            return Some(set);
        }
        None
    }

    pub fn find_safe_bi_connected_separator(&self) -> Option<FnvHashSet<usize>> {
        if self.data.is_empty() {
            return None;
        }
        for guess in self.data.keys().copied() {
            let mut c = 0;
            let mut l: FnvHashMap<usize, usize> = FnvHashMap::default();
            let mut d: FnvHashMap<usize, usize> = FnvHashMap::default();
            let mut ignore: FnvHashSet<usize> = FnvHashSet::default();
            ignore.insert(guess);
            let u = self.data.keys().copied().find(|x| *x != guess)?;
            if let Some(p) = self.articulation_point_helper(u, u, &mut c, &mut l, &mut d, &ignore) {
                let mut set: FnvHashSet<usize> = FnvHashSet::default();
                set.insert(guess);
                set.insert(p);
                return Some(set);
            }
        }
        None
    }

    pub fn connected_components(&self) -> Vec<FnvHashSet<usize>> {
        self.separate(&FnvHashSet::default())
    }

    pub fn find_safe_tri_connected_separator(&self) -> Option<FnvHashSet<usize>> {
        if self.data.is_empty() {
            return None;
        }
        for f1 in self.data.keys().copied() {
            for f2 in self
                .data
                .keys()
                .copied()
                .filter(|forbidden2| *forbidden2 > f1)
            {
                let u = self.data.keys().copied().find(|x| *x != f1 && *x != f2)?;
                let mut c = 0;
                let mut l: FnvHashMap<usize, usize> = FnvHashMap::default();
                let mut d: FnvHashMap<usize, usize> = FnvHashMap::default();
                let mut ignore: FnvHashSet<usize> = FnvHashSet::default();
                ignore.insert(f1);
                ignore.insert(f2);
                if let Some(p) =
                    self.articulation_point_helper(u, u, &mut c, &mut l, &mut d, &ignore)
                {
                    // found separator size 3, check for safety (3 is safe when (almost) clique)
                    if self.data.get(&p).unwrap().contains(&f1)
                        || self.data.get(&p).unwrap().contains(&f2)
                        || self.data.get(&f1).unwrap().contains(&f2)
                    {
                        let mut set: FnvHashSet<usize> = FnvHashSet::default();
                        set.insert(f1);
                        set.insert(f2);
                        set.insert(p);
                        return Some(set);
                    }
                    // check if graph contains vertex v with N(v) = {p, f1, f2}
                    if self
                        .data
                        .iter()
                        .find(|(_, v)| {
                            v.len() == 3 && v.contains(&p) && v.contains(&f1) && v.contains(&f2)
                        })
                        .is_none()
                    {
                        let mut set: FnvHashSet<usize> = FnvHashSet::default();
                        set.insert(f1);
                        set.insert(f2);
                        set.insert(p);
                        return Some(set);
                    }

                    // separates into > 2 components OR 2 components with each component containing more than 1 vertex
                    let mut separator = FnvHashSet::default();
                    separator.insert(f1);
                    separator.insert(f2);
                    separator.insert(p);
                    let components = self.separate(&separator);
                    if components.len() > 2
                        || components.len() == 2
                            && components.iter().find(|c| c.len() <= 1).is_none()
                    {
                        let mut set: FnvHashSet<usize> = FnvHashSet::default();
                        set.insert(f1);
                        set.insert(f2);
                        set.insert(p);
                        return Some(set);
                    }
                }
            }
        }
        None
    }

    pub fn find_minor_safe_separator(
        &self,
        tree_decomposition: Option<TreeDecomposition>,
    ) -> Option<MinorSafeResult> {
        return match tree_decomposition {
            None => {
                let mut new_td = heuristic_elimination_decompose::<MinFillSelector>(self.clone());
                new_td.flatten();
                self.minor_safe_helper(new_td, 25)
            }
            Some(working_td) => {
                match self.minor_safe_helper(working_td, 25) {
                    None => {
                        let mut new_td = heuristic_elimination_decompose::<MinFillSelector>(self.clone());
                        new_td.flatten();
                        self.minor_safe_helper(new_td, 25)
                    }
                    Some(result) => {
                        Some(result)
                    }
                }
            }
        }
        /*let mut working_td = match tree_decomposition {
            None => {
                let mut td = heuristic_elimination_decompose::<MinFillSelector>(self.clone());
                td.flatten();
                td
            },
            Some(tree_decomposition) => tree_decomposition,
        };
        self.minor_safe_helper(working_td, 25)*/
    }

    fn minor_safe_helper(&self, td: TreeDecomposition, max_tries: u32) -> Option<MinorSafeResult> {
        for first_bag in td.bags.iter() {
            for idx in first_bag.neighbors.iter().copied().filter(|id| {
                *id >= first_bag.id && !td.bags[*id].vertex_set.eq(&first_bag.vertex_set)
            }) {
                let second_bag = &td.bags[idx].vertex_set;
                let candidate: FnvHashSet<usize> = first_bag
                    .vertex_set
                    .intersection(second_bag)
                    .copied()
                    .collect();
                if self.is_minor_safe(&candidate, max_tries) {
                    return Some(MinorSafeResult {
                        separator: candidate,
                        belongs_to: (min(first_bag.id, idx), max(first_bag.id, idx)),
                        tree_decomposition: td,
                    });
                }
            }
        }
        None
    }

    fn is_minor_safe(&self, separator: &FnvHashSet<usize>, max_tries: u32) -> bool {
        let components = self.separate(separator);
        if components.len() < 2 {
            return false;
        }
        let mut rng = ThreadRng::default();

        for component in components.iter() {
            let rest: FnvHashSet<usize> = self
                .data
                .keys()
                .copied()
                .filter(|k| !component.contains(k))
                .collect();

            let mut tmp = self.vertex_induced(&rest);
            let mut missing_edges: Vec<_> = Vec::new();
            for u in separator.iter().copied() {
                for v in separator
                    .iter()
                    .copied()
                    .filter(|v| u < *v && !tmp.data.get(v).unwrap().contains(&u))
                {
                    missing_edges.push(MissingEdge { u, v });
                }
            }
            let mut is_minor = false;
            'outer: for _ in 0..max_tries {
                missing_edges.shuffle(&mut rng);
                for missing_edge in missing_edges.iter() {
                    if tmp
                        .data
                        .get(&missing_edge.v)
                        .unwrap()
                        .contains(&missing_edge.u)
                    {
                        continue;
                    }
                    let u = missing_edge.u;
                    let v = missing_edge.v;
                    let common_neighbors: Vec<_> = tmp
                        .data
                        .get(&u)
                        .unwrap()
                        .iter()
                        .copied()
                        .filter(|x| !separator.contains(x) && tmp.data.get(&v).unwrap().contains(x))
                        .collect();
                    if common_neighbors.len() > 0 {
                        let idx: usize = rng.gen_range(0, common_neighbors.len());
                        let contractor = common_neighbors[idx];
                        if rng.gen_bool(0.5) {
                            tmp.contract(contractor, v);
                        } else {
                            tmp.contract(contractor, u);
                        }
                    } else {
                        let mut queue = VecDeque::new();
                        queue.push_back(v);
                        let mut pre: FnvHashMap<_, _> =
                            separator.iter().copied().map(|v| (v, v)).collect();
                        pre.remove(&u);
                        while !pre.contains_key(&u) && !queue.is_empty() {
                            let x = queue.pop_front().unwrap();
                            for k in tmp.data.get(&x).unwrap().iter().copied() {
                                if pre.contains_key(&k) {
                                    continue;
                                }
                                pre.insert(k, x);
                                queue.push_back(k);
                                if k == u {
                                    break;
                                }
                            }
                        }
                        let value = pre.get(&u);
                        if value.is_some() {
                            let mut current = *value.unwrap();
                            while current != v {
                                tmp.contract(current, u);
                                current = *pre.get(&current).unwrap();
                            }
                        } else {
                            continue 'outer;
                        }
                    }
                }
                is_minor = true;
                break;
            }
            if !is_minor {
                return false;
            }
        }
        true
    }

    pub fn vertex_induced(&self, vertices: &FnvHashSet<usize>) -> Self {
        let data: FnvHashMap<usize, FnvHashSet<usize>> = self
            .data
            .iter()
            .filter(|(vertex, _)| vertices.contains(vertex))
            .map(|(vertex, neighborhood)| {
                (
                    *vertex,
                    neighborhood
                        .iter()
                        .copied()
                        .filter(|x| vertices.contains(x))
                        .collect(),
                )
            })
            .collect();
        Self { data }
    }

    fn is_minimal_separator(&self, separator: &FnvHashSet<usize>) -> bool {
        let components = self.separate(separator);
        components.len() > 1
            && components
                .iter()
                .find(|component| {
                    separator
                        .iter()
                        .find(|v| {
                            self.data
                                .get(v)
                                .unwrap()
                                .iter()
                                .find(|u| component.contains(u))
                                .is_none()
                        })
                        .is_some()
                })
                .is_none()
    }

    pub fn find_almost_clique_minimal_separator(&self) -> Option<FnvHashSet<usize>> {
        for v in self.data.keys().copied() {
            let mut ignore = FnvHashSet::default();
            ignore.insert(v);
            if let Some(mut separator) = self.clique_minimal_separator_helper(&ignore) {
                separator.insert(v);
                if self.is_minimal_separator(&separator) {
                    return Some(separator);
                }
            }
        }
        None
    }

    pub fn find_clique_minimal_separator(&self) -> Option<FnvHashSet<usize>> {
        let empty = FnvHashSet::default();
        self.clique_minimal_separator_helper(&empty)
    }

    fn closed_neighborhood_helper(
        &self,
        component: &FnvHashSet<usize>,
        separator: &FnvHashSet<usize>,
    ) -> FnvHashSet<usize> {
        if component.is_empty() || separator.is_empty() || self.data.is_empty() {
            return FnvHashSet::default();
        }
        let mut closed_neighborhood: FnvHashSet<_> = FnvHashSet::default();

        let mut stack: Vec<_> = Vec::with_capacity(component.len());
        let mut visited: FnvHashSet<_> = FnvHashSet::with_capacity_and_hasher(
            component.len() + separator.len(),
            Default::default(),
        );
        let first = component.iter().copied().next().unwrap();
        stack.push(first);
        visited.insert(first);

        while let Some(v) = stack.pop() {
            for x in self.data.get(&v).unwrap().iter() {
                if visited.contains(x) {
                    continue;
                }
                if separator.contains(x) {
                    closed_neighborhood.insert(*x);
                    visited.insert(*x);
                }
                stack.push(*x);
            }
        }
        closed_neighborhood
    }

    fn clique_minimal_separator_helper(
        &self,
        ignore: &FnvHashSet<usize>,
    ) -> Option<FnvHashSet<usize>> {
        // Algorithm in 'An Introduction to Clique Minimal SeparatorDecomposition'
        let mut working_graph = self.clone();
        for v in ignore.iter().copied() {
            working_graph.remove_vertex(v);
        }
        let mut h = self.clone();

        let mut alpha = Vec::with_capacity(self.data.len());
        let mut generators: FnvHashSet<usize> = FnvHashSet::default();
        let mut labels: FnvHashMap<usize, usize> =
            working_graph.data.keys().copied().map(|k| (k, 0)).collect();

        let mut s: Option<usize> = None;

        let n = self.order() - ignore.len();
        for _ in 0..n {
            let x = *working_graph
                .data
                .keys()
                .max_by(|a, b| labels.get(a).cmp(&labels.get(b)))
                .unwrap();
            let mut y = working_graph.data.get(&x).unwrap().clone();
            if s.is_some() && labels.get(&x).unwrap() <= &s.unwrap() {
                generators.insert(x);
            }
            s = Some(*labels.get(&x).unwrap());

            let mut reached: FnvHashSet<_> = FnvHashSet::default();
            reached.insert(x);
            let mut reach: FnvHashMap<usize, FnvHashSet<usize>> =
                (0..n).map(|i| (i, FnvHashSet::default())).collect();
            working_graph.data.get(&x).unwrap().iter().for_each(|i| {
                reached.insert(*i);
                reach.get_mut(labels.get(i).unwrap()).unwrap().insert(*i);
            });

            for j in 0..n {
                while !reach.get(&j).unwrap().is_empty() {
                    let r = *reach.get(&j).unwrap().iter().next().unwrap();
                    reach.get_mut(&j).unwrap().remove(&r);
                    for z in working_graph.data.get(&r).unwrap().iter() {
                        if reached.contains(z) {
                            continue;
                        }
                        reached.insert(*z);
                        if labels.get(z).unwrap() > &j {
                            y.insert(*z);
                            reach.get_mut(labels.get(z).unwrap()).unwrap().insert(*z);
                        } else {
                            reach.get_mut(&j).unwrap().insert(*z);
                        }
                    }
                }
            }

            for y in y.iter() {
                h.add_edge(x, *y);
                *labels.get_mut(y).unwrap() += 1;
            }

            alpha.push(x);
            working_graph.remove_vertex(x);
        }

        for x in alpha.iter().copied().rev() {
            if generators.contains(&x) {
                let s = h.data.get(&x).unwrap();
                let mut is_clique = true;
                for v in s.iter() {
                    for u in s.iter().filter(|u| *u > v) {
                        if !self.data.get(&v).unwrap().contains(&u) {
                            is_clique = false;
                            break;
                        }
                    }
                    if !is_clique {
                        break;
                    }
                }
                if is_clique {
                    return Some(s.clone());
                }
            }
            h.remove_vertex(x);
        }
        None
    }

    pub fn separate(&self, separator: &FnvHashSet<usize>) -> Vec<FnvHashSet<usize>> {
        let mut components: Vec<FnvHashSet<_>> = Vec::with_capacity(2);

        let mut stack: Vec<_> = Vec::with_capacity(self.data.len());
        let mut visited = FnvHashSet::with_capacity_and_hasher(self.data.len(), Default::default());
        let mut iter = self.data.keys().copied();
        while let Some(u) = iter.next() {
            if separator.contains(&u) || visited.contains(&u) {
                continue;
            }
            stack.push(u);
            visited.insert(u);
            let mut component: FnvHashSet<_> = FnvHashSet::default();
            component.insert(u);
            while let Some(v) = stack.pop() {
                for x in self.data.get(&v).unwrap().iter() {
                    if component.contains(x) || separator.contains(x) {
                        continue;
                    }
                    stack.push(*x);
                    component.insert(*x);
                    visited.insert(*x);
                }
            }
            components.push(component);
        }
        components
    }

    pub fn vertex_induced_subgraph(&self, vertex_set: &FnvHashSet<usize>) -> Self {
        let mut subgraph = HashMapGraph::with_capacity(vertex_set.len());
        for u in vertex_set.iter() {
            for v in vertex_set
                .iter()
                .filter(|v| u < *v && self.data.get(*v).unwrap().contains(u))
            {
                subgraph.add_edge(*u, *v);
            }
        }
        subgraph
    }

    fn articulation_point_helper(
        &self,
        u: usize,
        v: usize,
        c: &mut usize,
        l: &mut FnvHashMap<usize, usize>,
        d: &mut FnvHashMap<usize, usize>,
        ignore: &FnvHashSet<usize>,
    ) -> Option<usize> {
        *c += 1;
        let mut children = 0;
        l.insert(v, *c);
        d.insert(v, *c);
        let nb = self.data.get(&v).unwrap();
        for w in nb.iter().filter(|w| !ignore.contains(*w)) {
            if !d.contains_key(w) {
                children += 1;
                if let Some(cut) = self.articulation_point_helper(v, *w, c, l, d, ignore) {
                    return Some(cut);
                }
                let a = *l.get(&v).unwrap();
                let b = *l.get(&w).unwrap();
                l.insert(v, min(a, b));
                if l.get(&w).unwrap() >= d.get(&v).unwrap() && u != v {
                    return Some(v);
                }
            } else if *w != u && d.get(&w).unwrap() < d.get(&v).unwrap() {
                let a = *l.get(&v).unwrap();
                let b = *d.get(&w).unwrap();
                l.insert(v, min(a, b));
            }
        }
        if u == v && children > 1 {
            return Some(v);
        }
        None
    }
}

pub struct HashMapGraphDfs<'a> {
    graph: &'a HashMapGraph,
    stack: Vec<usize>,
    visited: BitSet,
}

impl<'a> Iterator for HashMapGraphDfs<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.stack.pop()?;
        for c in self.graph.data.get(&current).unwrap().iter().copied() {
            if !self.visited[c] {
                self.stack.push(c);
                self.visited.set_bit(c);
            }
        }
        Some(current)
    }
}

impl MutableGraph for HashMapGraph {
    fn add_vertex(&mut self, u: usize) {
        if !self.data.contains_key(&u) {
            self.data.insert(u, FnvHashSet::default());
        }
    }

    fn add_vertex_with_capacity(&mut self, u: usize, capacity: usize) {
        if !self.data.contains_key(&u) {
            self.data.insert(
                u,
                FnvHashSet::with_capacity_and_hasher(capacity, Default::default()),
            );
        }
    }

    fn remove_vertex(&mut self, u: usize) {
        if let Some(neighbors) = self.data.remove(&u) {
            for i in neighbors.iter() {
                self.data.get_mut(i).unwrap().remove(&u);
            }
        }
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v);
        let first = self.data.entry(u).or_insert(FnvHashSet::default());
        first.insert(v);
        let second = self.data.entry(v).or_insert(FnvHashSet::default());
        second.insert(u);
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v);
        if let Some(x) = self.data.get_mut(&u) {
            x.remove(&v);
        }
        if let Some(x) = self.data.get_mut(&v) {
            x.remove(&u);
        }
    }

    fn eliminate_vertex(&mut self, u: usize) {
        assert!(self.data.contains_key(&u));
        let nb = self.data.remove(&u).unwrap();
        for i in &nb {
            self.data.get_mut(&i).unwrap().remove(&u);
        }
        for i in &nb {
            for j in &nb {
                if i < j {
                    self.data.get_mut(&i).unwrap().insert(*j);
                    self.data.get_mut(&j).unwrap().insert(*i);
                }
            }
        }
    }

    fn contract(&mut self, u: usize, v: usize) {
        assert_ne!(u, v);
        assert!(self.data.contains_key(&u));
        assert!(self.data.contains_key(&v));

        let nb = self.data.remove(&u).unwrap();

        for vertex in nb {
            if vertex == v {
                continue;
            }
            let a = self.data.get_mut(&vertex).unwrap();
            a.remove(&u);
            a.insert(v);
            let b = self.data.get_mut(&v).unwrap();
            b.insert(vertex);
        }
        self.data.get_mut(&v).unwrap().remove(&u);
    }

    fn new() -> Self {
        HashMapGraph {
            data: FnvHashMap::default(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        HashMapGraph {
            data: FnvHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }
}

impl Graph for HashMapGraph {
    fn degree(&self, u: usize) -> usize {
        assert!(self.data.contains_key(&u));
        self.data.get(&u).unwrap().len()
    }

    fn order(&self) -> usize {
        self.data.len()
    }

    fn is_clique(&self, vertices: &[usize]) -> bool {
        for (i, v) in vertices.iter().enumerate() {
            assert!(self.data.contains_key(&v));
            for u in vertices.iter().skip(i + 1) {
                assert!(self.data.contains_key(&u));
                if !self.data.get(&v).unwrap().contains(&u)
                    || !self.data.get(&u).unwrap().contains(&v)
                {
                    return false;
                }
            }
        }
        true
    }

    fn is_neighborhood_clique(&self, u: usize) -> bool {
        let nb = self.data.get(&u).unwrap();
        self.is_clique(Vec::from_iter(nb.iter().copied()).as_slice())
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.data.get(&u).unwrap().contains(&v)
    }

    fn is_simplicial(&self, u: usize) -> bool {
        return self.is_neighborhood_clique(u);
    }

    fn is_almost_simplicial(&self, u: usize) -> bool {
        let mut check: Option<FnvHashSet<_>> = None;
        let nb = self.data.get(&u).unwrap();
        for v in nb.iter().copied() {
            for w in nb
                .iter()
                .copied()
                .filter(|w| v < *w && !self.has_edge(v, *w))
            {
                if check.is_some() {
                    check.as_mut().unwrap().retain(|x| *x == v || *x == w);
                    if check.as_ref().unwrap().is_empty() {
                        return false;
                    }
                } else {
                    check = Some([v, w].iter().copied().collect());
                }
            }
        }
        (check.is_none() && check.unwrap().len() == 1)
    }

    fn vertices(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        let keys = self.data.keys().copied();
        Box::new(keys)
    }

    fn neighborhood(&self, u: usize) -> Box<dyn Iterator<Item = usize> + '_> {
        Box::new(self.data.get(&u).unwrap().iter().copied())
    }

    fn fill_in_count(&self, u: usize) -> usize {
        let mut count = 0;
        for x in self.neighborhood_set(u) {
            for y in self.neighborhood_set(u) {
                if x < y && !self.has_edge(*x, *y) {
                    count += 1;
                }
            }
        }
        count
    }

    fn min_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize> {
        self.data.keys().copied().min_by(cmp)
    }

    fn max_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize> {
        self.data.keys().copied().max_by(cmp)
    }

    fn min_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize> {
        self.data.get(&u).unwrap().iter().copied().min_by(cmp)
    }

    fn max_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize> {
        self.data.get(&u).unwrap().iter().copied().max_by(cmp)
    }
}

impl HashMapGraph {
    pub fn from_graph<G: Graph>(graph: &G) -> Self {
        let data = graph
            .vertices()
            .map(|v| (v, graph.neighborhood(v).collect()))
            .collect();
        HashMapGraph { data }
    }

    pub fn inplace_min_fill(&self) -> EliminationOrder {
        let mut data: Vec<usize> = Vec::with_capacity(self.order());
        let mut width = 0;
        let mut added: Vec<bool> = vec![false; self.order()];
        let mut min_fills: Vec<usize> = vec![0; self.order()];

        for v in self.data.keys() {
            min_fills[*v] = self.fill_in_count(*v);
        }

        let mut stack: Vec<usize> = Vec::with_capacity(self.order());
        while data.len() < self.order() {
            let v = {
                let mut min = 0;
                for i in self.data.keys() {
                    if !added[*i] && min_fills[*i] < min_fills[min] {
                        min = *i;
                    }
                }
                min
            };
            let degree_v = {
                let mut degree = 0;
                stack.push(v);
                let mut visited = vec![false; self.order()];
                while let Some(x) = stack.pop() {
                    for u in self.data.get(&x).unwrap() {
                        if !visited[*u] {
                            visited[*u] = true;
                            match data.iter().find(|x| *x == u) {
                                Some(u) => {
                                    stack.push(*u);
                                }
                                None => {
                                    if *u != v {
                                        degree += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                degree
            };
            width = max(width, degree_v);

            data.push(v);
            added[v] = true;
            for u in self.data.get(&v).unwrap().iter().filter(|u| !added[**u]) {
                for v in self
                    .data
                    .get(&v)
                    .unwrap()
                    .iter()
                    .filter(|v| !added[**v] && *v != u)
                {
                    if !self.has_edge(*u, *v) {
                        min_fills[*v] -= 1;
                    }
                }
            }
        }

        EliminationOrder::new(data, width)
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::graph::Graph;
    use crate::graph::hash_map_graph::HashMapGraph;
    use crate::graph::mutable_graph::MutableGraph;

    #[test]
    fn test_order() {
        let mut graph = HashMapGraph::new();
        assert_eq!(graph.order(), 0);

        graph.add_vertex(0);
        graph.add_vertex(0);
        assert_eq!(graph.order(), 1);
        graph.remove_vertex(0);
        assert_eq!(graph.order(), 0);

        graph.add_vertex_with_capacity(0, 0);
        graph.add_vertex_with_capacity(0, 0);
        assert_eq!(graph.order(), 1);
    }

    #[test]
    fn test_degree() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 1);

        assert_eq!(graph.order(), 2);

        graph.add_edge(0, 1);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 1);

        assert_eq!(graph.order(), 2);

        graph.remove_edge(0, 1);

        assert_eq!(graph.order(), 0);
    }

    #[test]
    fn cut_vertex() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(0, 2);

        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(3, 5);

        graph.add_edge(1, 4);
        graph.add_edge(1, 5);

        let separator = graph.find_cut_vertex();
        assert!(separator.is_some());
        let separator = separator.unwrap();
        assert_eq!(separator.len(), 1);
        assert!(separator.contains(&1));
        assert!(graph.separate(&separator).len() > 1);
    }

    #[test]
    fn no_cut_vertex() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(0, 2);

        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(3, 5);

        graph.add_edge(0, 3);
        graph.add_edge(1, 4);

        let cut_vertex = graph.find_cut_vertex();
        assert!(cut_vertex.is_none());
    }

    #[test]
    fn safe_bi_connected_separator() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 5);

        graph.add_edge(1, 2);
        graph.add_edge(1, 5);

        graph.add_edge(2, 3);
        graph.add_edge(2, 4);
        graph.add_edge(2, 5);

        graph.add_edge(3, 4);
        graph.add_edge(3, 5);

        graph.add_edge(4, 5);

        let separator = graph.find_safe_bi_connected_separator();
        assert!(separator.is_some());
        let separator = separator.unwrap();
        assert_eq!(separator.len(), 2);
        assert!(separator.contains(&2));
        assert!(separator.contains(&5));
        assert!(graph.separate(&separator).len() > 1);
    }

    #[test]
    fn no_safe_bi_connected_separator() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.add_edge(0, 5);

        graph.add_edge(1, 2);
        graph.add_edge(1, 5);

        graph.add_edge(2, 3);
        graph.add_edge(2, 4);
        graph.add_edge(2, 5);

        graph.add_edge(3, 4);
        graph.add_edge(3, 5);

        graph.add_edge(4, 5);

        let separator = graph.find_safe_bi_connected_separator();
        assert!(separator.is_none());
    }

    #[test]
    fn safe_tri_connected_separator() {
        let mut graph = HashMapGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.add_edge(0, 5);

        graph.add_edge(1, 2);
        graph.add_edge(1, 5);

        graph.add_edge(2, 3);
        graph.add_edge(2, 4);
        graph.add_edge(2, 5);

        graph.add_edge(3, 4);
        graph.add_edge(3, 5);

        graph.add_edge(4, 5);

        let separator = graph.find_safe_tri_connected_separator();
        assert!(separator.is_some());
        let separator = separator.unwrap();
        assert_eq!(separator.len(), 3);

        assert!(graph.separate(&separator).len() > 1);
    }

    #[test]
    fn safe_clique_separator() {
        let mut graph = HashMapGraph::new();
        // clique separator
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);

        // component 1
        graph.add_edge(10, 11);
        graph.add_edge(10, 12);
        graph.add_edge(10, 13);
        graph.add_edge(11, 12);
        graph.add_edge(11, 13);
        graph.add_edge(12, 13);

        // connect component 1
        graph.add_edge(0, 10);
        graph.add_edge(1, 11);
        graph.add_edge(2, 12);
        graph.add_edge(3, 13);

        // component 2
        graph.add_edge(100, 101);
        graph.add_edge(100, 102);
        graph.add_edge(100, 103);
        graph.add_edge(101, 102);
        graph.add_edge(101, 103);
        graph.add_edge(102, 103);

        // connect component 2
        graph.add_edge(0, 100);
        graph.add_edge(1, 101);
        graph.add_edge(2, 102);
        graph.add_edge(3, 103);

        let separator = graph.find_clique_minimal_separator();

        assert!(separator.is_some());
        let separator = separator.unwrap();
        assert!(separator.contains(&0));
        assert!(separator.contains(&1));
        assert!(separator.contains(&2));
        assert!(separator.contains(&3));
        let components = graph.separate(&separator);
        assert!(components.len() > 1);
    }

    #[test]
    fn safe_almost_clique_separator() {
        let mut graph = HashMapGraph::new();
        // almost clique separator
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        // missing:
        // graph.add_edge(2, 3);

        // component 1
        graph.add_edge(10, 11);
        graph.add_edge(10, 12);
        graph.add_edge(10, 13);
        graph.add_edge(11, 12);
        graph.add_edge(11, 13);
        graph.add_edge(12, 13);

        // connect component 1
        graph.add_edge(0, 10);
        graph.add_edge(1, 11);
        graph.add_edge(2, 12);
        graph.add_edge(3, 13);

        // component 2
        graph.add_edge(100, 101);
        graph.add_edge(100, 102);
        graph.add_edge(100, 103);
        graph.add_edge(101, 102);
        graph.add_edge(101, 103);
        graph.add_edge(102, 103);

        // connect component 2
        graph.add_edge(0, 100);
        graph.add_edge(1, 101);
        graph.add_edge(2, 102);
        graph.add_edge(3, 103);

        let separator = graph.find_almost_clique_minimal_separator();

        assert!(separator.is_some());
        let separator = separator.unwrap();
        assert!(separator.contains(&0));
        assert!(separator.contains(&1));
        assert!(separator.contains(&2));
        assert!(separator.contains(&3));
        let components = graph.separate(&separator);
        assert!(components.len() > 1);
    }
}
