use crate::datastructures::BinaryQueue;
use crate::graph::Graph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
#[cfg(feature = "handle-ctrlc")]
use crate::signals::received_ctrl_c;
use crate::solver::AtomSolver;
use crate::tree_decomposition::TreeDecomposition;
use fnv::{FnvHashMap, FnvHashSet};
use rand::prelude::*;
use std::cmp::max;
use std::collections::HashMap;

pub struct MinFillDegreeSelector {
    inner: MinFillSelector,
}

impl From<HashMapGraph> for MinFillDegreeSelector {
    fn from(graph: HashMapGraph) -> Self {
        Self {
            inner: MinFillSelector::from(graph),
        }
    }
}

impl Selector for MinFillDegreeSelector {
    fn graph(&self) -> &HashMapGraph {
        self.inner.graph()
    }

    fn value(&self, v: usize) -> i64 {
        self.inner.value(v) << 32 + (self.inner.graph.degree(v) as i64)
    }

    fn eliminate_vertex(&mut self, v: usize) {
        self.inner.eliminate_vertex(v);
    }
}

pub struct MinDegreeSelector {
    graph: HashMapGraph,
}

impl From<HashMapGraph> for MinDegreeSelector {
    fn from(graph: HashMapGraph) -> Self {
        Self { graph }
    }
}

impl Selector for MinDegreeSelector {
    fn graph(&self) -> &HashMapGraph {
        &self.graph
    }

    fn value(&self, v: usize) -> i64 {
        self.graph.degree(v) as i64
    }

    fn eliminate_vertex(&mut self, v: usize) {
        self.graph.eliminate_vertex(v);
    }
}

pub struct MinFillSelector {
    graph: HashMapGraph,
    cache: FnvHashMap<usize, usize>,
}

impl From<HashMapGraph> for MinFillSelector {
    fn from(graph: HashMapGraph) -> Self {
        let mut cache = FnvHashMap::with_capacity_and_hasher(graph.order(), Default::default());
        for u in graph.vertices() {
            cache.insert(u, 0);
        }
        for u in graph.vertices() {
            for v in graph.vertices().filter(|v| u < *v && graph.has_edge(u, *v)) {
                graph
                    .neighborhood_set(u)
                    .iter()
                    .copied()
                    .filter(|x| v < *x && graph.has_edge(*x, v))
                    .for_each(|x| {
                        *cache.get_mut(&x).unwrap() += 1;
                        *cache.get_mut(&u).unwrap() += 1;
                        *cache.get_mut(&v).unwrap() += 1;
                    })
            }
        }
        Self { graph, cache }
    }
}

impl Selector for MinFillSelector {
    fn graph(&self) -> &HashMapGraph {
        &self.graph
    }

    fn value(&self, v: usize) -> i64 {
        self.fill_in_count(v) as i64
    }

    fn eliminate_vertex(&mut self, v: usize) {
        if self.fill_in_count(v) == 0 {
            self.eliminate_fill0(v);
        } else {
            let mut to_add: Vec<(usize, usize)> = vec![];
            for u in self.graph.neighborhood_set(v) {
                for w in self
                    .graph
                    .neighborhood_set(v)
                    .iter()
                    .filter(|w| u < *w && !self.graph.has_edge(*u, **w))
                {
                    to_add.push((*u, *w));
                }
            }
            for (u, w) in to_add {
                self.add_edge(u, w);
            }
            self.remove_vertex(v);
        }
    }
}

impl MinFillSelector {
    fn add_edge(&mut self, u: usize, v: usize) {
        self.graph.add_edge(u, v);
        for x in self.graph.neighborhood_set(u) {
            if self.graph.has_edge(*x, v) {
                *self.cache.get_mut(x).unwrap() += 1;
                *self.cache.get_mut(&u).unwrap() += 1;
                *self.cache.get_mut(&v).unwrap() += 1;
            }
        }
    }

    fn remove_vertex(&mut self, u: usize) {
        for v in self.graph.neighborhood_set(u).clone() {
            self.remove_edge(u, v);
        }
        self.graph.remove_vertex(u);
        self.cache.remove(&u);
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        self.graph.remove_edge(u, v);

        for x in self.graph.neighborhood_set(u) {
            if self.graph.has_edge(*x, v) {
                *self.cache.get_mut(&x).unwrap() -= 1;
                *self.cache.get_mut(&u).unwrap() -= 1;
                *self.cache.get_mut(&v).unwrap() -= 1;
            }
        }
    }

    fn eliminate_vertex_v2(&mut self, v: usize) {
        if self.fill_in_count(v) == 0 {
            self.eliminate_fill0(v);
        } else {
            let mut close_neighborhood = DistanceTwoNeighbors::new(&self.graph, v);
            let mut disjoint_neighborhood = DisjointNeighborhood::new(&self.graph, v);
            self.update_cache(v, close_neighborhood, disjoint_neighborhood);
        }
    }

    fn eliminate_fill0(&mut self, u: usize) {
        if self.graph.degree(u) > 1 {
            let delta = self.graph.degree(u) - 1;
            let graph = &self.graph;
            let cache = &mut self.cache;
            graph.neighborhood_set(u).iter().copied().for_each(|v| {
                *cache.get_mut(&v).unwrap() -= delta;
            });
        }
        self.graph.remove_vertex(u);
    }

    fn fill_in_count(&self, u: usize) -> usize {
        let deg = self.graph.degree(u);
        (deg * deg - deg) / 2 - self.cache.get(&u).unwrap()
    }

    fn update_cache(
        &mut self,
        v: usize,
        close_neighborhood: DistanceTwoNeighbors,
        disjoint_neighborhood: DisjointNeighborhood,
    ) {
        let mut tmp = FnvHashMap::default();
        for u in self.graph.neighborhood_set(v).iter().copied() {
            if !disjoint_neighborhood
                .unaffected_neighbors
                .get(&u)
                .unwrap()
                .is_empty()
            {
                let fill_in_count = if !disjoint_neighborhood
                    .neighbors_to_be
                    .get(&u)
                    .unwrap()
                    .is_empty()
                {
                    let init = sqr_no_self_loop(
                        disjoint_neighborhood.neighbors_to_be.get(&u).unwrap().len(),
                    ) + self.fill_in_count(u);
                    let mut fill_in_count = disjoint_neighborhood
                        .neighbors_to_be
                        .get(&u)
                        .unwrap()
                        .iter()
                        .fold(init, |init, w| {
                            let a = disjoint_neighborhood.unaffected_neighbors.get(&u).unwrap();
                            let b = disjoint_neighborhood.unaffected_neighbors.get(&w).unwrap();
                            let intersection_count = a.intersection(b).count();
                            init - intersection_count
                        });
                    if !disjoint_neighborhood
                        .existing_neighbors
                        .get(&u)
                        .unwrap()
                        .is_empty()
                    {
                        let mut count = sqr_no_self_loop(
                            disjoint_neighborhood
                                .existing_neighbors
                                .get(&u)
                                .unwrap()
                                .len(),
                        ) / 2;
                        for x in disjoint_neighborhood.existing_neighbors.get(&u).unwrap() {
                            for y in disjoint_neighborhood
                                .existing_neighbors
                                .get(&u)
                                .unwrap()
                                .iter()
                                .filter(|y| x < *y)
                            {
                                if self.graph.has_edge(*x, *y) {
                                    count -= 1;
                                }
                            }
                        }
                        fill_in_count -= count;
                    }
                    fill_in_count
                } else {
                    let init = self.fill_in_count(u)
                        - disjoint_neighborhood
                            .unaffected_neighbors
                            .get(&u)
                            .unwrap()
                            .len();
                    disjoint_neighborhood
                        .existing_neighbors
                        .get(&u)
                        .unwrap()
                        .iter()
                        .fold(init, |init, w| {
                            let a = disjoint_neighborhood.existing_neighbors.get(&u).unwrap();
                            let b = disjoint_neighborhood.neighbors_to_be.get(&w).unwrap();
                            let intersection_count = a.intersection(b).filter(|x| x > &w).count();
                            init - intersection_count
                        })
                };
                let v = sqr_no_self_loop(disjoint_neighborhood.len()) / 2;
                tmp.insert(u, v - fill_in_count);
            } else {
                let len = disjoint_neighborhood
                    .existing_neighbors
                    .get(&u)
                    .unwrap()
                    .len()
                    + disjoint_neighborhood.neighbors_to_be.get(&u).unwrap().len();
                let v = sqr_no_self_loop(len) / 2;
                tmp.insert(u, v);
            }
        }

        for u in close_neighborhood.0 {
            let intersection: Vec<usize> = self
                .graph
                .neighborhood_set(v)
                .intersection(self.graph.neighborhood_set(u))
                .copied()
                .collect();
            let mut increment_by = 0;
            for x in intersection.iter().copied() {
                increment_by += intersection
                    .iter()
                    .copied()
                    .filter(|y| x < *y && !self.graph.has_edge(x, *y))
                    .count();
            }
            tmp.insert(u, self.cache.get(&u).unwrap() + increment_by);
        }
        for (key, value) in tmp {
            *self.cache.get_mut(&key).unwrap() = value;
        }
        self.graph.eliminate_vertex(v);
    }
}

const fn sqr_no_self_loop(i: usize) -> usize {
    i * (i - 1)
}

struct DisjointNeighborhood {
    neighbors_to_be: FnvHashMap<usize, FnvHashSet<usize>>,
    existing_neighbors: FnvHashMap<usize, FnvHashSet<usize>>,
    unaffected_neighbors: FnvHashMap<usize, FnvHashSet<usize>>,
}

impl DisjointNeighborhood {
    fn len(&self) -> usize {
        self.existing_neighbors.len() + self.neighbors_to_be.len() + self.unaffected_neighbors.len()
    }

    fn new(graph: &HashMapGraph, v: usize) -> Self {
        let capacity = graph.neighborhood_set(v).len();
        let mut neighbors_to_be =
            FnvHashMap::with_capacity_and_hasher(capacity, Default::default());
        let mut existing_neighbors =
            FnvHashMap::with_capacity_and_hasher(capacity, Default::default());
        let mut unaffected_neighbors =
            FnvHashMap::with_capacity_and_hasher(capacity, Default::default());
        graph.neighborhood_set(v).iter().copied().for_each(|u| {
            neighbors_to_be.insert(u, FnvHashSet::default());
            existing_neighbors.insert(u, FnvHashSet::default());
            unaffected_neighbors.insert(u, FnvHashSet::default());
        });

        for u in graph.neighborhood_set(v).iter().copied() {
            for w in graph
                .neighborhood_set(u)
                .iter()
                .copied()
                .filter(|w| u != *w)
            {
                if graph.neighborhood_set(v).contains(&w) {
                    existing_neighbors.get_mut(&u).unwrap().insert(w);
                } else {
                    unaffected_neighbors.get_mut(&u).unwrap().insert(w);
                }
            }
            for u2 in graph.neighborhood_set(v).iter().copied() {
                if u2 != u && !graph.has_edge(u, u2) {
                    neighbors_to_be.get_mut(&u).unwrap().insert(u2);
                }
            }
        }

        Self {
            neighbors_to_be,
            existing_neighbors,
            unaffected_neighbors,
        }
    }
}

struct DistanceTwoNeighbors(FnvHashSet<usize>);

impl DistanceTwoNeighbors {
    fn new(graph: &HashMapGraph, v: usize) -> Self {
        DistanceTwoNeighbors(
            graph
                .neighborhood_set(v)
                .iter()
                .flat_map(|u| {
                    graph
                        .neighborhood_set(*u)
                        .iter()
                        .filter(move |w| u < *w && !graph.neighborhood_set(v).contains(w))
                })
                .copied()
                .collect(),
        )
    }
}

pub trait Selector: From<HashMapGraph> {
    fn graph(&self) -> &HashMapGraph;
    fn value(&self, v: usize) -> i64;
    fn eliminate_vertex(&mut self, v: usize);
}

pub type MinFillDecomposer = HeuristicEliminationDecomposer<MinFillSelector>;
pub type MinDegreeDecomposer = HeuristicEliminationDecomposer<MinDegreeSelector>;
pub type MinFillDegree = HeuristicEliminationDecomposer<MinFillDegreeSelector>;

pub struct HeuristicEliminationDecomposer<S: Selector> {
    selector: S,
    lowerbound: usize,
    upperbound: usize,
}

impl<S: Selector> AtomSolver for HeuristicEliminationDecomposer<S> {
    fn with_graph(graph: &HashMapGraph) -> Self {
        Self {
            selector: S::from(graph.clone()),
            lowerbound: 0,
            upperbound: if graph.order() > 0 {
                graph.order() - 1
            } else {
                0
            },
        }
    }

    fn with_bounds(graph: &HashMapGraph, lowerbound: usize, upperbound: usize) -> Self {
        Self {
            selector: S::from(graph.clone()),
            lowerbound,
            upperbound,
        }
    }

    fn compute(self) -> Result<TreeDecomposition, ()> {
        let mut tree_decomposition = TreeDecomposition::default();
        if self.selector.graph().order() <= self.lowerbound + 1 {
            tree_decomposition.add_bag(self.selector.graph().vertices().collect());
            return Ok(tree_decomposition);
        }

        let mut max_bag = 2;
        let upperbound = self.upperbound;
        let lowerbound = self.lowerbound;
        let mut selector = self.selector;
        let mut pq = BinaryQueue::new();

        let mut bags: FnvHashMap<usize, FnvHashSet<usize>> = FnvHashMap::default();
        let mut eliminated_at: FnvHashMap<usize, usize> = FnvHashMap::default();

        for v in selector.graph().vertices() {
            pq.insert(v, selector.value(v))
        }

        let mut stack: Vec<usize> = vec![];
        while let Some((u, _)) = pq.pop_min() {
            if selector.graph().order() <= max_bag || selector.graph().order() <= lowerbound + 1 {
                break;
            }

            #[cfg(feature = "handle-ctrlc")]
            if crate::signals::received_ctrl_c() {
                // unknown lowerbound
                break;
            }

            if selector.graph().degree(u) > upperbound {
                return Err(());
            }

            let mut nb: FnvHashSet<usize> = selector.graph().neighborhood(u).collect();
            max_bag = max(max_bag, nb.len() + 1);
            stack.push(u);
            bags.insert(u, nb.clone());
            eliminated_at.insert(u, stack.len() - 1);
            selector.eliminate_vertex(u);

            let tmp: Vec<_> = nb
                .iter()
                .copied()
                .filter(|u| selector.graph().neighborhood_set(*u).len() < nb.len())
                .collect();

            // eliminate directly, as these are subsets of the current bag
            /*for u in tmp {
                selector.eliminate_vertex(u);
                nb.remove(&u);
                pq.remove(u);
            }*/
            for u in nb {
                pq.insert(u, selector.value(u));
            }
        }

        if selector.graph().order() > 0 {
            let mut rest: FnvHashSet<usize> = selector.graph().vertices().collect();
            let u = rest.iter().next().copied().unwrap();
            rest.remove(&u);
            bags.insert(u, rest);
            stack.push(u);
            eliminated_at.insert(u, stack.len() - 1);
        }

        for v in stack.iter().rev() {
            let mut nb = bags.remove(v).unwrap();
            let old_bag_id = match tree_decomposition
                .bags
                .iter()
                .find(|old_bag| old_bag.vertex_set.is_superset(&nb))
            {
                Some(old_bag) => Some(old_bag.id),
                None => None,
            };
            match old_bag_id {
                Some(old_bag_id) => {
                    nb.insert(*v);
                    let id = tree_decomposition.add_bag(nb);
                    tree_decomposition.add_edge(old_bag_id, id);
                }
                None => {
                    nb.insert(*v);
                    tree_decomposition.add_bag(nb);
                }
            }
        }
        Ok(tree_decomposition)
    }
}

pub fn heuristic_elimination_decompose<S: Selector>(graph: HashMapGraph) -> TreeDecomposition {
    let mut tree_decomposition = TreeDecomposition::default();
    if graph.order() <= 2 {
        tree_decomposition.add_bag(graph.vertices().collect());
        return tree_decomposition;
    }
    let mut max_bag = 2;
    let mut selector: S = S::from(graph);
    let mut pq = BinaryQueue::new();

    let mut bags: FnvHashMap<usize, FnvHashSet<usize>> = FnvHashMap::default();
    let mut eliminated_at: FnvHashMap<usize, usize> = FnvHashMap::default();

    for v in selector.graph().vertices() {
        pq.insert(v, selector.value(v))
    }

    let mut stack: Vec<usize> = vec![];
    while let Some((u, _)) = pq.pop_min() {
        if selector.graph().order() <= max_bag {
            break;
        }

        #[cfg(feature = "handle-ctrlc")]
        if received_ctrl_c() {
            // simply adds all remaining vertices into a single bag
            break;
        }

        let mut nb: FnvHashSet<usize> = selector.graph().neighborhood(u).collect();
        max_bag = max(max_bag, nb.len() + 1);
        stack.push(u);
        bags.insert(u, nb.clone());
        eliminated_at.insert(u, stack.len() - 1);
        selector.eliminate_vertex(u);

        let tmp: Vec<_> = nb
            .iter()
            .copied()
            .filter(|u| selector.graph().neighborhood_set(*u).len() < nb.len())
            .collect();

        // eliminate directly, as these are subsets of the current bag
        /*for u in tmp {
            selector.eliminate_vertex(u);
            nb.remove(&u);
            pq.remove(u);
        }*/
        for u in nb {
            pq.insert(u, selector.value(u));
        }
    }

    if selector.graph().order() > 0 {
        let mut rest: FnvHashSet<usize> = selector.graph().vertices().collect();
        let u = rest.iter().next().copied().unwrap();
        rest.remove(&u);
        bags.insert(u, rest);
        stack.push(u);
        eliminated_at.insert(u, stack.len() - 1);
    }

    for v in stack.iter().rev() {
        let mut nb = bags.remove(v).unwrap();
        let old_bag_id = match tree_decomposition
            .bags
            .iter()
            .find(|old_bag| old_bag.vertex_set.is_superset(&nb))
        {
            Some(old_bag) => Some(old_bag.id),
            None => None,
        };
        match old_bag_id {
            Some(old_bag_id) => {
                nb.insert(*v);
                let id = tree_decomposition.add_bag(nb);
                tree_decomposition.add_edge(old_bag_id, id);
            }
            None => {
                nb.insert(*v);
                tree_decomposition.add_bag(nb);
            }
        }
    }
    tree_decomposition
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::graph::HashMapGraph;
    use crate::graph::MutableGraph;
    use crate::heuristic_elimination_order::{DistanceTwoNeighbors, MinFillSelector, Selector};
    use crate::io::PaceReader;
    use fnv::FnvHashMap;
    use std::convert::TryFrom;
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::BufReader;
    use std::path::PathBuf;

    #[test]
    fn initial() {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("tests");
        d.push("td-validate");
        d.push("test");
        d.push("tw-solver-bugs");
        d.push("dimacs_anna.gr");

        let f = File::open(d).unwrap();
        let mut reader = BufReader::new(f);
        let reader = PaceReader(reader);
        let graph = HashMapGraph::try_from(reader).unwrap();

        let vertices: Vec<_> = graph.vertices().collect();
        let fc1: FnvHashMap<_, _> = vertices
            .iter()
            .map(|v| (*v, graph.fill_in_count(*v) as i64))
            .collect();

        let selector = MinFillSelector::from(graph);
        let fc2: FnvHashMap<_, _> = vertices.iter().map(|v| (*v, selector.value(*v))).collect();

        for v in fc1.keys() {
            assert_eq!(fc1.get(v).unwrap(), fc2.get(v).unwrap());
        }
    }

    #[test]
    fn eliminate() {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("tests");
        d.push("td-validate");
        d.push("test");
        d.push("tw-solver-bugs");
        d.push("dimacs_anna.gr");

        let f = File::open(d).unwrap();
        let mut reader = BufReader::new(f);
        let reader = PaceReader(reader);
        let mut graph = HashMapGraph::try_from(reader).unwrap();
        let mut selector = MinFillSelector::from(graph.clone());

        let mut vertices: Vec<_> = graph.vertices().collect();

        while let Some(v) = vertices.pop() {
            println!("Eliminating :{}", v);
            graph.eliminate_vertex(v);
            selector.eliminate_vertex(v);

            let mut a: Vec<_> = selector.graph.vertices().collect();
            a.sort();
            let mut b: Vec<_> = graph.vertices().collect();
            b.sort();
            println!("Checking equality!");
            assert_eq!(a, b);
            println!("This worked!");
            let fc1: FnvHashMap<_, _> = vertices
                .iter()
                .map(|v| (*v, graph.fill_in_count(*v) as i64))
                .collect();
            let fc2: FnvHashMap<_, _> = vertices.iter().map(|v| (*v, selector.value(*v))).collect();

            for v in fc1.keys() {
                assert_eq!(fc1.get(v).unwrap(), fc2.get(v).unwrap());
            }
        }
    }

    #[test]
    fn distance_two() {
        let mut graph = HashMapGraph::new();
        // direct neighbors of 0
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(0, 4);

        // distance two neighbors of 0
        graph.add_edge(1, 5);
        graph.add_edge(2, 6);
        graph.add_edge(3, 7);
        graph.add_edge(4, 8);

        // to be ignored
        graph.add_edge(3, 4);
        graph.add_edge(8, 9);
        graph.add_edge(5, 9);

        let distance_two_neighbors = DistanceTwoNeighbors::new(&graph, 0);
        assert_eq!(distance_two_neighbors.0.len(), 4);
        assert!(distance_two_neighbors.0.contains(&5));
        assert!(distance_two_neighbors.0.contains(&6));
        assert!(distance_two_neighbors.0.contains(&7));
        assert!(distance_two_neighbors.0.contains(&8));
    }
}
