use crate::datastructures::BinaryQueue;
use crate::graph::BaseGraph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::solver::{AtomSolver, Bounds, ComputationResult};
use crate::tree_decomposition::TreeDecomposition;
use fxhash::{FxHashMap, FxHashSet};
#[cfg(feature = "log")]
use log::info;
use std::cmp::max;

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
        (self.inner.value(v) << 32) + (self.inner.graph.degree(v) as i64)
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
    cache: FxHashMap<usize, usize>,
}

impl From<HashMapGraph> for MinFillSelector {
    fn from(graph: HashMapGraph) -> Self {
        let mut cache = FxHashMap::with_capacity_and_hasher(graph.order(), Default::default());
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
        self.eliminate_with_info(v);
    }
}

impl MinFillSelector {
    pub(crate) fn add_edge(&mut self, u: usize, v: usize) {
        self.graph.add_edge(u, v);
        for x in self.graph.neighborhood_set(u) {
            if self.graph.has_edge(*x, v) {
                *self.cache.get_mut(x).unwrap() += 1;
                *self.cache.get_mut(&u).unwrap() += 1;
                *self.cache.get_mut(&v).unwrap() += 1;
            }
        }
    }

    pub(crate) fn add_edges<'a, T: IntoIterator<Item = &'a (usize, usize)>>(&mut self, iter: T) {
        for (u, v) in iter {
            self.add_edge(*u, *v);
        }
    }

    pub(crate) fn remove_edges<'a, T: IntoIterator<Item = &'a (usize, usize)>>(&mut self, iter: T) {
        for (u, v) in iter {
            self.remove_edge(*u, *v);
        }
    }

    fn remove_vertex(&mut self, u: usize) {
        for v in self.graph.neighborhood_set(u).clone() {
            self.remove_edge(u, v);
        }
        self.graph.remove_vertex(u);
        self.cache.remove(&u);
    }

    pub(crate) fn remove_edge(&mut self, u: usize, v: usize) {
        self.graph.remove_edge(u, v);

        for x in self.graph.neighborhood_set(u) {
            if self.graph.has_edge(*x, v) {
                *self.cache.get_mut(&x).unwrap() -= 1;
                *self.cache.get_mut(&u).unwrap() -= 1;
                *self.cache.get_mut(&v).unwrap() -= 1;
            }
        }
    }

    fn eliminate_fill0(&mut self, u: usize) -> FillInfo {
        if self.graph.degree(u) > 1 {
            let delta = self.graph.degree(u) - 1;
            let graph = &self.graph;
            let cache = &mut self.cache;
            graph.neighborhood_set(u).iter().copied().for_each(|v| {
                *cache.get_mut(&v).unwrap() -= delta;
            });
        }
        let neighborhood = self.graph.neighborhood_set(u).clone();
        self.graph.remove_vertex(u);
        FillInfo {
            added_edges: vec![],
            neighborhood,
            eliminated_vertex: u,
        }
    }

    pub(crate) fn fill_in_count(&self, u: usize) -> usize {
        let deg = self.graph.degree(u);
        (deg * deg - deg) / 2 - self.cache.get(&u).unwrap()
    }

    pub(crate) fn eliminate_with_info(&mut self, v: usize) -> FillInfo {
        if self.fill_in_count(v) == 0 {
            self.eliminate_fill0(v)
        } else {
            let neighborhood = self.graph.neighborhood_set(v).clone();
            let mut added_edges: Vec<(usize, usize)> = vec![];
            for u in self.graph.neighborhood_set(v) {
                for w in self
                    .graph
                    .neighborhood_set(v)
                    .iter()
                    .filter(|w| u < *w && !self.graph.has_edge(*u, **w))
                {
                    added_edges.push((*u, *w));
                }
            }
            for (u, w) in &added_edges {
                self.add_edge(*u, *w);
            }
            self.remove_vertex(v);
            FillInfo {
                added_edges,
                neighborhood,
                eliminated_vertex: v,
            }
        }
    }

    pub(crate) fn undo_elimination(&mut self, fill_info: FillInfo) {
        for (u, v) in fill_info.added_edges {
            self.add_edge(u, v);
        }
        self.graph.add_vertex(fill_info.eliminated_vertex);
        self.cache
            .insert(fill_info.eliminated_vertex, Default::default());
        for u in fill_info.neighborhood {
            self.add_edge(u, fill_info.eliminated_vertex);
        }
    }
}

pub(crate) struct FillInfo {
    added_edges: Vec<(usize, usize)>,
    neighborhood: FxHashSet<usize>,
    eliminated_vertex: usize,
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

pub struct EliminationOrderDecomposer {
    graph: HashMapGraph,
    permutation: Vec<usize>,
    eliminated_in_bag: FxHashMap<usize, usize>,
}

impl EliminationOrderDecomposer {
    pub fn new(graph: HashMapGraph, permutation: Vec<usize>) -> Self {
        Self {
            graph,
            permutation,
            eliminated_in_bag: Default::default(),
        }
    }

    pub fn compute(mut self) -> PermutationDecompositionResult {
        let mut tree_decomposition: TreeDecomposition = Default::default();
        for u in self.permutation.iter().copied() {
            let nb: FxHashSet<usize> = self.graph.neighborhood(u).collect();
            let mut bag = nb.clone();
            bag.insert(u);
            self.eliminated_in_bag
                .insert(u, tree_decomposition.add_bag(bag));
            self.graph.eliminate_vertex(u);
        }
        permutation_td_connect_helper(
            &mut tree_decomposition,
            &self.permutation,
            &self.eliminated_in_bag,
        );

        PermutationDecompositionResult {
            permutation: self.permutation,
            tree_decomposition,
            eliminated_in_bag: self.eliminated_in_bag,
        }
    }
}

fn permutation_td_connect_helper(
    tree_decomposition: &mut TreeDecomposition,
    permutation: &[usize],
    eliminated_in_bag: &FxHashMap<usize, usize>,
) {
    for v in permutation.iter() {
        let bag_id = eliminated_in_bag.get(v).unwrap();

        let mut neighbor: Option<usize> = None;
        for u in tree_decomposition.bags[*bag_id].vertex_set.iter() {
            let candidate_neighbor = &tree_decomposition.bags[*eliminated_in_bag
                .get(u)
                .unwrap_or(&(tree_decomposition.bags.len() - 1))];
            if candidate_neighbor.id == *bag_id {
                continue;
            }
            neighbor = {
                if neighbor.is_none() || neighbor.unwrap() > candidate_neighbor.id {
                    Some(candidate_neighbor.id)
                } else {
                    neighbor
                }
            };
        }
        if let Some(neighbor) = neighbor {
            tree_decomposition.add_edge(*bag_id, neighbor);
        }
    }
}

impl<S: Selector> HeuristicEliminationDecomposer<S> {
    pub fn compute_order_and_decomposition(self) -> Option<PermutationDecompositionResult> {
        #[cfg(feature = "log")]
        info!("computing heuristic elimination td");
        let mut selector = self.selector;
        let mut permutation: Vec<usize> = vec![];
        let upperbound = self.upperbound;
        let lowerbound = self.lowerbound;
        let mut tree_decomposition = TreeDecomposition::default();
        let mut eliminated_in_bag: FxHashMap<usize, usize> = FxHashMap::default();

        if selector.graph().order() > self.lowerbound + 1 {
            let mut max_bag = 2;
            let mut pq = BinaryQueue::new();
            for v in selector.graph().vertices() {
                pq.insert(v, selector.value(v))
            }
            while let Some((u, _)) = pq.pop_min() {
                if selector.graph().order() <= max_bag || selector.graph().order() <= lowerbound + 1
                {
                    break;
                }

                #[cfg(feature = "handle-ctrlc")]
                if crate::signals::received_ctrl_c() {
                    // unknown lowerbound
                    #[cfg(feature = "log")]
                    info!("breaking heuristic elimination td due to ctrl+c");
                    break;
                }

                #[cfg(feature = "cli")]
                if crate::timeout::timeout() {
                    // unknown lowerbound
                    #[cfg(feature = "log")]
                    info!("breaking heuristic elimination td due to timeout!");
                    break;
                }

                if selector.graph().degree(u) > upperbound {
                    return None;
                }

                let nb: FxHashSet<usize> = selector.graph().neighborhood(u).collect();
                max_bag = max(max_bag, nb.len() + 1);
                permutation.push(u);
                let mut bag = nb.clone();
                bag.insert(u);
                eliminated_in_bag.insert(u, tree_decomposition.add_bag(bag));
                selector.eliminate_vertex(u);
                for u in nb {
                    pq.insert(u, selector.value(u));
                }
            }
        }

        // remaining vertices, arbitrary order
        while selector.graph().order() > 0 {
            let u = selector.graph().vertices().next().unwrap();
            let nb: FxHashSet<usize> = selector.graph().neighborhood(u).collect();
            permutation.push(u);
            let mut bag = nb.clone();
            bag.insert(u);
            eliminated_in_bag.insert(u, tree_decomposition.add_bag(bag));
            selector.eliminate_vertex(u);
        }

        permutation_td_connect_helper(&mut tree_decomposition, &permutation, &eliminated_in_bag);

        Some(PermutationDecompositionResult {
            permutation,
            tree_decomposition,
            eliminated_in_bag,
        })
    }
}

pub struct PermutationDecompositionResult {
    pub permutation: Vec<usize>,
    pub tree_decomposition: TreeDecomposition,
    pub eliminated_in_bag: FxHashMap<usize, usize>,
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

    fn compute(self) -> ComputationResult {
        let bounds = Bounds {
            lowerbound: self.lowerbound,
            upperbound: self.upperbound,
        };
        match self.compute_order_and_decomposition() {
            None => ComputationResult::Bounds(bounds),
            Some(result) => ComputationResult::ComputedTreeDecomposition(result.tree_decomposition),
        }
    }
}
/*
pub fn heuristic_elimination_decompose<S: Selector>(graph: HashMapGraph) -> TreeDecomposition {
    let mut tree_decomposition = TreeDecomposition::default();
    if graph.order() <= 2 {
        tree_decomposition.add_bag(graph.vertices().collect());
        return tree_decomposition;
    }
    let mut max_bag = 2;
    let mut selector: S = S::from(graph);
    let mut pq = BinaryQueue::new();

    let mut bags: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
    let mut eliminated_at: FxHashMap<usize, usize> = FxHashMap::default();

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

        let mut nb: FxHashSet<usize> = selector.graph().neighborhood(u).collect();
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
        let mut rest: FxHashSet<usize> = selector.graph().vertices().collect();
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
}*/

#[cfg(test)]
mod tests {
    use crate::graph::BaseGraph;
    use crate::graph::HashMapGraph;
    use crate::graph::MutableGraph;
    use crate::heuristic_elimination_order::{MinFillSelector, Selector};
    use crate::io::PaceReader;
    use fxhash::FxHashMap;
    use std::convert::TryFrom;
    use std::fs::File;
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
        let reader = BufReader::new(f);
        let reader = PaceReader(reader);
        let graph = HashMapGraph::try_from(reader).unwrap();

        let vertices: Vec<_> = graph.vertices().collect();
        let fc1: FxHashMap<_, _> = vertices
            .iter()
            .map(|v| (*v, graph.fill_in_count(*v) as i64))
            .collect();

        let selector = MinFillSelector::from(graph);
        let fc2: FxHashMap<_, _> = vertices.iter().map(|v| (*v, selector.value(*v))).collect();

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
        let reader = BufReader::new(f);
        let reader = PaceReader(reader);
        let mut graph = HashMapGraph::try_from(reader).unwrap();
        let mut selector = MinFillSelector::from(graph.clone());

        let mut vertices: Vec<_> = graph.vertices().collect();

        while let Some(v) = vertices.pop() {
            graph.eliminate_vertex(v);
            selector.eliminate_vertex(v);

            let mut a: Vec<_> = selector.graph.vertices().collect();
            a.sort_unstable();
            let mut b: Vec<_> = graph.vertices().collect();
            b.sort_unstable();
            assert_eq!(a, b);
            let fc1: FxHashMap<_, _> = vertices
                .iter()
                .map(|v| (*v, graph.fill_in_count(*v) as i64))
                .collect();
            let fc2: FxHashMap<_, _> = vertices.iter().map(|v| (*v, selector.value(*v))).collect();

            for v in fc1.keys() {
                assert_eq!(fc1.get(v).unwrap(), fc2.get(v).unwrap());
            }
        }
    }
}
