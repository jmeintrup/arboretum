use crate::datastructures::{BitSet, PartitionRefinement};
use crate::graph::{BaseGraph, HashMapGraph};
use crate::heuristic_elimination_order::{
    EliminationOrderDecomposer, HeuristicEliminationDecomposer, MinFillSelector, Selector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::solver::{AtomSolver, Bounds, ComputationResult};
use crate::tree_decomposition::TreeDecomposition;
use fxhash::{FxHashMap, FxHashSet};
use std::cmp::max;
use std::hash::{Hash, Hasher};

#[cfg(log)]
use log::info;

pub struct QuickBB {
    upperbound: usize,
    lowerbound: usize,
    og_graph: HashMapGraph,
    og_order: usize,
    working_graph: MinFillSelector,
    vertex_map: FxHashMap<usize, usize>,
    vertex_elimination_map: FxHashMap<BitSet, usize>,
    memory: FxHashMap<SearchState, usize>,
    clique: FxHashSet<usize>,
    best: Option<Vec<usize>>,
}

#[derive(Clone)]
struct SearchState {
    elimination_set: BitSet,
    width: usize,
    vertex: Option<usize>,
}

impl SearchState {
    fn root(size: usize) -> Self {
        Self {
            elimination_set: BitSet::new(size),
            width: 0,
            vertex: None,
        }
    }

    fn fork(&self, quick_bb: &QuickBB, vertex: usize) -> Self {
        let mut elimination_set = self.elimination_set.clone();
        elimination_set.set_bit(*quick_bb.vertex_map.get(&vertex).unwrap());
        Self {
            elimination_set,
            width: max(
                quick_bb
                    .working_graph
                    .graph()
                    .neighborhood_set(vertex)
                    .len(),
                self.width,
            ),
            vertex: Some(vertex),
        }
    }

    fn is_finished(&self, quick_bb: &QuickBB) -> bool {
        quick_bb.working_graph.graph().order() == 0
    }

    fn update_solution(&self, quick_bb: &mut QuickBB) -> bool {
        if !self.is_finished(quick_bb) || self.width >= quick_bb.upperbound {
            return false;
        }
        quick_bb.upperbound = self.width;
        let mut permutation: Vec<usize> = Vec::with_capacity(quick_bb.og_order);
        let mut vertex_set = BitSet::new(quick_bb.og_order);
        while vertex_set.cardinality() < quick_bb.og_order {
            let v = quick_bb.vertex_elimination_map.get(&vertex_set);
            if let Some(v) = v {
                permutation.push(*v);
                vertex_set.set_bit(*quick_bb.vertex_map.get(v).unwrap());
            } else {
                return true;
            }
        }
        quick_bb.best = Option::from(permutation);
        true
    }

    fn bound(&self, quick_bb: &QuickBB) -> bool {
        self.width >= quick_bb.upperbound
            || MinorMinWidth::with_graph(quick_bb.working_graph.graph()).compute()
                >= quick_bb.upperbound
    }

    fn branch(&self, quick_bb: &mut QuickBB) -> Vec<SearchState> {
        let mut children: Vec<SearchState> = vec![];

        // simplicial rule
        if let Some(vertex) = quick_bb
            .working_graph
            .graph()
            .vertices()
            .find(|v| quick_bb.working_graph.graph().is_simplicial(*v))
        {
            return vec![self.fork(quick_bb, vertex)];
        }

        // almost simplicial rule
        if let Some(vertex) = quick_bb
            .working_graph
            .graph()
            .vertices()
            .filter(|v| !quick_bb.clique.contains(v))
            .find(|v| {
                quick_bb.working_graph.graph().is_almost_simplicial(*v)
                    && quick_bb.working_graph.graph().degree(*v) < quick_bb.lowerbound
            })
        {
            return vec![self.fork(quick_bb, vertex)];
        }

        // twin rule
        let twins = Twins::for_graph(quick_bb.working_graph.graph()).find();
        for set in twins.values() {
            let v = *set.iter().next().unwrap();
            if quick_bb.clique.contains(&v)
                || (self.vertex.is_some()
                    && quick_bb
                        .working_graph
                        .graph()
                        .has_vertex(self.vertex.unwrap())
                    && quick_bb.working_graph.graph().has_vertex(v)
                    && quick_bb
                        .working_graph
                        .graph()
                        .has_edge(self.vertex.unwrap(), v))
            {
                continue;
            }
            children.push(self.fork(quick_bb, v));
        }

        children.sort_unstable_by(|a, b| {
            quick_bb
                .working_graph
                .fill_in_count(a.vertex.unwrap())
                .cmp(&quick_bb.working_graph.fill_in_count(b.vertex.unwrap()))
        });
        if children.is_empty() {
            if let Some(v) = quick_bb
                .clique
                .iter()
                .find(|v| quick_bb.working_graph.graph().has_vertex(**v))
            {
                children.push(self.fork(quick_bb, *v));
            }
        }
        children
    }
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.vertex.eq(&other.vertex) && self.elimination_set.eq(&other.elimination_set)
    }
}

impl Eq for SearchState {}

impl Hash for SearchState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.elimination_set.hash(state)
    }
}

impl QuickBB {
    fn branch_and_bound(&mut self, search_state: SearchState) -> usize {
        if search_state.update_solution(self) {
            return 0;
        }
        if search_state.bound(self) {
            return usize::MAX;
        }
        if !self.memory.contains_key(&search_state) {
            // apply edge addition rule
            let mut edges: Vec<(usize, usize)> = vec![];
            for v in self.working_graph.graph().vertices() {
                for u in self
                    .working_graph
                    .graph()
                    .vertices()
                    .filter(|u| v < *u && !self.working_graph.graph().has_edge(*u, v))
                {
                    let count: usize = self
                        .working_graph
                        .graph()
                        .neighborhood_set(u)
                        .iter()
                        .filter(|w| self.working_graph.graph().has_edge(**w, v))
                        .count();

                    if count > self.upperbound + 1 {
                        edges.push((u, v));
                    }
                }
            }
            self.working_graph.add_edges(&edges);

            // branch to children
            let children = search_state.branch(self);
            let mut width = usize::MAX;
            let mut final_branching_vertex: Option<usize> = None;
            for child in children {
                let v = child.vertex.unwrap();
                let delta = self.working_graph.graph().degree(v);

                let fill_info = self.working_graph.eliminate_with_info(v);
                self.vertex_elimination_map
                    .insert(child.elimination_set.clone(), v);
                let new_width = self.branch_and_bound(child);
                let m = max(new_width, delta);
                if m < width {
                    width = m;
                    final_branching_vertex = Some(v);
                }
                self.working_graph.undo_elimination(fill_info);
            }
            // undo edge addition
            self.working_graph.remove_edges(&edges);
            if let Some(v) = final_branching_vertex {
                self.vertex_elimination_map
                    .insert(search_state.elimination_set.clone(), v);
            }
            self.memory.insert(search_state.clone(), width);
        }
        *self.memory.get(&search_state).unwrap()
    }
}

impl AtomSolver for QuickBB {
    fn with_graph(graph: &HashMapGraph) -> Self
    where
        Self: Sized,
    {
        Self::with_bounds(graph, 0, graph.order())
    }

    fn with_bounds(graph: &HashMapGraph, lowerbound: usize, upperbound: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            upperbound,
            lowerbound,
            og_graph: graph.clone(),
            og_order: graph.order(),
            working_graph: MinFillSelector::from(graph.clone()),
            vertex_map: graph.vertices().enumerate().map(|(i, v)| (v, i)).collect(),
            vertex_elimination_map: Default::default(),
            memory: Default::default(),
            clique: Default::default(),
            best: None,
        }
    }

    fn compute(mut self) -> ComputationResult {
        if self.og_graph.order() <= self.lowerbound + 1 {
            return ComputationResult::ComputedTreeDecomposition(TreeDecomposition::with_root(
                self.og_graph.vertices().collect(),
            ));
        }

        if self.upperbound == self.lowerbound {
            return ComputationResult::Bounds(Bounds {
                lowerbound: self.lowerbound,
                upperbound: self.upperbound,
            });
        }
        let ub_decomposer: HeuristicEliminationDecomposer<MinFillSelector> =
            HeuristicEliminationDecomposer::with_graph(&self.og_graph);

        let result = ub_decomposer.compute_order_and_decomposition().unwrap();

        if (result.tree_decomposition.max_bag_size - 1) <= self.lowerbound {
            return ComputationResult::ComputedTreeDecomposition(result.tree_decomposition);
        }
        self.best = Some(result.permutation);

        if self.upperbound != self.lowerbound {
            let root = SearchState::root(self.og_graph.order());
            self.branch_and_bound(root);
            if self.best.is_some() {
                return ComputationResult::ComputedTreeDecomposition(
                    EliminationOrderDecomposer::new(self.og_graph.clone(), self.best.unwrap())
                        .compute()
                        .tree_decomposition,
                );
            }
        }

        ComputationResult::Bounds(Bounds {
            lowerbound: self.lowerbound,
            upperbound: self.upperbound,
        })
    }
}

struct Twins<'a> {
    graph: &'a HashMapGraph,
}

impl<'a> Twins<'a> {
    fn for_graph(graph: &'a HashMapGraph) -> Self {
        Self { graph }
    }

    fn find(self) -> FxHashMap<usize, FxHashSet<usize>> {
        let mut semi_twin_partition = PartitionRefinement::new(self.graph.vertices().collect());
        let mut full_twin_partition = PartitionRefinement::new(self.graph.vertices().collect());
        for v in self.graph.vertices() {
            let mut nb = self.graph.neighborhood_set(v).clone();
            semi_twin_partition.refine(&nb);
            nb.insert(v);
            full_twin_partition.refine(&nb);
        }
        let semi_twins = semi_twin_partition.partition().to_owned();
        let full_twins = full_twin_partition.partition().to_owned();

        let mut twins: FxHashMap<usize, FxHashSet<usize>> = Default::default();
        for v in self.graph.vertices() {
            let semi_twin = semi_twins.get(&v).unwrap().try_borrow().unwrap().clone();
            let full_twin = full_twins.get(&v).unwrap().try_borrow().unwrap().clone();
            twins.insert(
                v,
                if full_twin.len() > semi_twin.len() {
                    full_twin
                } else {
                    semi_twin
                },
            );
        }
        twins
    }
}
