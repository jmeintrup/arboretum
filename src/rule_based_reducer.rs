use crate::exact::TamakiPid;
use crate::graph::Graph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::heuristic_elimination_order::{
    heuristic_elimination_decompose, HeuristicEliminationDecomposer, MinFillDecomposer,
    MinFillSelector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::solver::AtomSolver;
use crate::tree_decomposition::{Bag, TreeDecomposition, TreeDecompositionValidationError};
use fnv::{FnvHashMap, FnvHashSet};
use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, RefCell};
use std::cmp::max;
use std::collections::VecDeque;
use std::hash::Hash;
use std::process::exit;
use std::rc::Rc;

#[cfg(feature = "handle-ctrlc")]
use crate::signals::received_ctrl_c;

#[inline]
fn eliminate(v: usize, graph: &mut HashMapGraph, stack: &mut Vec<FnvHashSet<usize>>) {
    let mut bag = graph.neighborhood_set(v).clone();
    bag.insert(v);
    stack.push(bag);
    graph.eliminate_vertex(v);
}

struct Cube {
    a: usize,
    b: usize,
    c: usize,
    x: usize,
    y: usize,
    z: usize,
    v: usize,
}

pub struct RuleBasedPreprocessor {
    stack: Vec<FnvHashSet<usize>>,
    pub lower_bound: usize,
    partial_tree_decomposition: TreeDecomposition,
    processed_graph: HashMapGraph,
}

impl RuleBasedPreprocessor {
    pub fn preprocess(&mut self) {
        self.lower_bound = 0;
        if self.processed_graph.order() == 0 {
            return;
        }

        self.remove_low_degree();
        if self.processed_graph.order() == 0 {
            self.process_stack();
            return;
        }
        while self.apply_rules() {
            #[cfg(feature = "handle-ctrlc")]
            if received_ctrl_c() { // unknown lowerbound
                self.lower_bound = 0;
                return break;
            }
        }
        if self.processed_graph.order() == 0 {
            self.process_stack();
        }
    }

    pub fn combine_into_td(
        mut self,
        mut td: TreeDecomposition,
        graph: &HashMapGraph,
    ) -> TreeDecomposition {
        let mut vertices = FnvHashSet::default();
        for bag in &td.bags {
            vertices.extend(bag.vertex_set.iter().copied())
        }
        let tmp = self.stack.clone();
        self.stack.push(vertices);
        self.process_stack();
        self.partial_tree_decomposition.flatten();
        self.partial_tree_decomposition.replace_bag(0, td);
        self.partial_tree_decomposition
    }

    pub fn into_td(mut self) -> TreeDecomposition {
        self.process_stack();
        self.partial_tree_decomposition
    }

    pub fn graph(&self) -> &HashMapGraph {
        &self.processed_graph
    }

    pub fn new(graph: &HashMapGraph) -> Self {
        Self {
            stack: vec![],
            lower_bound: 0,
            partial_tree_decomposition: TreeDecomposition::default(),
            processed_graph: graph.clone(),
        }
    }

    fn apply_rules(&mut self) -> bool {
        // islet
        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 0);
        if let Some(v) = found {
            let mut bag = FnvHashSet::with_capacity_and_hasher(1, Default::default());
            bag.insert(v);
            self.stack.push(bag);
            self.processed_graph.remove_vertex(v);
            return true;
        }
        // single-degree

        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 1);
        if let Some(v) = found {
            let mut bag = self.processed_graph.neighborhood_set(v).clone();
            bag.insert(v);
            self.stack.push(bag);
            self.processed_graph.remove_vertex(v);
            return true;
        }
        // series
        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 2);
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            self.lower_bound = max(self.lower_bound, 3);
            return true;
        }
        // triangle rule
        self.lower_bound = max(self.lower_bound, 4);

        let found = self
            .processed_graph
            .vertices()
            .filter(|v| self.processed_graph.degree(*v) == 3)
            .find(|v| {
                let tmp: Vec<_> = self
                    .processed_graph
                    .neighborhood_set(*v)
                    .iter()
                    .copied()
                    .collect();
                self.processed_graph.has_edge(tmp[0], tmp[1])
                    || self.processed_graph.has_edge(tmp[0], tmp[2])
                    || self.processed_graph.has_edge(tmp[1], tmp[2])
            });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        // buddy rule
        let found = self.processed_graph.vertices().find(|v| {
            self.processed_graph
                .vertices()
                .find(|u| {
                    v < u
                        && self.processed_graph.degree(*v) == 3
                        && self.processed_graph.degree(*u) == 3
                        && self.processed_graph.neighborhood_set(*u)
                            == self.processed_graph.neighborhood_set(*v)
                })
                .is_some()
        });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        // cube rule
        let mut cube: Option<Cube> = None;
        for v in self.processed_graph.vertices() {
            if self.processed_graph.degree(v) != 3 {
                continue;
            }
            let nb: Vec<_> = self
                .processed_graph
                .neighborhood_set(v)
                .iter()
                .copied()
                .collect();
            let (x, y, z) = (nb[0], nb[1], nb[2]);
            if self.processed_graph.degree(x) != 3
                || self.processed_graph.degree(y) != 3
                || self.processed_graph.degree(z) != 3
            {
                continue;
            }
            let x_nb: Vec<_> = self
                .processed_graph
                .neighborhood_set(x)
                .iter()
                .copied()
                .collect();
            let mut a = if x_nb[0] == v { x_nb[2] } else { x_nb[0] };
            let mut b = if x_nb[1] == v { x_nb[2] } else { x_nb[1] };
            if !(self.processed_graph.has_edge(y, a) && self.processed_graph.has_edge(z, b)) {
                std::mem::swap(&mut a, &mut b);
            }
            if !(self.processed_graph.has_edge(y, a) && self.processed_graph.has_edge(z, b)) {
                continue;
            }
            let c_option = self
                .processed_graph
                .neighborhood(y)
                .find(|c| *c != v && self.processed_graph.has_edge(z, *c));
            if c_option.is_none() {
                return false;
            }
            let c = c_option.unwrap();

            cube = Some(Cube {
                a,
                b,
                c,
                x,
                y,
                z,
                v,
            });
            break;
        }
        if let Some(cube) = cube {
            let mut bag = FnvHashSet::with_capacity_and_hasher(4, Default::default());
            bag.insert(cube.b);
            bag.insert(cube.z);
            bag.insert(cube.v);
            bag.insert(cube.c);
            self.processed_graph.remove_vertex(cube.z);
            self.processed_graph.add_edge(cube.a, cube.b);
            self.processed_graph.add_edge(cube.a, cube.c);
            self.processed_graph.add_edge(cube.a, cube.v);
            self.processed_graph.add_edge(cube.b, cube.c);
            self.processed_graph.add_edge(cube.b, cube.v);
            self.processed_graph.add_edge(cube.c, cube.v);
            self.stack.push(bag);

            return true;
        }

        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.is_simplicial(*v));
        if let Some(v) = found {
            self.lower_bound = max(self.lower_bound, self.processed_graph.degree(v));
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        let found = self.processed_graph.vertices().find(|v| {
            self.lower_bound > self.processed_graph.degree(*v)
                && self.processed_graph.is_almost_simplicial(*v)
        });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        false
    }

    fn process_stack(&mut self) {
        for new_bag in self.stack.drain(..).rev() {
            if let Some(old_bag) = self.partial_tree_decomposition.dfs().find(|old_bag| {
                new_bag
                    .iter()
                    .filter(|v| !old_bag.vertex_set.contains(*v))
                    .count()
                    <= 1
            }) {
                let old_id = old_bag.id;
                let id = self.partial_tree_decomposition.add_bag(new_bag);
                self.partial_tree_decomposition.add_edge(old_id, id);
            } else {
                self.partial_tree_decomposition.add_bag(new_bag);
            }
        }
    }

    fn remove_low_degree(&mut self) {
        // remove all low degree vertices. First remove all islets, then all islets and 1-degree,
        // then all islets, 1-degree and 2-degree
        for d in 0..3 {
            let mut visited: FnvHashSet<usize> = FnvHashSet::with_capacity_and_hasher(
                self.processed_graph.order(),
                Default::default(),
            );
            let mut queue = VecDeque::new();
            self.processed_graph
                .vertices()
                .filter(|v| self.processed_graph.degree(*v) <= d)
                .for_each(|v| {
                    visited.insert(v);
                    queue.push_back(v);
                });

            while let Some(v) = queue.pop_front() {
                let mut nb: FnvHashSet<_> = self.processed_graph.neighborhood(v).collect();
                if nb.len() > d {
                    continue;
                }
                self.processed_graph.eliminate_vertex(v);
                nb.iter().copied().for_each(|v| {
                    if self.processed_graph.has_vertex(v)
                        && self.processed_graph.degree(v) <= d
                        && !visited.contains(&v)
                    {
                        queue.push_back(v);
                        visited.insert(v);
                    }
                });
                nb.insert(v);
                self.lower_bound = max(self.lower_bound, nb.len() - 1);
                self.stack.push(nb);
            }
        }
    }
}
