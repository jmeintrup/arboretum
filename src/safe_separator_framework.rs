use crate::exact::TamakiPid;
use crate::graph::Graph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::tree_decomposition::{Bag, TreeDecomposition, TreeDecompositionValidationError};
use crate::heuristic_elimination_order::{
    heuristic_elimination_decompose, HeuristicEliminationDecomposer, MinFillDecomposer,
    MinFillSelector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::solver::AtomSolver;
use fnv::{FnvHashMap, FnvHashSet};
use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, RefCell};
use std::cmp::max;
use std::collections::VecDeque;
use std::hash::Hash;
use std::process::exit;
use std::rc::Rc;

#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Debug)]
enum SeparationLevel {
    Connected,
    BiConnected,
    TriConnected,
    Clique,
    AlmostClique,
    MinorSafeClique,
    Atomic,
}

impl SeparationLevel {
    pub fn increment(&self) -> Option<Self> {
        match self {
            Self::Connected => Some(Self::BiConnected),
            Self::BiConnected => Some(Self::TriConnected),
            Self::TriConnected => Some(Self::Clique),
            Self::Clique => Some(Self::AlmostClique),
            Self::AlmostClique => Some(Self::MinorSafeClique),
            Self::MinorSafeClique => Some(Self::Atomic),
            Self::Atomic => None,
        }
    }
}

struct SearchState {
    graph: HashMapGraph,
    separation_level: SeparationLevel,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
    upperbound_td: Option<TreeDecomposition>,
}

impl SearchState {
    fn graph_from_cc(&self, cc: &FnvHashSet<usize>, separator: &FnvHashSet<usize>) -> HashMapGraph {
        let mut graph = self.graph.vertex_induced(cc);
        for v in separator.iter() {
            for u in separator.iter().filter(|u| v < *u) {
                graph.add_edge(*v, *u);
            }
            for u in self
                .graph
                .neighborhood_set(*v)
                .iter()
                .filter(|u| cc.contains(*u))
            {
                graph.add_edge(*v, *u);
            }
        }
        graph
    }

    pub fn process_separator(&mut self, separator: &FnvHashSet<usize>) -> TreeDecomposition {
        let mut td = TreeDecomposition::new();
        let root = td.add_bag(separator.clone());
        if td.max_bag_size > 0 {
            let lb: &Cell<_> = self.lower_bound.borrow();
            lb.set(max(lb.get(), td.max_bag_size - 1));
        }
        for cc in self.graph.separate(&separator).iter() {
            let mut graph = self.graph_from_cc(cc, separator);

            let mut search_sate = SearchState {
                graph,
                separation_level: self.separation_level,
                lower_bound: self.lower_bound.clone(),
                log_state: self.log_state.clone(),
                upperbound_td: None,
            };
            let partial = search_sate.search();
            td.combine_with(root, partial);
            /*if td.max_bag_size > 0 {
                let lb: &Cell<_> = self.lower_bound.borrow();
                lb.set(max(lb.get(), td.max_bag_size - 1));
            }*/
        }
        return td;
    }

    pub fn search(&mut self) -> TreeDecomposition {
        let lowerbound = {
            let tmp: &Cell<_> = self.lower_bound.borrow();
            tmp.get()
        };
        if self.graph.order() <= lowerbound + 1 {
            let mut td = TreeDecomposition::new();
            td.add_bag(self.graph.vertices().collect());
            return td;
        }
        while self.separation_level != SeparationLevel::MinorSafeClique {
            if let Some(separator) = self.find_separator() {
                println!("c found safe separator: {:?}", self.separation_level);
                let mut log_state = self.log_state.get();
                log_state.increment(self.separation_level);
                self.log_state.set(log_state);
                return self.process_separator(&separator);
            } else {
                self.separation_level = self.separation_level.increment().unwrap();
            }
        }
        if self.separation_level == SeparationLevel::MinorSafeClique {
            if let Some(mut result) = self
                .graph
                .find_minor_safe_separator(self.upperbound_td.clone())
            {
                println!("c found safe separator: {:?}", self.separation_level);
                let mut log_state = self.log_state.get();
                log_state.increment(self.separation_level);
                self.log_state.set(log_state);

                let mut heuristic_td = result.tree_decomposition;

                let separator = result.separator;

                let mut td = TreeDecomposition::new();
                let root = td.add_bag(separator.clone());
                if td.max_bag_size > 0 {
                    let lb: &Cell<_> = self.lower_bound.borrow();
                    lb.set(max(lb.get(), td.max_bag_size - 1));
                }
                let components = self.graph.separate(&separator);
                for cc in self.graph.separate(&separator) {
                    let mut graph = self.graph_from_cc(&cc, &separator);

                    let full_vertex_set: FnvHashSet<_> = graph.vertices().collect();

                    let mut partial_heuristic_bags: Vec<_> = heuristic_td
                        .bags
                        .iter()
                        .filter(|b| full_vertex_set.is_superset(&b.vertex_set))
                        .cloned()
                        .collect();
                    let old_to_new: FnvHashMap<usize, usize> = partial_heuristic_bags
                        .iter()
                        .enumerate()
                        .map(|(id, b)| (b.id, id))
                        .collect();

                    for bag in partial_heuristic_bags.iter_mut() {
                        bag.id = *old_to_new.get(&bag.id).unwrap();
                        bag.neighbors = bag
                            .neighbors
                            .iter()
                            .filter(|i| old_to_new.contains_key(i))
                            .map(|i| old_to_new.get(i).unwrap())
                            .copied()
                            .collect();
                    }
                    if partial_heuristic_bags.len() == 0 {
                        let mut search_sate = SearchState {
                            graph,
                            separation_level: self.separation_level,
                            lower_bound: self.lower_bound.clone(),
                            log_state: self.log_state.clone(),
                            upperbound_td: None,
                        };

                        let partial = search_sate.search();
                        td.combine_with(root, partial);
                    } else {
                        let max_bag_size = partial_heuristic_bags
                            .iter()
                            .map(|b| b.vertex_set.len())
                            .max()
                            .unwrap();
                        let upperbound_td = TreeDecomposition {
                            bags: partial_heuristic_bags,
                            root: Some(0),
                            max_bag_size,
                        };
                        let mut search_sate = SearchState {
                            graph,
                            separation_level: self.separation_level,
                            lower_bound: self.lower_bound.clone(),
                            log_state: self.log_state.clone(),
                            upperbound_td: Some(upperbound_td),
                        };

                        let partial = search_sate.search();
                        td.combine_with(root, partial);
                    }
                }
                return td;
            }
        }

        let mut log_state = self.log_state.get();
        log_state.increment(self.separation_level);
        self.log_state.set(log_state);
        //todo: add solver builder that returns a solver that is then called on the graph
        let upperbound =
            MinFillDecomposer::with_bounds(&self.graph, lowerbound, self.graph.order())
                .compute()
                .unwrap();
        let mmw = MinorMinWidth::with_graph(&self.graph).compute();
        {
            let lb: &Cell<_> = self.lower_bound.borrow();
            if mmw > lb.get() {
                println!("c Found new Lowerbound. Previous {} Now {}", lb.get(), mmw);
                lb.set(mmw);
            }
            println!(
                "c Atom with size {} has upperbound {}. Global lowerbound is {}",
                self.graph.order(),
                upperbound.max_bag_size - 1,
                lb.get()
            );
        }
        let lowerbound = {
            let tmp: &Cell<_> = self.lower_bound.borrow();
            tmp.get()
        };
        let mut solver =
            TamakiPid::with_bounds(self.graph.borrow(), lowerbound, upperbound.max_bag_size - 1);
        match solver.compute() {
            Ok(td) => td,
            Err(_) => upperbound,
        }
    }

    pub fn find_separator(&self) -> Option<FnvHashSet<usize>> {
        match self.separation_level {
            SeparationLevel::Connected => self.graph.find_cut_vertex(),
            SeparationLevel::BiConnected => self.graph.find_safe_bi_connected_separator(),
            SeparationLevel::TriConnected => {
                if self.graph.order() < 300 {
                    self.graph.find_safe_tri_connected_separator()
                } else {
                    None
                }
            }
            SeparationLevel::Clique => self.graph.find_clique_minimal_separator(),
            SeparationLevel::AlmostClique => {
                if self.graph.order() < 300 {
                    self.graph.find_almost_clique_minimal_separator()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub struct SafeSeparatorFramework {
    graph: HashMapGraph,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
    search_state: SearchState,
}

impl SafeSeparatorFramework {
    pub fn new(graph: HashMapGraph, lower_bound: usize) -> Self {
        let lb = Rc::new(Cell::new(lower_bound));
        let log_state = Rc::new(Cell::new(DecompositionInformation::default()));
        Self {
            graph: graph.clone(),
            lower_bound: lb.clone(),
            log_state: log_state.clone(),
            search_state: SearchState {
                graph,
                separation_level: SeparationLevel::Connected,
                lower_bound: lb,
                log_state,
                upperbound_td: None,
            },
        }
    }

    pub fn compute(mut self) -> DecompositionResult {
        let td = self.search_state.search();
        let lowerbound = self.lower_bound.get();
        let log_state = self.log_state.get();

        DecompositionResult {
            tree_decomposition: td,
            decomposition_information: self.log_state.get(),
            lowerbound,
        }
    }
}

#[derive(Debug, Clone)]
struct AtomState {
    graph: HashMapGraph,
    target_bag: usize,
    tree_decomposition: Option<TreeDecomposition>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DecompositionInformation {
    n_separators: usize,
    n_cut_vertices: usize,
    n_degree_two_separators: usize,
    n_degree_three_separators: usize,
    n_clique_separators: usize,
    n_almost_clique_separators: usize,
    n_minor_safe_separators: usize,
    n_atoms: usize,
    max_atom: usize,
}

impl DecompositionInformation {
    fn increment(&mut self, separation_level: SeparationLevel) {
        match separation_level {
            SeparationLevel::Atomic => {}
            _ => {
                self.n_separators += 1;
            }
        }
        match separation_level {
            SeparationLevel::Connected => self.n_cut_vertices += 1,
            SeparationLevel::BiConnected => self.n_degree_two_separators += 1,
            SeparationLevel::TriConnected => self.n_degree_three_separators += 1,
            SeparationLevel::Clique => self.n_clique_separators += 1,
            SeparationLevel::AlmostClique => self.n_almost_clique_separators += 1,
            SeparationLevel::MinorSafeClique => self.n_minor_safe_separators += 1,
            SeparationLevel::Atomic => self.n_atoms += 1,
        }
    }

    fn set_max_atom(&mut self, max_atom: usize) {
        self.max_atom = max_atom;
    }
}

pub struct DecompositionResult {
    pub tree_decomposition: TreeDecomposition,
    pub decomposition_information: DecompositionInformation,
    pub lowerbound: usize,
}
