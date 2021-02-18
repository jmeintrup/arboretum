use crate::exact::TamakiPid;
use crate::graph::Graph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::heuristic_elimination_order::{
    heuristic_elimination_decompose, HeuristicEliminationDecomposer, MinFillDecomposer,
    MinFillSelector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::solver::{AtomSolver, AtomSolverType, LowerboundHeuristicType, UpperboundHeuristicType};
use crate::tree_decomposition::{Bag, TreeDecomposition, TreeDecompositionValidationError};
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

struct SearchState<'a> {
    graph: HashMapGraph,
    separation_level: SeparationLevel,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
    upperbound_td: Option<TreeDecomposition>,
    limits: &'a SafeSeparatorLimits,
    algorithms: &'a AlgorithmTypes,
    seed: Option<u64>,
}

struct AlgorithmTypes {
    atom_solver: AtomSolverType,
    upperbound: UpperboundHeuristicType,
    lowerbound: LowerboundHeuristicType,
}

impl Default for AlgorithmTypes {
    fn default() -> Self {
        Self {
            atom_solver: AtomSolverType::Tamaki,
            upperbound: UpperboundHeuristicType::All,
            lowerbound: LowerboundHeuristicType::MinorMinWidth,
        }
    }
}

impl<'a> SearchState<'a> {
    fn fork(&self, graph: HashMapGraph, upperbound_td: Option<TreeDecomposition>) -> Self {
        Self {
            graph,
            separation_level: self.separation_level,
            lower_bound: self.lower_bound.clone(),
            log_state: self.log_state.clone(),
            upperbound_td,
            limits: &self.limits,
            algorithms: &self.algorithms,
            seed: self.seed,
        }
    }

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

            let mut search_sate = self.fork(graph, None);
            let partial = search_sate.search();
            td.combine_with(root, partial);
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
        while self.separation_level < SeparationLevel::MinorSafeClique {
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
        if self.separation_level == SeparationLevel::MinorSafeClique
            && self.graph.order() < self.limits.minor_safe_separator
        {
            if let Some(mut result) = self
                .graph
                .find_minor_safe_separator(self.upperbound_td.clone(), self.seed)
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
                        let mut search_sate = self.fork(graph, None);

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
                        let mut search_sate = self.fork(graph, Some(upperbound_td));

                        let partial = search_sate.search();
                        td.combine_with(root, partial);
                    }
                }
                return td;
            }
        }
        self.separation_level = SeparationLevel::Atomic;

        let mut log_state = self.log_state.get();
        log_state.increment(self.separation_level);
        self.log_state.set(log_state);
        //todo: add solver builder that returns a solver that is then called on the graph
        /*let upperbound =
        MinFillDecomposer::with_bounds(&self.graph, lowerbound, self.graph.order())
            .compute()
            .unwrap();*/
        let upperbound_td = self.algorithms.upperbound.compute(&self.graph, lowerbound);
        let upperbound = match &upperbound_td {
            None => self.graph.order() - 1,
            Some(td) => td.max_bag_size - 1,
        };
        let atom_lowerbound = self.algorithms.lowerbound.compute(&self.graph);
        {
            let lb: &Cell<_> = self.lower_bound.borrow();
            if atom_lowerbound > lb.get() {
                println!("c Found new Lowerbound. Previous {} Now {}", lb.get(), atom_lowerbound);
                lb.set(atom_lowerbound);
            }

            println!(
                "c Atom with size {} has upperbound {}. Global lowerbound is {}",
                self.graph.order(),
                upperbound,
                lb.get()
            );
        }
        let lowerbound = {
            let tmp: &Cell<_> = self.lower_bound.borrow();
            tmp.get()
        };
        match self.algorithms.atom_solver.compute(self.graph.borrow(), lowerbound, upperbound) {
            Ok(td) => td,
            Err(_) => match upperbound_td {
                None => {
                    let mut td = TreeDecomposition::new();
                    td.add_bag(self.graph.vertices().collect());
                    td
                }
                Some(td) => td,
            },
        }
    }

    pub fn find_separator(&self) -> Option<FnvHashSet<usize>> {
        match self.separation_level {
            SeparationLevel::Connected => {
                if self.graph.order() < self.limits.size_one_separator {
                    self.graph.find_safe_bi_connected_separator()
                } else {
                    None
                }
            }
            SeparationLevel::BiConnected => {
                if self.graph.order() < self.limits.size_two_separator {
                    self.graph.find_safe_bi_connected_separator()
                } else {
                    None
                }
            }
            SeparationLevel::TriConnected => {
                if self.graph.order() < self.limits.size_three_separator {
                    self.graph.find_safe_tri_connected_separator()
                } else {
                    None
                }
            }
            SeparationLevel::Clique => {
                if self.graph.order() < self.limits.clique_separator {
                    self.graph.find_clique_minimal_separator()
                } else {
                    None
                }
            }
            SeparationLevel::AlmostClique => {
                if self.graph.order() < self.limits.almost_clique_separator {
                    self.graph.find_almost_clique_minimal_separator()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub struct SafeSeparatorLimits {
    size_one_separator: usize,
    size_two_separator: usize,
    size_three_separator: usize,
    clique_separator: usize,
    almost_clique_separator: usize,
    minor_safe_separator: usize,
    minor_safe_separator_max_missing: usize,
    minor_safe_separator_tries: usize,
}

impl Default for SafeSeparatorLimits {
    fn default() -> Self {
        Self {
            size_one_separator: usize::MAX,
            size_two_separator: usize::MAX,
            size_three_separator: 300,
            clique_separator: usize::MAX,
            almost_clique_separator: 300,
            minor_safe_separator: usize::MAX,
            minor_safe_separator_max_missing: usize::MAX,
            minor_safe_separator_tries: 25,
        }
    }
}

impl SafeSeparatorLimits {
    pub fn size_one_separator(mut self, limit: usize) -> Self {
        self.size_one_separator = limit;
        self
    }
    pub fn size_two_separator(mut self, limit: usize) -> Self {
        if limit < self.size_one_separator {
            panic!("Size two separator limit can not be smaller than size one limit");
        }
        self.size_two_separator = limit;
        self
    }
    pub fn size_three_separator(mut self, limit: usize) -> Self {
        if limit < self.size_one_separator {
            panic!("Size three separator limit can not be smaller than size two limit");
        }
        self.size_three_separator = limit;
        self
    }
    pub fn clique_separator(mut self, limit: usize) -> Self {
        self.clique_separator = limit;
        self
    }
    pub fn almost_clique_separator(mut self, limit: usize) -> Self {
        self.almost_clique_separator = limit;
        self
    }
    pub fn minor_safe_separator(mut self, limit: usize) -> Self {
        self.minor_safe_separator = limit;
        self
    }
    pub fn minor_safe_separator_max_missing(mut self, limit: usize) -> Self {
        self.minor_safe_separator_max_missing = limit;
        self
    }
    pub fn minor_safe_separator_tries(mut self, limit: usize) -> Self {
        self.minor_safe_separator_tries = limit;
        self
    }
}

pub struct SafeSeparatorFramework {
    safe_separator_limits: SafeSeparatorLimits,
    algorithms: AlgorithmTypes::default(),
}

impl Default for SafeSeparatorFramework {
    fn default() -> Self {
        Self {
            safe_separator_limits: SafeSeparatorLimits::default(),
            algorithms: AlgorithmTypes::default(),
        }
    }
}

impl SafeSeparatorFramework {
    pub fn compute(mut self, graph: &HashMapGraph, lowerbound: usize) -> DecompositionResult {
        let lowerbound = Rc::new(Cell::new(lowerbound));
        let log_state = Rc::new(Cell::new(DecompositionInformation::default()));
        let limits = self.safe_separator_limits;
        let algorithms = self.algorithms;
        let td = (SearchState {
            graph: graph.clone(),
            separation_level: SeparationLevel::Connected,
            lower_bound: lowerbound.clone(),
            log_state: log_state.clone(),
            upperbound_td: None,
            limits: &limits,
            algorithms: &algorithms,
            seed: None,
        })
        .search();
        DecompositionResult {
            tree_decomposition: td,
            decomposition_information: log_state.get(),
            lowerbound: lowerbound.get(),
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
