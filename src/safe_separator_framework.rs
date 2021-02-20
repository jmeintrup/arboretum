use crate::graph::BaseGraph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::solver::{AlgorithmTypes, ComputationResult};
use crate::tree_decomposition::TreeDecomposition;
use fxhash::{FxHashMap, FxHashSet};
use std::borrow::Borrow;
use std::cell::Cell;
use std::cmp::max;
use std::rc::Rc;

#[cfg(feature = "log")]
use log::info;

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
    delayed_separation_levels: Vec<SeparationLevel>,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
    upperbound_td: Option<TreeDecomposition>,
    limits: &'a SafeSeparatorLimits,
    algorithms: &'a AlgorithmTypes,
    seed: Option<u64>,
}

impl<'a> SearchState<'a> {
    fn fork(&self, graph: HashMapGraph, upperbound_td: Option<TreeDecomposition>) -> Self {
        Self {
            graph,
            separation_level: self.separation_level,
            delayed_separation_levels: self.delayed_separation_levels.clone(),
            lower_bound: self.lower_bound.clone(),
            log_state: self.log_state.clone(),
            upperbound_td,
            limits: &self.limits,
            algorithms: &self.algorithms,
            seed: self.seed,
        }
    }

    fn graph_from_cc(&self, cc: &FxHashSet<usize>, separator: &FxHashSet<usize>) -> HashMapGraph {
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

    pub fn process_separator(&mut self, separator: &FxHashSet<usize>) -> TreeDecomposition {
        let mut td = TreeDecomposition::default();
        let root = td.add_bag(separator.clone());
        if td.max_bag_size > 0 {
            let lb: &Cell<_> = &self.lower_bound;
            lb.set(max(lb.get(), td.max_bag_size - 1));
        }
        for cc in self.graph.separate(&separator).iter() {
            let graph = self.graph_from_cc(cc, separator);

            let search_sate = self.fork(graph, None);
            let partial = search_sate.search();
            td.combine_with(root, partial);
        }
        td
    }

    pub fn search(mut self) -> TreeDecomposition {
        let lowerbound = {
            let tmp: &Cell<_> = &self.lower_bound;
            tmp.get()
        };
        if self.graph.order() <= lowerbound + 1 {
            return TreeDecomposition::with_root(self.graph.vertices().collect());
        }
        #[cfg(feature = "handle-ctrlc")]
        if crate::signals::received_ctrl_c() {
            // unknown lowerbound
            return TreeDecomposition::with_root(self.graph.vertices().collect());
        }
        while self.separation_level < SeparationLevel::MinorSafeClique {
            #[cfg(feature = "handle-ctrlc")]
            if crate::signals::received_ctrl_c() {
                // unknown lowerbound
                return TreeDecomposition::with_root(self.graph.vertices().collect());
            }
            match Self::find_separator(&self.graph, &self.limits, &self.separation_level) {
                SeparatorSearchResult::Some(separator) => {
                    #[cfg(log)]
                    info!(" found safe separator: {:?}", self.separation_level);
                    let mut log_state = self.log_state.get();
                    log_state.increment(self.separation_level);
                    self.log_state.set(log_state);
                    return self.process_separator(&separator);
                }
                SeparatorSearchResult::None => {
                    self.separation_level = self.separation_level.increment().unwrap();
                }
                SeparatorSearchResult::Delayed => {
                    if self.limits.check_again_before_atom {
                        #[cfg(feature = "log")]
                        info!(" delaying separator search: {:?}", self.separation_level);
                        self.delayed_separation_levels.push(self.separation_level);
                    }
                    self.separation_level = self.separation_level.increment().unwrap();
                }
            }
        }
        #[cfg(feature = "log")]
        info!(" searching for minor safe");
        if self.separation_level == SeparationLevel::MinorSafeClique
            && self.graph.order() < self.limits.minor_safe_separator
        {
            #[cfg(feature = "handle-ctrlc")]
            if crate::signals::received_ctrl_c() {
                // unknown lowerbound
                return match self.upperbound_td {
                    None => TreeDecomposition::with_root(self.graph.vertices().collect()),
                    Some(upperbound_td) => upperbound_td,
                };
            }
            if let Some(result) = self.graph.find_minor_safe_separator(
                self.upperbound_td.clone(),
                self.seed,
                self.limits.minor_safe_separator_tries,
                self.limits.minor_safe_separator_max_missing,
            ) {
                #[cfg(feature = "log")]
                info!(" found safe separator: {:?}", self.separation_level);
                let mut log_state = self.log_state.get();
                log_state.increment(self.separation_level);
                self.log_state.set(log_state);

                let heuristic_td = result.tree_decomposition;
                #[cfg(feature = "handle-ctrlc")]
                if crate::signals::received_ctrl_c() {
                    return heuristic_td;
                }

                let separator = result.separator;

                let mut td = TreeDecomposition::default();
                let root = td.add_bag(separator.clone());
                if td.max_bag_size > 0 {
                    let lb: &Cell<_> = &self.lower_bound;
                    lb.set(max(lb.get(), td.max_bag_size - 1));
                }
                for cc in self.graph.separate(&separator) {
                    let graph = self.graph_from_cc(&cc, &separator);

                    let full_vertex_set: FxHashSet<_> = graph.vertices().collect();

                    let mut partial_heuristic_bags: Vec<_> = heuristic_td
                        .bags
                        .iter()
                        .filter(|b| full_vertex_set.is_superset(&b.vertex_set))
                        .cloned()
                        .collect();
                    let old_to_new: FxHashMap<usize, usize> = partial_heuristic_bags
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
                    if partial_heuristic_bags.is_empty() {
                        let search_sate = self.fork(graph, None);

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
                        let search_sate = self.fork(graph, Some(upperbound_td));

                        let partial = search_sate.search();
                        td.combine_with(root, partial);
                    }
                }
                return td;
            }
        }
        if self.limits.check_again_before_atom {
            while !self.delayed_separation_levels.is_empty() {
                #[cfg(feature = "handle-ctrlc")]
                if crate::signals::received_ctrl_c() {
                    return match self.upperbound_td {
                        None => TreeDecomposition::with_root(self.graph.vertices().collect()),
                        Some(upperbound_td) => upperbound_td,
                    };
                }
                match Self::find_separator(&self.graph, &self.limits, &self.separation_level) {
                    SeparatorSearchResult::Some(separator) => {
                        #[cfg(feature = "log")]
                        info!(" found safe separator: {:?}", self.separation_level);
                        let mut log_state = self.log_state.get();
                        log_state.increment(self.separation_level);
                        self.log_state.set(log_state);
                        return self.process_separator(&separator);
                    }
                    _ => {
                        self.delayed_separation_levels.pop();
                    }
                }
            }
        }
        #[cfg(feature = "log")]
        info!(" solving atom");
        self.separation_level = SeparationLevel::Atomic;
        #[cfg(feature = "handle-ctrlc")]
        if crate::signals::received_ctrl_c() {
            return match self.upperbound_td {
                None => TreeDecomposition::with_root(self.graph.vertices().collect()),
                Some(upperbound_td) => upperbound_td,
            };
        }

        let mut log_state = self.log_state.get();
        log_state.increment(self.separation_level);
        log_state.set_max_atom(self.graph.order());
        self.log_state.set(log_state);
        #[cfg(feature = "log")]
        info!(" computing upperbound_td");
        let upperbound_td = self.algorithms.upperbound.compute(&self.graph, lowerbound);
        #[cfg(feature = "handle-ctrlc")]
        if crate::signals::received_ctrl_c() {
            // unknown lowerbound
            return match upperbound_td {
                Some(td) => td,
                None => match self.upperbound_td {
                    None => TreeDecomposition::with_root(self.graph.vertices().collect()),
                    Some(upperbound_td) => upperbound_td,
                },
            };
        }
        let upperbound = match &upperbound_td {
            None => self.graph.order() - 1,
            Some(td) => td.max_bag_size - 1,
        };
        let atom_lowerbound = self.algorithms.lowerbound.compute(&self.graph);
        {
            let lb: &Cell<_> = self.lower_bound.borrow();
            if atom_lowerbound > lb.get() {
                #[cfg(feature = "log")]
                info!(
                    "c Found new Lowerbound. Previous {} Now {}",
                    lb.get(),
                    atom_lowerbound
                );
                lb.set(atom_lowerbound);
            }

            #[cfg(feature = "log")]
            info!(
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
        #[cfg(feature = "handle-ctrlc")]
        if crate::signals::received_ctrl_c() {
            // unknown lowerbound
            return match upperbound_td {
                Some(upperbound_td) => upperbound_td,
                None => TreeDecomposition::with_root(self.graph.vertices().collect()),
            };
        }
        match self
            .algorithms
            .atom_solver
            .compute(self.graph.borrow(), lowerbound, upperbound)
        {
            ComputationResult::ComputedTreeDecomposition(td) => td,
            _ => match upperbound_td {
                None => match self.upperbound_td {
                    Some(upperbound_td) => upperbound_td,
                    None => TreeDecomposition::with_root(self.graph.vertices().collect()),
                },
                Some(td) => td,
            },
        }
    }

    fn find_separator(
        graph: &HashMapGraph,
        limits: &SafeSeparatorLimits,
        separation_level: &SeparationLevel,
    ) -> SeparatorSearchResult {
        match separation_level {
            SeparationLevel::Connected => {
                if graph.order() <= limits.size_one_separator {
                    match graph.find_safe_bi_connected_separator() {
                        None => SeparatorSearchResult::None,
                        Some(s) => SeparatorSearchResult::Some(s),
                    }
                } else {
                    SeparatorSearchResult::Delayed
                }
            }
            SeparationLevel::BiConnected => {
                if graph.order() <= limits.size_two_separator {
                    match graph.find_safe_bi_connected_separator() {
                        None => SeparatorSearchResult::None,
                        Some(s) => SeparatorSearchResult::Some(s),
                    }
                } else {
                    SeparatorSearchResult::Delayed
                }
            }
            SeparationLevel::TriConnected => {
                if graph.order() <= limits.size_three_separator {
                    match graph.find_safe_tri_connected_separator() {
                        None => SeparatorSearchResult::None,
                        Some(s) => SeparatorSearchResult::Some(s),
                    }
                } else {
                    SeparatorSearchResult::Delayed
                }
            }
            SeparationLevel::Clique => {
                if graph.order() <= limits.clique_separator {
                    match graph.find_clique_minimal_separator() {
                        None => SeparatorSearchResult::None,
                        Some(s) => SeparatorSearchResult::Some(s),
                    }
                } else {
                    SeparatorSearchResult::Delayed
                }
            }
            SeparationLevel::AlmostClique => {
                if graph.order() <= limits.almost_clique_separator {
                    match graph.find_almost_clique_minimal_separator() {
                        None => SeparatorSearchResult::None,
                        Some(s) => SeparatorSearchResult::Some(s),
                    }
                } else {
                    SeparatorSearchResult::Delayed
                }
            }
            _ => SeparatorSearchResult::None,
        }
    }
}

enum SeparatorSearchResult {
    Some(FxHashSet<usize>),
    None,
    Delayed,
}

#[derive(Clone, Copy)]
pub struct SafeSeparatorLimits {
    size_one_separator: usize,
    size_two_separator: usize,
    size_three_separator: usize,
    clique_separator: usize,
    almost_clique_separator: usize,
    minor_safe_separator: usize,
    minor_safe_separator_max_missing: usize,
    minor_safe_separator_tries: usize,
    check_again_before_atom: bool,
}

impl Default for SafeSeparatorLimits {
    fn default() -> Self {
        Self {
            size_one_separator: 100_000,
            size_two_separator: 1_000,
            size_three_separator: 300,
            clique_separator: 10_000,
            almost_clique_separator: 300,
            minor_safe_separator: 10_000,
            minor_safe_separator_max_missing: 1_000,
            minor_safe_separator_tries: 25,
            check_again_before_atom: false,
        }
    }
}

impl SafeSeparatorLimits {
    pub fn skip_all() -> Self {
        Self {
            size_one_separator: 0,
            size_two_separator: 0,
            size_three_separator: 0,
            clique_separator: 0,
            almost_clique_separator: 0,
            minor_safe_separator: 0,
            minor_safe_separator_max_missing: 0,
            minor_safe_separator_tries: 0,
            check_again_before_atom: false,
        }
    }

    pub fn only_cut_vertex() -> Self {
        Self {
            size_one_separator: usize::MAX,
            size_two_separator: 0,
            size_three_separator: 0,
            clique_separator: 0,
            almost_clique_separator: 0,
            minor_safe_separator: 0,
            minor_safe_separator_max_missing: 0,
            minor_safe_separator_tries: 0,
            check_again_before_atom: false,
        }
    }

    pub fn size_two_separator(mut self, limit: usize) -> Self {
        if limit < self.size_one_separator {
            panic!("Size two separator limit can not be smaller than size one limit");
        }
        if limit > self.size_three_separator {
            panic!("Size two separator limit can not be larger than size three limit");
        }
        self.size_two_separator = limit;
        self
    }
    pub fn size_three_separator(mut self, limit: usize) -> Self {
        if limit < self.size_one_separator {
            panic!("Size three separator limit can not be smaller than size one limit");
        }
        if limit < self.size_two_separator {
            panic!("Size three separator limit can not be smaller than size two limit");
        }
        self.size_three_separator = limit;
        self
    }
    impl_setter!(self, clique_separator, usize);
    impl_setter!(self, almost_clique_separator, usize);
    impl_setter!(self, minor_safe_separator, usize);
    impl_setter!(self, minor_safe_separator_max_missing, usize);
    impl_setter!(self, minor_safe_separator_tries, usize);
}

pub struct SafeSeparatorFramework {
    safe_separator_limits: SafeSeparatorLimits,
    algorithms: AlgorithmTypes,
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
    impl_setter!(self, safe_separator_limits, SafeSeparatorLimits);
    impl_setter!(self, algorithms, AlgorithmTypes);

    pub fn compute(self, graph: &HashMapGraph, lowerbound: usize) -> DecompositionResult {
        let lowerbound = Rc::new(Cell::new(lowerbound));
        let log_state = Rc::new(Cell::new(DecompositionInformation::default()));
        let limits = self.safe_separator_limits;
        let algorithms = self.algorithms;
        let td = (SearchState {
            graph: graph.clone(),
            separation_level: SeparationLevel::Connected,
            delayed_separation_levels: vec![],
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

    fn set_max_atom(&mut self, candidate: usize) {
        self.max_atom = max(self.max_atom, candidate);
    }
}

pub struct DecompositionResult {
    pub tree_decomposition: TreeDecomposition,
    pub decomposition_information: DecompositionInformation,
    pub lowerbound: usize,
}
