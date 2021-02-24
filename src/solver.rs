use crate::exact::TamakiPid;
use crate::graph::{BaseGraph, HashMapGraph, MutableGraph};
use crate::heuristic_elimination_order::{
    HeuristicEliminationDecomposer, MinDegreeSelector, MinFillDegreeSelector, MinFillSelector,
    Selector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::rule_based_reducer::RuleBasedPreprocessor;
use crate::safe_separator_framework::{SafeSeparatorFramework, SafeSeparatorLimits, SeparationLevel, SeparatorSearchResult};
use crate::tree_decomposition::{TreeDecomposition, TreeDecompositionValidationError};
#[cfg(feature = "log")]
use log::info;
use std::cmp::{max, Ordering};
use std::collections::BinaryHeap;
use std::rc::Rc;
use std::cell::{Cell, RefCell, RefMut, Ref};
use fxhash::FxHashSet;

pub trait DynamicUpperboundHeuristic: AtomSolver {}
impl<S: Selector> DynamicUpperboundHeuristic for HeuristicEliminationDecomposer<S> {}

pub type Lowerbound = usize;
pub type Upperbound = usize;
pub type DynamicUpperbound = fn(&HashMapGraph, Lowerbound, Upperbound) -> ComputationResult;
pub type DynamicExact = fn(&HashMapGraph, Lowerbound, Upperbound) -> ComputationResult;
pub type DynamicLowerbound = fn(&HashMapGraph) -> usize;

#[derive(Clone, Copy)]
pub enum LowerboundHeuristicType {
    None,
    MinorMinWidth(usize),
    Custom(DynamicLowerbound),
}

impl Default for LowerboundHeuristicType {
    fn default() -> Self {
        Self::MinorMinWidth(10_000)
    }
}

impl LowerboundHeuristicType {
    pub(crate) fn compute(&self, graph: &HashMapGraph) -> Lowerbound {
        match self {
            LowerboundHeuristicType::None => {
                graph.vertices().map(|v| graph.degree(v)).min().unwrap_or(0)
            }
            LowerboundHeuristicType::MinorMinWidth(limit) => {
                if limit > &graph.order() {
                    graph.vertices().map(|v| graph.degree(v)).min().unwrap_or(0)
                } else {
                    MinorMinWidth::with_graph(&graph).compute()
                }
            }
            LowerboundHeuristicType::Custom(heuristic) => heuristic(&graph),
        }
    }
}

#[derive(Clone, Copy)]
pub enum UpperboundHeuristicType {
    None,
    MinFill,
    MinDegree,
    MinFillDegree,
    All,
    Custom(DynamicUpperbound),
}

impl Default for UpperboundHeuristicType {
    fn default() -> Self {
        Self::All
    }
}

impl UpperboundHeuristicType {
    pub(crate) fn compute(
        &self,
        graph: &HashMapGraph,
        lowerbound: usize,
    ) -> Option<TreeDecomposition> {
        match self {
            UpperboundHeuristicType::None => None,
            UpperboundHeuristicType::MinFill => {
                let decomposer: HeuristicEliminationDecomposer<MinFillSelector> =
                    HeuristicEliminationDecomposer::with_bounds(&graph, lowerbound, graph.order());
                Some(decomposer.compute().computed_tree_decomposition().unwrap())
            }
            UpperboundHeuristicType::MinDegree => {
                let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
                    HeuristicEliminationDecomposer::with_bounds(&graph, lowerbound, graph.order());
                Some(decomposer.compute().computed_tree_decomposition().unwrap())
            }
            UpperboundHeuristicType::MinFillDegree => {
                let decomposer: HeuristicEliminationDecomposer<MinFillDegreeSelector> =
                    HeuristicEliminationDecomposer::with_bounds(&graph, lowerbound, graph.order());
                Some(decomposer.compute().computed_tree_decomposition().unwrap())
            }
            UpperboundHeuristicType::All => {
                let a = HeuristicEliminationDecomposer::<MinFillSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
                .computed_tree_decomposition()
                .unwrap();
                let b = HeuristicEliminationDecomposer::<MinDegreeSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
                .computed_tree_decomposition()
                .unwrap();
                let c = HeuristicEliminationDecomposer::<MinFillDegreeSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
                .computed_tree_decomposition()
                .unwrap();
                let mut best = a;
                if best.max_bag_size > b.max_bag_size {
                    best = b;
                }
                if best.max_bag_size > c.max_bag_size {
                    best = c;
                }
                Some(best)
            }
            UpperboundHeuristicType::Custom(decomposer) => {
                decomposer(&graph, lowerbound, graph.order() - 1).computed_tree_decomposition()
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum AtomSolverType {
    None,
    MinFill,
    MinDegree,
    MinFillDegree,
    Tamaki,
    Custom(DynamicExact),
}

impl Default for AtomSolverType {
    fn default() -> Self {
        Self::Tamaki
    }
}

impl AtomSolverType {
    pub(crate) fn compute(
        &self,
        sub_graph: &HashMapGraph,
        lowerbound: usize,
        upperbound: usize,
    ) -> ComputationResult {
        match self {
            AtomSolverType::None => ComputationResult::Bounds(Bounds {
                lowerbound,
                upperbound,
            }),
            AtomSolverType::MinFill => {
                HeuristicEliminationDecomposer::<MinFillSelector>::with_bounds(
                    sub_graph, lowerbound, upperbound,
                )
                .compute()
            }
            AtomSolverType::MinDegree => {
                HeuristicEliminationDecomposer::<MinDegreeSelector>::with_bounds(
                    sub_graph, lowerbound, upperbound,
                )
                .compute()
            }
            AtomSolverType::MinFillDegree => {
                HeuristicEliminationDecomposer::<MinFillDegreeSelector>::with_bounds(
                    sub_graph, lowerbound, upperbound,
                )
                .compute()
            }
            AtomSolverType::Tamaki => {
                TamakiPid::with_bounds(sub_graph, lowerbound, upperbound).compute()
            }
            AtomSolverType::Custom(solver) => solver(sub_graph, lowerbound, upperbound),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct AlgorithmTypes {
    pub atom_solver: AtomSolverType,
    pub upperbound: UpperboundHeuristicType,
    pub lowerbound: LowerboundHeuristicType,
}

impl AlgorithmTypes {
    impl_setter!(self, atom_solver, AtomSolverType);
    impl_setter!(self, upperbound, UpperboundHeuristicType);
    impl_setter!(self, lowerbound, LowerboundHeuristicType);
}

pub struct Solver {
    algorithm_types: AlgorithmTypes,
    safe_separator_limits: SafeSeparatorLimits,
    apply_reduction_rules: bool,
    use_atom_width_as_lower_bound: bool,
    seed: Option<u64>,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            algorithm_types: AlgorithmTypes::default(),
            safe_separator_limits: SafeSeparatorLimits::default(),
            apply_reduction_rules: true,
            use_atom_width_as_lower_bound: true,
            seed: None,
        }
    }
}

impl Solver {
    pub fn default_heuristic() -> Self {
        Self {
            algorithm_types: AlgorithmTypes::default().atom_solver(AtomSolverType::None),
            safe_separator_limits: SafeSeparatorLimits::default(),
            apply_reduction_rules: true,
            use_atom_width_as_lower_bound: true,
            seed: None,
        }
    }

    pub fn auto(graph: &HashMapGraph) -> Self {
        match graph.order() {
            0..=3000 => {
                Self::default_heuristic().algorithm_types(AlgorithmTypes::default().atom_solver(
                    AtomSolverType::Custom(|graph, lowerbound, upperbound| {
                        if graph.order() <= 150 {
                            #[cfg(feature = "log")]
                            info!("Attempting to solve atom exactly");
                            TamakiPid::with_bounds(graph, lowerbound, upperbound).compute()
                        } else {
                            #[cfg(feature = "log")]
                            info!("Atom too large to be solved exactly");

                            ComputationResult::Bounds(Bounds {
                                lowerbound,
                                upperbound,
                            })
                        }
                    }),
                ))
            }
            3001..=10000 => Self::default_heuristic(),
            10001..=50000 => {
                let lowerbound = LowerboundHeuristicType::None;
                let upperbound = UpperboundHeuristicType::MinDegree;
                let atom_solver = AtomSolverType::None;
                let algorithm_types = AlgorithmTypes {
                    atom_solver,
                    upperbound,
                    lowerbound,
                };
                let limits = SafeSeparatorLimits::only_cut_vertex();
                Self::default()
                    .algorithm_types(algorithm_types)
                    .safe_separator_limits(limits)
                    .apply_reduction_rules(true)
            }
            _ => {
                let lowerbound = LowerboundHeuristicType::None;
                let upperbound = UpperboundHeuristicType::MinDegree;
                let atom_solver = AtomSolverType::None;
                let algorithm_types = AlgorithmTypes {
                    atom_solver,
                    upperbound,
                    lowerbound,
                };
                let limits = SafeSeparatorLimits::skip_all();
                Self::default()
                    .algorithm_types(algorithm_types)
                    .safe_separator_limits(limits)
                    .apply_reduction_rules(false)
            }
        }
    }

    pub fn default_exact() -> Self {
        Self::default()
    }

    impl_setter!(self, algorithm_types, AlgorithmTypes);
    impl_setter!(self, safe_separator_limits, SafeSeparatorLimits);
    impl_setter!(self, apply_reduction_rules, bool);
    impl_setter!(self, use_atom_width_as_lower_bound, bool);
    impl_setter!(self, seed, Option<u64>);

    pub fn solve(&self, graph: &HashMapGraph) -> TreeDecomposition {
        #[cfg(feature = "log")]
        info!("attempting to solve graph with {} vertices", graph.order());
        let mut td = TreeDecomposition::default();
        if graph.order() == 0 {
            return td;
        } else if graph.order() <= 2 {
            td.add_bag(graph.vertices().collect());
            return td;
        } else {
            let mut lowerbound = 0;
            let components = graph.connected_components();
            #[cfg(feature = "log")]
            info!("obtained {} components", components.len());
            if components.len() > 1 {
                td.add_bag(Default::default());
            }
            'outer: for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
                if sub_graph.order() <= 2 {
                    let idx = td.add_bag(sub_graph.vertices().collect());
                    if td.bags.len() > 1 {
                        td.add_edge(0, idx);
                    }
                    continue;
                }
                let mut star_solver = StarSolver::from(sub_graph.clone());
                loop {
                    match star_solver.step() {
                        StarResult::Solver(solver) => {
                            star_solver = solver;
                        }
                        StarResult::Finished(tmp) => {
                            if td.bags.len() == 0 {
                                td = tmp;
                            } else {
                                td.combine_with(0, tmp);
                            }
                            continue 'outer;
                        }
                    }
                }

                #[cfg(feature = "log")]
                info!(
                    "attempting to solve subgraph with {} vertices",
                    sub_graph.order()
                );
                if sub_graph.order() <= 2 {
                    let idx = td.add_bag(sub_graph.vertices().collect());
                    if td.bags.len() > 1 {
                        td.add_edge(0, idx);
                    }
                    continue;
                }
                lowerbound = max(lowerbound, 1);

                let reducer: Option<_> = if self.apply_reduction_rules {
                    #[cfg(feature = "log")]
                    info!("applying reduction rules");
                    let mut tmp = RuleBasedPreprocessor::new(&sub_graph);
                    tmp.preprocess();

                    #[cfg(feature = "log")]
                    info!("reduced graph to: {}", tmp.graph().order());

                    if tmp.graph().order() <= lowerbound + 1 {
                        td.combine_with_or_replace(0, tmp.into_td());
                        continue;
                    }
                    Some(tmp)
                } else {
                    None
                };

                let (reduced_graph, new_lowerbound) = match reducer.as_ref() {
                    None => (&sub_graph, 0),
                    Some(reducer) => (reducer.graph(), reducer.lower_bound),
                };
                lowerbound = max(lowerbound, new_lowerbound);

                #[cfg(feature = "log")]
                info!("solving reduced graph");

                let result = SafeSeparatorFramework::default()
                    .algorithms(self.algorithm_types)
                    .seed(self.seed)
                    .safe_separator_limits(self.safe_separator_limits)
                    .use_atom_width_as_lower_bound(self.use_atom_width_as_lower_bound)
                    .compute(reduced_graph, lowerbound);
                lowerbound = max(lowerbound, result.lowerbound);
                let mut partial_td = result.tree_decomposition;
                partial_td.flatten();
                if self.use_atom_width_as_lower_bound {
                    lowerbound = max(lowerbound, partial_td.max_bag_size - 1);
                }
                match partial_td.verify(reduced_graph) {
                    Ok(_) => {
                        #[cfg(feature = "log")]
                        info!("partial td computed after reduction rules is valid!");
                    }
                    Err(e) => {
                        panic!(
                            " partial td computed after reduction rules is invalid: {}",
                            e
                        );
                    }
                }
                match reducer {
                    None => td.combine_with_or_replace(0, partial_td),
                    Some(reducer) => {
                        td.combine_with_or_replace(0, reducer.combine_into_td(partial_td))
                    }
                };
            }
        }
        td
    }
}

pub trait AtomSolver {
    fn with_graph(graph: &HashMapGraph) -> Self
    where
        Self: Sized;
    fn with_bounds(graph: &HashMapGraph, lowerbound: usize, upperbound: usize) -> Self
    where
        Self: Sized;
    fn compute(self) -> ComputationResult;
}

pub struct Bounds {
    pub lowerbound: usize,
    pub upperbound: usize,
}

pub enum ComputationResult {
    ComputedTreeDecomposition(TreeDecomposition),
    Bounds(Bounds),
}

impl ComputationResult {
    pub fn computed_tree_decomposition(self) -> Option<TreeDecomposition> {
        match self {
            ComputationResult::ComputedTreeDecomposition(td) => Some(td),
            _ => None,
        }
    }
}

#[derive(Default)]
struct StarState {
    lowerbound:Rc<Cell<usize>>,
    level: SeparationLevel,
    graph: HashMapGraph,
    parent_state: Option<Rc<RefCell<StarState>>>,
    td: TreeDecomposition,
    number_of_unprocessed_children: usize,
}

impl StarState {
    fn pull_up(self) -> Option<TreeDecomposition> {
        let td = self.td;
        let parent = self.parent_state;
        if parent.is_none() {
            return Some(td);
        }
        let number_of_unprocessed_children;
        let parent = parent.unwrap();
        {
            let mut parent: RefMut<_> = parent.borrow_mut();
            parent.td.combine_with_or_replace(0, td);
            parent.number_of_unprocessed_children-=1;
            number_of_unprocessed_children = parent.number_of_unprocessed_children;
        }

        if number_of_unprocessed_children == 0 {
            let parent = parent.replace(Default::default());
            return parent.pull_up();
        }
        /*if let Some(mut parent) = parent {
            let mut parent: RefMut<_> = parent.borrow_mut();
            parent.td.combine_with_or_replace(0, td);
            parent.number_of_unprocessed_children-=1;
            /*if parent.number_of_unprocessed_children == 0 {
                return parent.pull_up();
            }*/
        } else if self.number_of_unprocessed_children == 0 { // finished
            return Some(td)
        }*/
        None
    }
}

impl PartialOrd for StarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Option::from(self.graph.order().cmp(&other.graph.order()))
    }
}

impl Ord for StarState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.graph.order().cmp(&other.graph.order())
    }
}

impl PartialEq for StarState {
    fn eq(&self, other: &Self) -> bool {
        self.graph.order().eq(&other.graph.order())
    }
}

impl Eq for StarState {}


pub struct StarSolver {
    queue: Vec<Rc<RefCell<StarState>>>,
    seed: Option<u64>,
}


impl From<HashMapGraph> for StarSolver {
    fn from(graph: HashMapGraph) -> Self {
        let mut tmp = Self {
            queue: Default::default(),
            seed: None,
        };
         let state = StarState {
            lowerbound: Rc::new(Cell::new(0)),
            level: SeparationLevel::Connected,
            graph,
            td: Default::default(),
            parent_state: Default::default(),
            number_of_unprocessed_children: 0
        };
        let state = Rc::new(RefCell::new(state));
        tmp.queue.push(state);
        tmp
    }
}

const NO_LIMITS: SafeSeparatorLimits = SafeSeparatorLimits {
    size_one_separator: usize::MAX,
    size_two_separator: usize::MAX,
    size_three_separator: usize::MAX,
    clique_separator: usize::MAX,
    almost_clique_separator: usize::MAX,
    minor_safe_separator: usize::MAX,
    minor_safe_separator_max_missing: usize::MAX,
    minor_safe_separator_tries: 25,
    check_again_before_atom: false,
    use_min_degree_for_minor_safe: false
};

pub enum StarResult {
    Solver(StarSolver),
    Finished(TreeDecomposition),
}

impl<'a> StarSolver {
    fn atom(self, state: Rc<RefCell<StarState>>) -> StarResult {
        let mut state = state.replace(Default::default());
        let pid_result = TamakiPid::with_graph(&state.graph).compute();
        let td = match pid_result {
            ComputationResult::ComputedTreeDecomposition(td) => td,
            ComputationResult::Bounds(_) => {
                TreeDecomposition::with_root(state.graph.vertices().collect())
            },
        };
        state.td = td;

        match state.pull_up() {
            None => StarResult::Solver(self),
            Some(td) => StarResult::Finished(td),
        }
    }

    fn fork(mut self, state: Rc<RefCell<StarState>>, separator: FxHashSet<usize>) -> StarResult {
        let connected_components;
        let lowerbound;
        let level;
        {
            let mut tmp: RefMut<_> = state.borrow_mut();
            connected_components = tmp.graph.separate(&separator);
            tmp.number_of_unprocessed_children = connected_components.len();
            tmp.td.add_bag(separator.clone());
            lowerbound = tmp.lowerbound.clone();
            level = tmp.level.clone();
        }
        
        for cc in connected_components {
            let mut graph = state.borrow().graph.vertex_induced(&cc);
            for v in separator.iter() {
                for u in separator.iter().filter(|u| v < *u) {
                    graph.add_edge(*v, *u);
                }
                for u in state.borrow().graph
                    .neighborhood_set(*v)
                    .iter()
                    .filter(|u| cc.contains(*u))
                {
                    graph.add_edge(*v, *u);
                }
            }

            let next_star_state = StarState {
                lowerbound: lowerbound.clone(),
                level,
                graph,
                parent_state: Option::from(state.clone()),
                td: Default::default(),
                number_of_unprocessed_children: 0
            };
            self.queue.push(Rc::new(RefCell::new(next_star_state)));
        }
        StarResult::Solver(self)
    }

    pub fn step(mut self) -> StarResult {
        let mut state= self.queue.pop().unwrap();
        let order;
        let lowerbound;
        let level;
        {
            let tmp: Ref<_> = state.borrow();
            order = tmp.graph.order();
            lowerbound = tmp.lowerbound.get();
            level = tmp.level;
        }
        if order <= lowerbound {
            self.atom(state)
        } else {
            let result;
            {
                let level = state.borrow().level;
                result = level.find_separator(&state.borrow().graph, &NO_LIMITS, self.seed);
            }
            match result {
                SeparatorSearchResult::Some(separator) => {
                    self.fork(state, separator)
                }
                _ => {
                    let level;
                    {
                        let star_state: Ref<_> = state.borrow();
                        level = star_state.level.increment();
                    }
                    match level {
                        None => {
                            self.atom(state)
                        }
                        Some(next_level) => {
                            {
                                state.borrow_mut().level = next_level;
                            }
                            self.queue.push(state);
                            StarResult::Solver(self)
                        }
                    }
                }
            }
        }
    }
}