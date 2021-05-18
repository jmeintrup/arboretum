use crate::exact::{QuickBB, TamakiPid};
use crate::graph::{BaseGraph, HashMapGraph};
use crate::heuristic_elimination_order::{
    EliminationOrderDecomposer, HeuristicEliminationDecomposer, MinDegreeSelector,
    MinFillDegreeSelector, MinFillSelector, PermutationDecompositionResult, Selector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::meta_heuristics::TabuLocalSearch;
use crate::rule_based_reducer::RuleBasedPreprocessor;
use crate::safe_separator_framework::{SafeSeparatorFramework, SafeSeparatorLimits};
use crate::tree_decomposition::TreeDecomposition;
#[cfg(feature = "log")]
use log::info;
use std::cmp::max;

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
    TabuLocalSearch,
    TabuLocalSearchInfinite,
    QuickBB,
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
        seed: u64,
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
            AtomSolverType::TabuLocalSearch => {
                TabuLocalSearch::with_bounds(sub_graph, lowerbound, upperbound)
                    .seed(seed)
                    .compute()
            }
            AtomSolverType::TabuLocalSearchInfinite => {
                TabuLocalSearch::with_bounds(sub_graph, lowerbound, upperbound)
                    .seed(seed)
                    .epochs(usize::MAX)
                    .compute()
            }
            AtomSolverType::QuickBB => {
                QuickBB::with_bounds(sub_graph, lowerbound, upperbound).compute()
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
            algorithm_types: AlgorithmTypes::default().atom_solver(AtomSolverType::TabuLocalSearch),
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
            for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
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
