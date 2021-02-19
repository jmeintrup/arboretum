use crate::exact::TamakiPid;
use crate::graph::{Graph, HashMapGraph};
use crate::heuristic_elimination_order::{
    HeuristicEliminationDecomposer, MinDegreeSelector,
    MinFillDegreeSelector, MinFillSelector, Selector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::rule_based_reducer::RuleBasedPreprocessor;
use crate::safe_separator_framework::{SafeSeparatorFramework, SafeSeparatorLimits};
use crate::tree_decomposition::{TreeDecomposition};
#[cfg(feature = "log")]
use log::info;
use std::cmp::max;

pub trait DynamicUpperboundHeuristic: AtomSolver {}
impl<S: Selector> DynamicUpperboundHeuristic for HeuristicEliminationDecomposer<S> {}

pub type Lowerbound = usize;
pub type Upperbound = usize;
pub type DynamicUpperbound =
    fn(&HashMapGraph, Lowerbound, Upperbound) -> Result<TreeDecomposition, ()>;
pub type DynamicExact = fn(&HashMapGraph, Lowerbound, Upperbound) -> Result<TreeDecomposition, ()>;
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
                Some(decomposer.compute().unwrap())
            }
            UpperboundHeuristicType::MinDegree => {
                let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
                    HeuristicEliminationDecomposer::with_bounds(&graph, lowerbound, graph.order());
                Some(decomposer.compute().unwrap())
            }
            UpperboundHeuristicType::MinFillDegree => {
                let decomposer: HeuristicEliminationDecomposer<MinFillDegreeSelector> =
                    HeuristicEliminationDecomposer::with_bounds(&graph, lowerbound, graph.order());
                Some(decomposer.compute().unwrap())
            }
            UpperboundHeuristicType::All => {
                let a = HeuristicEliminationDecomposer::<MinFillSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
                .unwrap();
                let b = HeuristicEliminationDecomposer::<MinDegreeSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
                .unwrap();
                let c = HeuristicEliminationDecomposer::<MinFillDegreeSelector>::with_bounds(
                    &graph,
                    lowerbound,
                    graph.order(),
                )
                .compute()
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
                Some(decomposer(&graph, lowerbound, graph.order() - 1).unwrap())
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
    ) -> Result<TreeDecomposition, ()> {
        match self {
            AtomSolverType::None => Err(()),
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
    use_atom_bag_size_for_lowerbound: bool,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            algorithm_types: AlgorithmTypes::default(),
            safe_separator_limits: SafeSeparatorLimits::default(),
            apply_reduction_rules: true,
            use_atom_bag_size_for_lowerbound: true,
        }
    }
}

impl Solver {
    pub fn default_heuristic() -> Self {
        Self {
            algorithm_types: AlgorithmTypes::default().atom_solver(AtomSolverType::None),
            safe_separator_limits: SafeSeparatorLimits::default(),
            apply_reduction_rules: true,
            use_atom_bag_size_for_lowerbound: true,
        }
    }

    pub fn auto(graph: &HashMapGraph) -> Self {
        match graph.order() {
            0..=3000 => {
                Self::default_heuristic().algorithm_types(AlgorithmTypes::default().atom_solver(
                    AtomSolverType::Custom(|graph, lowerbound, upperbound| {
                        if graph.order() <= 150 {
                            #[cfg(feature = "log")]
                            info!(" Attempting to solve atom exactly");
                            TamakiPid::with_bounds(graph, lowerbound, upperbound).compute()
                        } else {
                            #[cfg(feature = "log")]
                            info!(" Atom too large to be solved exactly");
                            Err(())
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
    impl_setter!(self, use_atom_bag_size_for_lowerbound, bool);

    pub fn solve(&self, graph: &HashMapGraph) -> TreeDecomposition {
        #[cfg(feature = "log")]
        info!(
            "c attempting to solve graph with {} vertices",
            graph.order()
        );
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
            info!(" obtained {} components", components.len());
            if components.len() > 1 {
                td.add_bag(Default::default());
            }
            for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
                #[cfg(feature = "log")]
                info!(
                    "c attempting to solve subgraph with {} vertices",
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
                    info!(" applying reduction rules");
                    let mut tmp = RuleBasedPreprocessor::new(&sub_graph);
                    tmp.preprocess();

                    #[cfg(feature = "log")]
                    info!(" reduced graph to: {}", tmp.graph().order());

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
                info!(" solving reduced graph");

                let result = SafeSeparatorFramework::default()
                    .algorithms(self.algorithm_types)
                    .safe_separator_limits(self.safe_separator_limits)
                    .compute(reduced_graph, lowerbound);
                lowerbound = max(lowerbound, result.lowerbound);
                let mut partial_td = result.tree_decomposition;
                partial_td.flatten();
                if self.use_atom_bag_size_for_lowerbound {
                    lowerbound = max(lowerbound, partial_td.max_bag_size - 1);
                }
                match
                    partial_td.verify(reduced_graph) {
                    Ok(_) => {
                        #[cfg(feature = "log")]
                        info!(" partial td computed after reduction rules is valid!");
                    }
                    Err(e) => {
                        panic!(" partial td computed after reduction rules is invalid: {}", e);
                    }
                }
                match reducer {
                    None => td.combine_with_or_replace(0, partial_td),
                    Some(reducer) => td.combine_with_or_replace(
                        0,
                        reducer.combine_into_td(partial_td),
                    ),
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
    fn compute(self) -> Result<TreeDecomposition, ()>;
}
