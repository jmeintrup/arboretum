use crate::exact::tamakipid::TamakiPid;
use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::tree_decomposition::TreeDecomposition;
use crate::heuristic_elimination_order::{
    heuristic_elimination_decompose, HeuristicEliminationDecomposer, MinDegreeDecomposer,
    MinDegreeSelector, MinFillDecomposer, MinFillDegree, MinFillDegreeSelector, MinFillSelector,
    Selector,
};
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::preprocessing::{Preprocessor, RuleBasedPreprocessor, SafeSeparatorFramework};
use std::array;
use std::cmp::max;
use std::hash::Hash;
use std::process::exit;

macro_rules! impl_setter {
    ($self:ident, $field:ident, $type:ty) => {
        pub fn $field(mut $self, $field: $type) -> Self {
            $self.$field = $field;
            $self
        }
    }
}

pub trait DynamicUpperboundHeuristic: AtomSolver {}
impl<S: Selector> DynamicUpperboundHeuristic for HeuristicEliminationDecomposer<S> {}

pub type Lowerbound = usize;
pub type Upperbound = usize;
pub type DynamicUpperbound =
    fn(&HashMapGraph, Lowerbound, Upperbound) -> Result<TreeDecomposition, ()>;
pub type DynamicExact = fn(&HashMapGraph, Lowerbound, Upperbound) -> Result<TreeDecomposition, ()>;
pub type DynamicLowerbound = fn(&HashMapGraph) -> usize;

pub enum LowerboundHeuristicType {
    None,
    MinorMinWidth,
    Custom(DynamicLowerbound),
}

impl LowerboundHeuristicType {
    fn compute(&self, graph: &HashMapGraph) -> Lowerbound {
        match self {
            LowerboundHeuristicType::None => {
                graph.vertices().map(|v| graph.degree(v)).min().unwrap_or(0)
            }
            LowerboundHeuristicType::MinorMinWidth => MinorMinWidth::with_graph(&graph).compute(),
            LowerboundHeuristicType::Custom(heuristic) => heuristic(&graph),
        }
    }
}

pub enum UpperboundHeuristicType {
    None,
    MinFill,
    MinDegree,
    MinFillDegree,
    All,
    Custom(DynamicUpperbound),
}

impl UpperboundHeuristicType {
    fn compute(&self, graph: &HashMapGraph, lowerbound: usize) -> Option<TreeDecomposition> {
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
                if best.max_bag_size < b.max_bag_size {
                    best = b;
                }
                if best.max_bag_size < c.max_bag_size {
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

struct SafeSeparatorParameters {
    size_one_separator_limit: Option<usize>,
    size_two_separator_limit: Option<usize>,
    size_three_separator_limit: Option<usize>,
    clique_separator_limit: Option<usize>,
    almost_clique_separator_limit: Option<usize>,
    minor_safe_separator_limit: Option<usize>,
    minor_safe_separator_tries: usize,
    minor_safe_separator_max_missing_edges: Option<usize>,
}

impl Default for SafeSeparatorParameters {
    fn default() -> Self {
        Self {
            size_one_separator_limit: None,
            size_two_separator_limit: None,
            size_three_separator_limit: Some(250),
            clique_separator_limit: None,
            almost_clique_separator_limit: Some(250),
            minor_safe_separator_limit: None,
            minor_safe_separator_tries: 25,
            minor_safe_separator_max_missing_edges: None,
        }
    }
}

impl SafeSeparatorParameters {
    impl_setter!(self, size_one_separator_limit, Option<usize>);
    impl_setter!(self, size_two_separator_limit, Option<usize>);
    impl_setter!(self, size_three_separator_limit, Option<usize>);
    impl_setter!(self, clique_separator_limit, Option<usize>);
    impl_setter!(self, almost_clique_separator_limit, Option<usize>);
    impl_setter!(self, minor_safe_separator_limit, Option<usize>);
    impl_setter!(self, minor_safe_separator_tries, usize);
    impl_setter!(self, minor_safe_separator_max_missing_edges, Option<usize>);
}

pub enum AtomSolverType {
    None,
    MinFill,
    MinDegree,
    MinFillDegree,
    Tamaki,
    Custom(DynamicExact),
}

impl AtomSolverType {
    fn compute(
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

pub struct Solver {
    apply_reduction_rules: bool,
    apply_safe_separator_decomposition: bool,
    safe_separator_params: SafeSeparatorParameters,
    lowerbound_heuristic_type: LowerboundHeuristicType,
    upperbound_heuristic_type: UpperboundHeuristicType,
    atom_solver_type: AtomSolverType,
    use_atom_bag_size_for_lowerbound: bool,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            apply_reduction_rules: true,
            apply_safe_separator_decomposition: true,
            safe_separator_params: SafeSeparatorParameters::default(),
            lowerbound_heuristic_type: LowerboundHeuristicType::MinorMinWidth,
            upperbound_heuristic_type: UpperboundHeuristicType::All,
            atom_solver_type: AtomSolverType::Tamaki,
            use_atom_bag_size_for_lowerbound: true,
        }
    }
}

impl Solver {
    impl_setter!(self, atom_solver_type, AtomSolverType);
    impl_setter!(self, upperbound_heuristic_type, UpperboundHeuristicType);
    impl_setter!(self, lowerbound_heuristic_type, LowerboundHeuristicType);
    impl_setter!(self, apply_safe_separator_decomposition, bool);
    impl_setter!(self, apply_reduction_rules, bool);
    impl_setter!(self, use_atom_bag_size_for_lowerbound, bool);

    pub fn solve(&self, graph: &HashMapGraph) -> TreeDecomposition {
        let mut td = TreeDecomposition::new();
        if graph.order() == 0 {
            return td;
        } else if graph.order() <= 2 {
            td.add_bag(graph.vertices().collect());
            return td;
        } else {
            let mut lowerbound = 0;
            let components = graph.connected_components();
            if components.len() > 1 {
                td.add_bag(Default::default());
            }
            for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
                if sub_graph.order() <= 2 {
                    let idx = td.add_bag(sub_graph.vertices().collect());
                    if td.bags.len() > 1 {
                        td.add_edge(0, idx);
                    }
                    continue;
                }
                lowerbound = max(lowerbound, 1);

                let mut reducer: Option<_> = if self.apply_reduction_rules {
                    let mut tmp = RuleBasedPreprocessor::new(&sub_graph);
                    tmp.preprocess();
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

                let mut partial_td = if self.apply_safe_separator_decomposition {
                    let framework = SafeSeparatorFramework::new(reduced_graph.clone(), lowerbound);
                    let result = framework.compute();
                    lowerbound = max(lowerbound, result.lowerbound);
                    result.tree_decomposition
                } else {
                    lowerbound = max(
                        lowerbound,
                        self.lowerbound_heuristic_type.compute(&sub_graph),
                    );
                    let upperbound_td: Option<_> = self
                        .upperbound_heuristic_type
                        .compute(&sub_graph, lowerbound);
                    match upperbound_td {
                        None => {
                            match self.atom_solver_type.compute(
                                &sub_graph,
                                lowerbound,
                                sub_graph.order() - 1,
                            ) {
                                Ok(td) => td,
                                Err(_) => {
                                    let mut td = TreeDecomposition::new();
                                    td.add_bag(sub_graph.vertices().collect());
                                    td
                                }
                            }
                        }
                        Some(upperbound_td) => {
                            if upperbound_td.max_bag_size - 1 <= lowerbound {
                                upperbound_td
                            } else {
                                match self.atom_solver_type.compute(
                                    &sub_graph,
                                    lowerbound,
                                    upperbound_td.max_bag_size - 1,
                                ) {
                                    Ok(td) => td,
                                    Err(_) => upperbound_td,
                                }
                            }
                        }
                    }
                };
                partial_td.flatten();
                if self.use_atom_bag_size_for_lowerbound {
                    lowerbound = max(lowerbound, partial_td.max_bag_size - 1);
                }
                match reducer {
                    None => td.combine_with_or_replace(0, partial_td),
                    Some(reducer) => td.combine_with_or_replace(
                        0,
                        reducer.combine_into_td(partial_td, &sub_graph),
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
