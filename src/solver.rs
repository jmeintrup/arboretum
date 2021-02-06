use crate::exact::tamakipid::TamakiPid;
use crate::exact::ExactSolver;
use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::tree_decomposition::TreeDecomposition;
use crate::heuristic_elimination_order::{heuristic_elimination_decompose, MinFillSelector};
use crate::lowerbound::LowerboundHeuristic;
use crate::preprocessing::{Preprocessor, RuleBasedPreprocessor, SafeSeparatorFramework};
use std::process::exit;

pub struct SolverBuilder {
    heuristic: bool,
    apply_reduction_rules: bool,
    apply_safe_separator_decomposition: bool,
}

impl SolverBuilder {
    pub fn new() -> Self {
        Self {
            heuristic: false,
            apply_reduction_rules: true,
            apply_safe_separator_decomposition: true,
        }
    }

    pub fn heuristic() -> Self {
        Self {
            heuristic: true,
            apply_reduction_rules: true,
            apply_safe_separator_decomposition: true,
        }
    }

    pub fn reduction_rules(mut self, apply: bool) -> Self {
        self.apply_reduction_rules = apply;
        self
    }

    pub fn safe_separator_decomposition(mut self, apply: bool) -> Self {
        self.apply_safe_separator_decomposition = apply;
        self
    }

    pub fn build(self) -> Solver {
        Solver {
            heuristic: self.heuristic,
            apply_reduction_rules: self.apply_reduction_rules,
            apply_safe_separator_decomposition: self.apply_safe_separator_decomposition,
        }
    }
}

pub struct Solver {
    heuristic: bool,
    apply_reduction_rules: bool,
    apply_safe_separator_decomposition: bool,
}

impl Solver {
    pub fn solve(&self, graph: &HashMapGraph) -> TreeDecomposition {
        let mut td = TreeDecomposition::new();
        if graph.order() == 0 {
            return td;
        } else if graph.order() <= 2 {
            td.add_bag(graph.vertices().collect());
            return td;
        } else {
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
                } else if !self.heuristic {
                    let mut reducer = RuleBasedPreprocessor::new(&sub_graph);
                    reducer.preprocess();
                    let reduced_graph = reducer.graph();
                    if reduced_graph.order() == 0 {
                        td.combine_with_or_replace(0, reducer.into_td())
                    } else {
                        let framework = SafeSeparatorFramework::new(reduced_graph.clone(), 4);
                        let result = framework.compute();
                        let mut partial_td = result.tree_decomposition;
                        partial_td.flatten();
                        td.combine_with_or_replace(
                            0,
                            reducer.combine_into_td(partial_td, &sub_graph),
                        );
                    }
                } else {
                    let mut reducer = RuleBasedPreprocessor::new(&sub_graph);
                    reducer.preprocess();
                    let reduced_graph = reducer.graph();
                    if reduced_graph.order() == 0 {
                        td.combine_with_or_replace(0, reducer.into_td())
                    } else {
                        let mut partial_td =
                            heuristic_elimination_decompose::<MinFillSelector>(sub_graph.clone());
                        partial_td.flatten();
                        td.combine_with_or_replace(
                            0,
                            reducer.combine_into_td(partial_td, &sub_graph),
                        );
                    }
                }
            }
        }
        td.flatten();
        td
    }
}
