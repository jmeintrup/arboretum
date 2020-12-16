use crate::exact::pid::PID;
use crate::exact::ExactSolver;
use crate::graph::bag::TreeDecomposition;
use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::lowerbound::LowerboundHeuristic;
use crate::preprocessing::{Preprocessor, RuleBasedPreprocessor, SafeSeparatorFramework};
use crate::upperbound::UpperboundHeuristic;

pub struct SolverBuilder {
    apply_reduction_rules: bool,
    apply_safe_separator_decomposition: bool,
}

impl SolverBuilder {
    pub fn new() -> Self {
        Self {
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
            apply_reduction_rules: self.apply_reduction_rules,
            apply_safe_separator_decomposition: self.apply_safe_separator_decomposition,
        }
    }
}

pub struct Solver {
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
                } else {
                    let mut reducer = RuleBasedPreprocessor::new(&sub_graph);
                    reducer.preprocess();
                    let reduced_graph = reducer.graph();
                    if reduced_graph.order() == 0 {
                        td.combine_with_or_replace(0, reducer.into_td())
                    } else {
                        let framework = SafeSeparatorFramework::new(reduced_graph.clone(), 4);
                        let result = framework.compute();
                        let partial_td = result.tree_decomposition;
                        td.combine_with_or_replace(0, reducer.combine_into_td(partial_td));
                    }
                }
            }
        }
        td
    }
}
