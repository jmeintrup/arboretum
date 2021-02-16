use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use std::cmp::max;
use std::collections::HashSet;

pub trait LowerboundHeuristic {
    fn with_graph(graph: &HashMapGraph) -> Self
    where
        Self: Sized;
    fn compute(self) -> usize;
}

pub struct MinorMinWidth {
    graph: HashMapGraph,
}

impl LowerboundHeuristic for MinorMinWidth {
    fn with_graph(graph: &HashMapGraph) -> Self {
        Self {
            graph: graph.clone(),
        }
    }

    fn compute(self) -> usize {
        let mut graph = self.graph;
        let mut lb = 0;
        while graph.order() > 0 {
            if let Some(v) = graph
                .vertices()
                .filter(|v| graph.degree(*v) > 0)
                .min_by(|v, u| graph.degree(*v).cmp(&graph.degree(*u)))
            {
                lb = max(lb, graph.degree(v));
                let u = graph
                    .neighborhood(v)
                    .min_by(|v, u| graph.degree(*v).cmp(&graph.degree(*u)))
                    .unwrap();
                graph.contract(v, u);
            } else {
                break;
            }
        }
        lb
    }
}
