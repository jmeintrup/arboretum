use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use std::cmp::max;
use std::collections::HashSet;

pub trait LowerboundHeuristic {
    fn compute<G: MutableGraph>(graph: &G) -> usize;
}

pub struct MinorMinWidth {}

impl LowerboundHeuristic for MinorMinWidth {
    fn compute<G: MutableGraph>(graph: &G) -> usize {
        let mut graph = graph.clone();
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
