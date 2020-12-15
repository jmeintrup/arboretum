use crate::graph::mutable_graph::MutableGraph;
use std::cmp::max;
use std::collections::HashSet;

pub trait LowerboundHeuristic {
    fn compute(self) -> usize;
}

pub struct MinorMinWidth<G: MutableGraph> {
    graph: G
}

impl<G: MutableGraph> MinorMinWidth<G> {
    pub fn new(graph: G) -> Self {
        Self { graph }
    }
}

impl<G: MutableGraph> LowerboundHeuristic for MinorMinWidth<G> {
    fn compute(mut self) -> usize {
        let mut lb = self.graph.degree(
            self.graph
                .min_vertex_by(|v, u| self.graph.degree(*v).cmp(&self.graph.degree(*u)))
                .unwrap(),
        );
        while self.graph.order() > max(lb, 2) {
            let first = self.graph
                .min_vertex_by(|u, v| self.graph.degree(*u).cmp(&self.graph.degree(*v)))
                .unwrap();
            let second = self.graph
                .neighborhood(first)
                .min_by(|u, v| self.graph.degree(*u).cmp(&self.graph.degree(*v)))
                .unwrap();
            // from second to first
            self.graph.contract(second, first);
            lb = max(lb, self.graph.degree(first));
        }
        lb
    }
}
