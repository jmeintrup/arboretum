use crate::graph::mutable_graph::MutableGraph;
use std::cmp::max;
use std::collections::HashSet;

pub fn minor_min_width<G>(graph: &G) -> usize
where
    G: MutableGraph,
{
    let mut lb = graph.degree(
        graph
            .min_vertex_by(|v, u| graph.degree(*v).cmp(&graph.degree(*u)))
            .unwrap(),
    );
    let mut cloned = graph.clone();
    while cloned.order() > max(lb, 2) {
        let first = cloned
            .min_vertex_by(|u, v| cloned.degree(*u).cmp(&cloned.degree(*v)))
            .unwrap();
        let second = cloned
            .neighborhood(first)
            .min_by(|u, v| cloned.degree(*u).cmp(&cloned.degree(*v)))
            .unwrap();
        // from second to first
        cloned.contract(second, first);
        lb = max(lb, cloned.degree(first));
    }
    lb
}
