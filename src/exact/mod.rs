use crate::graph::bag::TreeDecomposition;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::graph::Graph;

pub mod pid;

pub trait ExactSolver<G: Graph> {
    fn with_graph(graph: &G) -> Self;
    fn with_bounds(graph: &G, lowerbound: u32, upperbound: u32) -> Self;
    fn compute_exact(self) -> Result<TreeDecomposition,()>;
}
