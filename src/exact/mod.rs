use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::tree_decomposition::TreeDecomposition;

pub mod tamakipid;

pub trait ExactSolver<G: Graph> {
    fn with_graph(graph: &G) -> Self;
    fn with_bounds(graph: &G, lowerbound: usize, upperbound: usize) -> Self;
    fn compute_exact(self) -> Result<TreeDecomposition, ()>;
}
