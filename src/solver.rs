use crate::graph::hash_map_graph::HashMapGraph;
use crate::preprocessing::{RuleBasedPreprocessor, SafeSeparatorFramework};
use crate::upperbound::HeuristicDecomposer;

pub trait AtomSolver {}

pub struct Solver<A: AtomSolver> {
    graph: HashMapGraph,
    preprocessor: RuleBasedPreprocessor,
    splitter: SafeSeparatorFramework,
    atom_solver: A,
}
