/*use crate::graph::hash_map_graph::HashMapGraph;
use crate::preprocessing::{RuleBasedPreprocessor, SafeSeparatorFramework, Preprocessor};
use crate::upperbound::{HeuristicEliminationOrderDecomposer, UpperboundHeuristic};
use crate::graph::mutable_graph::MutableGraph;
use crate::graph::bag::TreeDecomposition;
use crate::exact::ExactSolver;
use crate::graph::graph::Graph;
use crate::lowerbound::LowerboundHeuristic;
use std::ops::Deref;

#[derive(Debug, Clone)]
struct Atom {
    graph: HashMapGraph,
    target_bag: usize,
}

#[derive(Debug, Clone)]
struct SolvedAtom {
    atom: Atom,
    tree_decomposition: TreeDecomposition,
}

impl SolvedAtom {
    fn new(atom: Atom, tree_decomposition: TreeDecomposition) -> Self {
        Self {
            atom,
            tree_decomposition,
        }
    }
}

pub trait AtomSolver<G: Graph> {
    fn compute(self, tree_decomposition: TreeDecomposition, atoms: &[Atom]) -> Vec<SolvedAtom>;
}

pub trait PreprocessorBuilder {
    fn build(graph: HashMapGraph) -> Box<dyn Preprocessor>;
}

pub trait AtomSolverBuilder {
    fn build(graph: HashMapGraph) -> Box<dyn AtomSolver<HashMapGraph>>;
}

pub trait UpperboundHeuristicBuilder {
    fn build(graph: HashMapGraph) -> Box<dyn UpperboundHeuristic>;
}

pub trait LowerboundHeuristicBuilder {
    fn build(graph: HashMapGraph) -> Box<dyn LowerboundHeuristic>;
}

pub struct Solver {
    graph: HashMapGraph,
    splitter: SafeSeparatorFramework,
    preprocessor_builder: Box<dyn PreprocessorBuilder>,
    atom_solver_builder: Box<dyn ExactSolverBuilder>,
}

impl Solver {
    fn solve(mut self) -> TreeDecomposition {
        let mut preprocessor: Box<dyn Preprocessor> = self.preprocessor_builder.build(self.graph);
        preprocessor.preprocess();
        let reduced_graph = preprocessor.graph();
        if reduced_graph.order() == 0 {
            return preprocessor.into_td();
        }

    }
}
*/