use crate::graph::{BaseGraph, HashMapGraph, MutableGraph};
use crate::heuristic_elimination_order::{
    EliminationOrderDecomposer, PermutationDecompositionResult,
};
use crate::solver::{AtomSolver, Bounds, ComputationResult};
use crate::tree_decomposition::TreeDecomposition;
use fxhash::{FxHashMap, FxHashSet};
use std::cmp::max;

const DEFAULT_EPOCHS: usize = 10;
const DEFAULT_STEPS: usize = 10;
const DEFAULT_SEED: usize = 1337;
const DEFAULT_MAX_TABU: usize = 7;

pub struct TabuLocalSearch {
    graph: HashMapGraph,
    epochs: usize,
    steps: usize,
    best: Option<PermutationDecompositionResult>,
    upperbound: usize,
    lowerbound: usize,
    seed: usize,
    tabu: Vec<usize>,
    max_tabu_size: usize,
}

impl TabuLocalSearch {
    fn fitness(&self, permutation: &[usize]) -> f64 {
        let vertex_to_position: FxHashMap<_, _> = permutation
            .iter()
            .enumerate()
            .map(|(i, v)| (*v, i))
            .collect();

        let mut working_graph = self.graph.clone();

        let mut max_degree = 0;
        let mut max_bag = 0;
        let mut fitness = 0.0;
        let m = permutation.len();
        for (v_pos, v) in permutation.iter().enumerate() {
            max_degree = max(max_degree, working_graph.degree(*v));
            let bag: Vec<_> = working_graph
                .neighborhood_set(*v)
                .iter()
                .filter(|u| {
                    let u_pos = *vertex_to_position.get(*u).unwrap();
                    u_pos > v_pos
                })
                .copied()
                .collect();
            max_bag = max(max_bag, bag.len());
            fitness += bag.len() * bag.len();
            for a in &bag {
                for b in bag.iter().filter(|b| a < *b) {
                    working_graph.add_edge(*a, *b);
                }
            }
        }
        fitness + max_bag.pow(2) * m.pow(2)
    }

    pub fn new(graph: HashMapGraph) -> Self {
        let upperbound = graph.order() - 1;
        Self {
            graph,
            epochs: DEFAULT_EPOCHS,
            steps: DEFAULT_STEPS,
            best: None,
            upperbound,
            lowerbound: 0,
            seed: DEFAULT_SEED,
            tabu: Default::default(),
            max_tabu_size: DEFAULT_MAX_TABU,
        }
    }

    pub fn max_tabu_size(mut self, max_tabu_size: usize) -> Self {
        self.max_tabu_size = max_tabu_size;
        self
    }

    pub fn lowerbound(mut self, lowerbound: usize) -> Self {
        self.lowerbound = lowerbound;
        self
    }

    pub fn upperbound(mut self, upperbound: usize) -> Self {
        self.upperbound = upperbound;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    pub fn seed(mut self, seed: usize) -> Self {
        self.seed = seed;
        self
    }

    pub fn best_permutation(mut self, best_permutation: PermutationDecompositionResult) -> Self {
        self.best_permutation = best_permutation;
        self
    }
}

impl AtomSolver for TabuLocalSearch {
    fn with_graph(graph: &HashMapGraph) -> Self
    where
        Self: Sized,
    {
        Self::new(graph.clone())
    }

    fn with_bounds(graph: &HashMapGraph, lowerbound: usize, upperbound: usize) -> Self
    where
        Self: Sized,
    {
        Self::new(graph.clone())
            .lowerbound(lowerbound)
            .upperbound(upperbound)
    }

    fn compute(mut self) -> ComputationResult {
        if self.best_permutation.is_none() {
            self.best_permutation = EliminationOrderDecomposer::new(
                self.graph.clone(),
                self.graph.vertices().collect(),
            )
            .compute();
        }

        for epoch in 0..self.epochs {

            let mut working_permutation;
            let mut working_bag_map;
            let mut working_td;
            if epoch == 0 {
                let tmp = self.best.as_ref().unwrap();
                working_permutation = tmp.permutation.clone();
                working_bag_map = tmp.eliminated_in_bag.clone();
                working_td = tmp.tree_decomposition.clone();
            } else {
                let tmp = EliminationOrderDecomposer::new(
                    self.graph.clone(),
                    working_permutation.unwrap().clone(),
                )
                .compute();
                working_permutation = tmp.permutation.clone();
                working_bag_map = tmp.eliminated_in_bag.clone();
                working_td = tmp.tree_decomposition.clone();
            }


            let mut working_fitness = self.fitness(&working_permutation);
            let vertex_to_position: FxHashMap<_, _> = working_permutation
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, i))
                .collect();



        }

        ComputationResult::Bounds(Bounds {
            upperbound: self.upperbound,
            lowerbound: self.lowerbound,
        })
    }
}
