use crate::graph::{BaseGraph, HashMapGraph, MutableGraph};
use crate::heuristic_elimination_order::{
    EliminationOrderDecomposer, PermutationDecompositionResult,
};
use crate::solver::{AtomSolver, Bounds, ComputationResult};
use fxhash::{FxHashMap, FxHashSet};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::max;

const DEFAULT_EPOCHS: usize = 10;
const DEFAULT_STEPS: usize = 10;
const DEFAULT_SEED: u64 = 1337;
const DEFAULT_MAX_TABU: usize = 7;

pub struct TabuLocalSearch {
    graph: HashMapGraph,
    epochs: usize,
    steps: usize,
    best: Option<PermutationDecompositionResult>,
    upperbound: usize,
    lowerbound: usize,
    seed: u64,
    max_tabu_size: usize,
}

fn permutation_map(permutation: &[usize]) -> FxHashMap<usize, usize> {
    permutation
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect()
}

impl TabuLocalSearch {
    fn fitness(&self, permutation: &[usize]) -> f64 {
        let vertex_to_position = permutation_map(permutation);

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
            fitness += bag.len().pow(2) as f64;
            for a in &bag {
                for b in bag.iter().filter(|b| a < *b) {
                    working_graph.add_edge(*a, *b);
                }
            }
        }
        fitness + max_bag.pow(2) as f64 * m.pow(2) as f64
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

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn best_permutation(mut self, best_permutation: PermutationDecompositionResult) -> Self {
        self.best = Option::from(best_permutation);
        self
    }

    fn swap(&self, permutation: &[usize], v_pos: usize, u_pos: usize) -> Vec<usize> {
        let mut tmp = Vec::from(permutation);
        tmp.swap(v_pos, u_pos);
        tmp
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
        assert!(self.max_tabu_size + 2 < self.graph.order());

        let mut rng: StdRng = SeedableRng::seed_from_u64(self.seed);
        if self.best.is_none() {
            self.best = Option::from(
                EliminationOrderDecomposer::new(
                    self.graph.clone(),
                    self.graph.vertices().collect(),
                )
                .compute(),
            );
        }

        let mut eval_opt = self.fitness(&self.best.as_ref().unwrap().permutation);
        let mut permutation = self.best.as_ref().unwrap().permutation.clone();
        let mut tabu: Vec<usize> = vec![];
        for _ in 0..self.epochs {
            let tmp =
                EliminationOrderDecomposer::new(self.graph.clone(), permutation.clone()).compute();

            let mut map = tmp.eliminated_in_bag;
            let mut pos = permutation_map(&permutation);
            let mut td = tmp.tree_decomposition;
            let mut eval = self.fitness(&permutation);

            for _ in 0..self.steps {
                let mut best_neighbor_perm: Option<Vec<usize>> = None;
                let mut best_neighbor: Option<usize> = None;
                let mut eval_tmp = f64::MAX;

                for v in &permutation {
                    if !tabu.contains(v) {
                        let mut minw = None;
                        let mut min = *pos.get(v).unwrap();
                        let mut maxw = None;
                        let mut max = *pos.get(v).unwrap();

                        for w in &td.bags[*map.get(v).unwrap()].vertex_set {
                            let pw = *pos.get(w).unwrap();
                            if pw < min {
                                minw = Some(w);
                                min = pw;
                            }
                            if pw > max {
                                maxw = Some(w);
                                max = pw;
                            }
                        }

                        if let Some(maxw) = maxw {
                            let perm_max = self.swap(&permutation, *pos.get(v).unwrap(), max);
                            let eval_max = self.fitness(&permutation);
                            if eval_max < eval_tmp {
                                eval_tmp = eval_max;
                                best_neighbor_perm = Some(perm_max);
                                best_neighbor = Some(*maxw);
                            }
                        }
                        if let Some(minw) = minw {
                            let perm_min = self.swap(&permutation, *pos.get(v).unwrap(), min);
                            let eval_min = self.fitness(&permutation);
                            if eval_min < eval_tmp {
                                eval_tmp = eval_min;
                                best_neighbor_perm = Some(perm_min);
                                best_neighbor = Some(*minw);
                            }
                        }
                    }
                }

                if eval_tmp < eval {
                    permutation = best_neighbor_perm.unwrap();
                    let tmp =
                        EliminationOrderDecomposer::new(self.graph.clone(), permutation.clone())
                            .compute();
                    map = tmp.eliminated_in_bag;
                    pos = permutation_map(&permutation);
                    td = tmp.tree_decomposition;
                    eval = self.fitness(&permutation);

                    tabu.push(best_neighbor.unwrap());
                    if tabu.len() > self.max_tabu_size {
                        tabu.remove(0);
                    }
                } else {
                    break;
                }
            }

            if eval < eval_opt {
                let tmp = EliminationOrderDecomposer::new(self.graph.clone(), permutation.clone())
                    .compute();
                if tmp.tree_decomposition.max_bag_size
                    < self.best.as_ref().unwrap().tree_decomposition.max_bag_size
                {
                    self.best = Some(tmp);
                    eval_opt = eval;
                }
            }

            let mut choices: FxHashSet<_> = permutation.iter().copied().collect();
            for v in &tabu {
                choices.remove(v);
            }
            assert!(choices.len() > 1);
            let mut choices: Vec<_> = choices.iter().copied().collect();
            let i = rng.gen_range(0, choices.len());
            let v = choices[i];
            let v_pos = *pos.get(&v).unwrap();
            choices.swap_remove(i);

            let j = rng.gen_range(0, choices.len());
            let u = choices[j];
            let u_pos = *pos.get(&u).unwrap();

            permutation = self.swap(&permutation, v_pos, u_pos);
        }
        let td = self.best.unwrap().tree_decomposition;
        if td.max_bag_size - 1 < self.upperbound {
            ComputationResult::ComputedTreeDecomposition(td)
        } else {
            ComputationResult::Bounds(Bounds {
                lowerbound: self.lowerbound,
                upperbound: self.upperbound,
            })
        }
    }
}
