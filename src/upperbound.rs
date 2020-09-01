use crate::graph::bag::TreeDecomposition;
use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use crate::util::{get_width, EliminationOrder, Stopper};
use fnv::{FnvHashMap, FnvHashSet};
use rand::prelude::*;
use std::cmp::max;

pub struct HeuristicDecomposer<G: MutableGraph, S: SelectionStrategy<G>> {
    graph: G,
    selection_strategy: S,
    bags: FnvHashMap<usize, FnvHashSet<usize>>,
    eliminated_at: FnvHashMap<usize, usize>,
    stack: Vec<usize>,
    tree_decomposition: TreeDecomposition,
}

impl<G: MutableGraph, S: SelectionStrategy<G>> HeuristicDecomposer<G, S> {
    pub fn new(graph: G, selection_strategy: S) -> Self {
        let capacity = graph.order();
        Self {
            graph,
            selection_strategy,
            bags: FnvHashMap::default(),
            eliminated_at: FnvHashMap::default(),
            stack: Vec::with_capacity(capacity),
            tree_decomposition: TreeDecomposition::new(),
        }
    }

    pub fn decompose(mut self) -> TreeDecomposition {
        let mut max_bag = 2;

        while self.graph.order() >= max_bag {
            let u = self.selection_strategy.next(&self.graph);
            let neighbors: FnvHashSet<usize> = self.graph.neighborhood(u).collect();
            max_bag = max(max_bag, neighbors.len() + 1);
            self.bags.insert(u, neighbors);
            self.stack.push(u);
            self.eliminated_at.insert(u, self.stack.len() - 1);
            self.graph.eliminate_vertex(u);
        }
        if self.graph.order() > 0 {
            let mut rest: FnvHashSet<usize> = self.graph.vertices().collect();
            let u = rest.iter().next().copied().unwrap();
            rest.remove(&u);
            self.bags.insert(u, rest);
            self.stack.push(u);
            self.eliminated_at.insert(u, self.stack.len() - 1);
        }

        for v in self.stack.iter().rev() {
            let mut nb = self.bags.remove(v).unwrap();
            let old_bag_id = match self
                .tree_decomposition
                .bags
                .iter()
                .find(|old_bag| old_bag.vertex_set.is_superset(&nb))
            {
                Some(old_bag) => Some(old_bag.id),
                None => None,
            };
            match old_bag_id {
                Some(old_bag_id) => {
                    nb.insert(*v);
                    let id = self.tree_decomposition.add_bag(nb);
                    self.tree_decomposition.add_edge(old_bag_id, id);
                }
                None => {
                    nb.insert(*v);
                    self.tree_decomposition.add_bag(nb);
                }
            }
        }

        self.tree_decomposition
    }
}

pub struct IterativeLocalSearch<'a, G: MutableGraph, S: Stopper> {
    graph: &'a G,
    stopper: S,
    best: Vec<usize>,
    fitness: Option<f64>,
    width: Option<usize>,
}

impl<'a, G: MutableGraph, S: Stopper> IterativeLocalSearch<'a, G, S> {
    pub fn new(graph: &'a G, stopper: S) -> Self {
        let mut initial_solution: Vec<usize> = graph.vertices().collect();
        let mut rng = thread_rng();
        initial_solution.shuffle(&mut rng);
        Self {
            graph,
            stopper,
            best: initial_solution,
            fitness: None,
            width: None,
        }
    }

    pub fn with_solution(graph: &'a G, stopper: S, initial_solution: Vec<usize>) -> Self {
        Self {
            graph,
            stopper,
            best: initial_solution,
            fitness: None,
            width: None,
        }
    }

    pub fn run(&mut self) -> () {
        let mut s1 = self.best.clone();
        let mut s1_fitness = None;

        self.stopper.init();
        while !self.stopper.stop() {
            let (s2, s2_fitness, s2_width) = self.construct(&s1);
            if s1_fitness.is_none() || s2_fitness > s1_fitness.unwrap() {
                s1 = s2.clone();
                s1_fitness = Some(s2_fitness);
            } else {
                s1 = self.best.clone();
                s1_fitness = self.fitness;
            }

            self.pertubate(s1.as_mut());

            if self.fitness.is_none() || s2_fitness > self.fitness.unwrap() {
                self.best = s2;
                self.fitness = Some(s2_fitness);
                self.width = Some(s2_width);
            }
        }
    }

    fn pertubate(&self, target: &mut [usize]) {
        let mut rng = thread_rng();
        for _ in 0..((target.len() as f64).ln().floor() as usize) {
            let i = rng.gen_range(0, target.len());
            let j = rng.gen_range(0, target.len());
            target.swap(i, j);
        }
    }

    fn construct(&self, old: &[usize]) -> (Vec<usize>, f64, usize) {
        let mut rng = thread_rng();
        let mut solution = Vec::from(old);

        let mut sum = 0;
        let mut max_degree = 0;
        let mut i = 0;
        let mut graph = self.graph.clone();
        for (idx, v) in solution.iter().copied().enumerate() {
            let degree = graph.degree(v);
            sum += degree;
            if max_degree < degree {
                max_degree = v;
                i = idx;
            }
            graph.eliminate_vertex(v)
        }
        let j = rng.gen_range(0, solution.len());

        solution.swap(i, j);
        let width = get_width(self.graph, &solution);

        let w = width as f64;
        let n = solution.len() as f64;
        let sum = sum as f64;
        let fitness = 1f64 / ((w * n * n) + (sum * n));
        (solution, fitness, width)
    }

    pub fn fitness(&self) -> Option<f64> {
        return self.fitness;
    }

    pub fn width(&self) -> Option<usize> {
        return self.width;
    }

    pub fn get_elimination_order(&self) -> Option<EliminationOrder> {
        match self.width {
            Some(width) => Some(EliminationOrder::new(self.best.clone(), width as u32)),
            None => None,
        }
    }
}

fn min_by<G: MutableGraph, S: SelectionStrategy<G> + Copy + Clone>(
    graph: &G,
    strategy: S,
) -> EliminationOrder {
    let mut cloned = graph.clone();
    let mut data = Vec::with_capacity(cloned.order());
    let mut width: u32 = 1;
    let m = if cloned.order() < 2 {
        0
    } else {
        cloned.order() - 2
    };
    while data.len() < m {
        let to_be_eliminated = strategy.next(&cloned);
        width = max(width, cloned.degree(to_be_eliminated) as u32);
        cloned.eliminate_vertex(to_be_eliminated);
        data.push(to_be_eliminated);
    }
    // order <= 2 the order does not matter, and the width does not change
    cloned.vertices().for_each(|v| {
        data.push(v);
    });

    EliminationOrder::new(data, width)
}

pub fn min_degree<G: MutableGraph>(graph: &G) -> EliminationOrder {
    min_by(graph, MinDegreeStrategy)
}

pub fn min_fill_degree<G: MutableGraph>(graph: &G, alpha: f64, beta: f64) -> EliminationOrder {
    assert!(alpha < 1f64);
    assert!(beta < 1f64);
    min_by(graph, MinFillDegreeStrategy { alpha, beta })
}

pub fn min_fill<G: MutableGraph>(graph: &G) -> EliminationOrder {
    min_by(graph, MinFillStrategy)
}

pub trait SelectionStrategy<G: Graph> {
    fn next(&self, graph: &G) -> usize;
}

#[derive(Clone, Copy)]
pub struct MinDegreeStrategy;

impl<G: Graph> SelectionStrategy<G> for MinDegreeStrategy {
    fn next(&self, graph: &G) -> usize {
        graph
            .min_vertex_by(|u, v| graph.degree(*u).cmp(&graph.degree(*v)))
            .unwrap()
    }
}

#[derive(Clone, Copy)]
pub struct MinFillStrategy;

impl<G: Graph> SelectionStrategy<G> for MinFillStrategy {
    fn next(&self, graph: &G) -> usize {
        graph
            .min_vertex_by(|u, v| graph.fill_in_count(*u).cmp(&graph.fill_in_count(*v)))
            .unwrap()
    }
}

#[derive(Clone, Copy)]
pub struct MinFillDegreeStrategy {
    pub alpha: f64,
    pub beta: f64,
}

impl<G: Graph> SelectionStrategy<G> for MinFillDegreeStrategy {
    fn next(&self, graph: &G) -> usize {
        graph
            .min_vertex_by(|u, v| {
                (self.alpha * graph.fill_in_count(*u) as f64 + self.beta * graph.degree(*u) as f64)
                    .partial_cmp(
                        &(self.alpha * graph.fill_in_count(*v) as f64
                            + self.beta * graph.degree(*v) as f64),
                    )
                    .unwrap()
            })
            .unwrap()
    }
}
