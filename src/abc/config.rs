use crate::abc::candidate::{Candidate, WorkingWrapper};
use crate::graph::mutable_graph::MutableGraph;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

pub trait Config<S: Clone> {
    fn create_solution(&self, current_best: Option<&S>) -> S;
    fn evaluate_fitness(&self, solution: &S) -> f64;
    fn explore(&self, c: &[WorkingWrapper<S>], index: usize) -> Candidate<S>;
}

pub struct TreeWidthConfig<'a, G: MutableGraph> {
    graph: &'a G,
}

impl<'a, G: MutableGraph> TreeWidthConfig<'a, G> {
    pub fn new(graph: &'a G) -> Self {
        Self { graph }
    }
}

impl<'a, G: MutableGraph> Config<Vec<usize>> for TreeWidthConfig<'a, G> {
    fn create_solution(&self, current_best: Option<&Vec<usize>>) -> Vec<usize> {
        let mut rng = thread_rng();

        return match current_best {
            Some(current_best) => {
                let p: f64 = rng.gen();
                if p < 0.2f64 {
                    let mut solution: Vec<usize> = self.graph.vertices().collect();
                    solution.shuffle(&mut rng);
                    solution
                } else if p < 0.6f64 {
                    let solution: Vec<usize> = Vec::from(current_best.as_slice());
                    solution
                } else {
                    let mut solution: Vec<usize> = Vec::from(current_best.as_slice());
                    for _ in 0..((solution.len() as f64).ln().floor() as usize) {
                        let i = rng.gen_range(0, solution.len());
                        let j = rng.gen_range(0, solution.len());
                        solution.swap(i, j);
                    }
                    solution
                }
            }
            None => {
                let mut solution: Vec<usize> = self.graph.vertices().collect();
                solution.shuffle(&mut rng);
                solution
            }
        };
    }

    fn evaluate_fitness(&self, solution: &Vec<usize>) -> f64 {
        let mut graph = self.graph.clone();
        let degrees: Vec<usize> = solution
            .iter()
            .map(|v| {
                let degree = graph.degree(*v);
                graph.eliminate_vertex(*v);
                degree
            })
            .collect();
        let w = *degrees.iter().max().unwrap() as f64;
        let n = solution.len() as f64;
        let sum: usize = degrees.iter().sum();
        let sum = sum as f64;
        1f64 / ((w * n * n) + (sum * n))
    }

    fn explore(&self, c: &[WorkingWrapper<Vec<usize>>], index: usize) -> Candidate<Vec<usize>> {
        let mut solution = c[index].candidate.solution.clone();

        let i = {
            let mut max_degree = 0;
            let mut i = 0;
            let mut graph = self.graph.clone();
            for (idx, v) in solution.iter().copied().enumerate() {
                if max_degree < graph.degree(v) {
                    max_degree = v;
                    i = idx;
                }
                graph.eliminate_vertex(v)
            }
            i
        };
        let mut rng = thread_rng();
        let j = rng.gen_range(0, solution.len());

        solution.swap(i, j);
        let fitness = self.evaluate_fitness(&solution);

        Candidate { solution, fitness }
    }
}
