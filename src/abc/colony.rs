use crate::abc::builder::Builder;
use crate::abc::candidate::{Candidate, WorkingWrapper};
use crate::abc::config::Config;
use rand::Rng;
use std::time::{Duration, SystemTime};

pub struct Colony<S: Clone, C: Config<S>> {
    builder: Builder<S, C>,
    working: Vec<WorkingWrapper<S>>,
    best: Candidate<S>,
}

impl<S: Clone, C: Config<S>> Colony<S, C> {
    pub(crate) fn new(builder: Builder<S, C>) -> Self {
        let working: Vec<WorkingWrapper<S>> = (0..builder.num_workers)
            .map(|_| {
                let solution = builder.config.create_solution(None);
                let fitness = builder.config.evaluate_fitness(&solution);
                WorkingWrapper::new(Candidate { solution, fitness }, builder.num_retries)
            })
            .collect();
        let best = working
            .iter()
            .max_by(|a, b| {
                a.candidate
                    .fitness
                    .partial_cmp(&b.candidate.fitness)
                    .unwrap()
            })
            .unwrap()
            .candidate
            .clone();
        Self {
            builder,
            working,
            best,
        }
    }

    pub fn get(&self) -> &Candidate<S> {
        &self.best
    }

    fn is_improvement(&self, candidate: &Candidate<S>) -> bool {
        candidate.fitness > self.best.fitness
    }

    fn work_on(&mut self, idx: usize) {
        // worker bee finds a new solution in the neighborhood
        let candidate = self.builder.config.explore(self.working.as_slice(), idx);
        if self.is_improvement(&candidate) {
            self.best = candidate.clone();
        } else {
            self.working[idx].decay();
            if self.working[idx].is_expired() {
                // scouter bee searches for a new solution
                let solution = self
                    .builder
                    .config
                    .create_solution(Some(&self.best.solution));
                let fitness = self.builder.config.evaluate_fitness(&solution);
                let candidate = Candidate { solution, fitness };
                if self.is_improvement(&candidate) {
                    self.best = candidate.clone();
                }
                self.working[idx] = WorkingWrapper::new(candidate, self.builder.num_retries);
            }
        }
    }

    fn select(&self) -> usize {
        let mut fitness_vector = self
            .working
            .iter()
            .map(|candidate| candidate.candidate.fitness)
            .collect::<Vec<f64>>();

        // normalize
        let sum: f64 = fitness_vector
            .iter()
            .fold(0f64, |acc, fitness| acc + *fitness);
        fitness_vector.iter_mut().for_each(|v| *v = *v / sum);

        // running totals
        let totals = fitness_vector
            .iter()
            .enumerate()
            .scan(0f64, |acc, (i, fitness)| {
                *acc += *fitness;
                Some((i, *acc))
            })
            .collect::<Vec<(usize, f64)>>();

        // roulette selection
        let p: f64 = rand::thread_rng().gen();
        match totals.iter().find(|(_, running_total)| *running_total >= p) {
            Some((idx, _)) => *idx,
            None => totals.len() - 1,
        }
    }

    pub fn step(&mut self) {
        for _ in 0..(self.builder.num_onlookers) {
            self.work_on(self.select());
        }
    }

    pub fn run_for_rounds(&mut self, rounds: usize) {
        for _ in 0..rounds {
            self.step()
        }
    }

    pub fn run_for(&mut self, duration: Duration) {
        let now = SystemTime::now();
        while now.elapsed().unwrap() < duration {
            for _ in 0..(self.builder.num_onlookers) {
                if now.elapsed().unwrap() < duration {
                    break;
                }
                self.work_on(self.select());
            }
        }
    }
}
