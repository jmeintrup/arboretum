use crate::abc::colony::Colony;
use crate::abc::config::Config;
use std::marker::PhantomData;

pub struct Builder<S: Clone, C: Config<S>> {
    n: PhantomData<S>,
    pub(in crate::abc) config: C,
    pub(in crate::abc) num_workers: usize,
    pub(in crate::abc) num_onlookers: usize,
    pub(in crate::abc) num_retries: usize,
}

impl<S: Clone, C: Config<S>> Builder<S, C> {
    pub fn new(config: C, num_workers: usize) -> Self {
        Self {
            n: PhantomData,
            config,
            num_workers,
            num_onlookers: num_workers,
            num_retries: num_workers,
        }
    }

    pub fn set_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    pub fn set_num_onlookers(mut self, num_onlookers: usize) -> Self {
        self.num_onlookers = num_onlookers;
        self
    }

    pub fn set_num_tries(mut self, num_retries: usize) -> Self {
        self.num_retries = num_retries;
        self
    }

    pub fn build(self) -> Colony<S, C> {
        Colony::new(self)
    }
}
