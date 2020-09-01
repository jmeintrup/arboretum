#[derive(Clone)]
pub struct Candidate<S: Clone> {
    pub solution: S,
    pub fitness: f64,
}

#[derive(Clone)]
pub struct WorkingWrapper<S: Clone> {
    pub(in crate::abc) candidate: Candidate<S>,
    pub(in crate::abc) retries: usize,
}

impl<S: Clone> WorkingWrapper<S> {
    pub fn new(candidate: Candidate<S>, retries: usize) -> Self {
        Self { candidate, retries }
    }

    pub fn is_expired(&self) -> bool {
        self.retries <= 0
    }

    pub fn decay(&mut self) {
        self.retries -= 1;
    }
}
