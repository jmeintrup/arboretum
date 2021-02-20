use std::cmp::Ordering;
use std::fmt::Debug;

pub trait BaseGraph: Clone + Debug {
    fn degree(&self, u: usize) -> usize;
    fn order(&self) -> usize;
    fn is_clique(&self, vertices: &[usize]) -> bool;
    fn is_neighborhood_clique(&self, u: usize) -> bool;
    fn has_edge(&self, u: usize, v: usize) -> bool;
    fn is_simplicial(&self, u: usize) -> bool;
    fn is_almost_simplicial(&self, u: usize) -> bool;
    fn vertices(&self) -> Box<dyn Iterator<Item = usize> + '_>;
    fn neighborhood(&self, u: usize) -> Box<dyn Iterator<Item = usize> + '_>;
    fn fill_in_count(&self, u: usize) -> usize;
    fn min_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize>;
    fn max_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize>;
    fn min_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize>;
    fn max_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize>;
}
