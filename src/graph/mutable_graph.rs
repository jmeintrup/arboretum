use crate::graph::graph::Graph;

pub trait MutableGraph: Graph {
    fn add_vertex(&mut self, u: usize);
    fn add_vertex_with_capacity(&mut self, u: usize, capacity: usize);
    fn remove_vertex(&mut self, u: usize);
    fn add_edge(&mut self, u: usize, v: usize);
    fn remove_edge(&mut self, u: usize, v: usize);
    fn eliminate_vertex(&mut self, u: usize);
    fn contract(&mut self, u: usize, v: usize);
    fn new() -> Self;
    fn with_capacity(capacity: usize) -> Self;
    fn make_clique(&mut self, vertices: &[usize]) {
        for (i, v) in vertices.iter().enumerate() {
            for u in vertices.iter().skip(i + 1) {
                self.add_edge(*u, *v);
            }
        }
    }
}
