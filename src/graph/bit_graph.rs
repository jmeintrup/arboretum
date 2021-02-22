use crate::datastructures::BitSet;
use crate::graph::base_graph::BaseGraph;
use fxhash::FxHashMap;
use std::borrow::Borrow;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct BitGraph {
    graph: Vec<BitSet>,
}

impl From<Vec<BitSet>> for BitGraph {
    fn from(graph: Vec<BitSet>) -> Self {
        Self { graph }
    }
}

impl From<&[BitSet]> for BitGraph {
    fn from(slice: &[BitSet]) -> Self {
        Self {
            graph: Vec::from(slice),
        }
    }
}

impl BitGraph {
    pub fn from_graph<G: BaseGraph>(og_graph: &G, og_to_self: &FxHashMap<u32, u32>) -> Self {
        let mut graph = vec![BitSet::new(og_graph.order()); og_graph.order()];

        for v in og_graph.vertices() {
            for u in og_graph.neighborhood(v) {
                let a = (*og_to_self.get(&(v as u32)).unwrap()) as usize;
                let b = (*og_to_self.get(&(u as u32)).unwrap()) as usize;
                graph[a].set_bit(b);
            }
        }
        Self { graph }
    }

    pub fn neighborhood_as_bitset(&self, u: usize) -> &BitSet {
        self.graph[u].borrow()
    }

    pub fn exterior_border(&self, c: &BitSet) -> BitSet {
        let mut border = BitSet::new(c.len());

        for v in c.iter() {
            border.or(&self.graph[v])
        }
        border.and_not(&c);
        border
    }
}

impl BaseGraph for BitGraph {
    fn degree(&self, u: usize) -> usize {
        self.graph[u].cardinality()
    }

    fn order(&self) -> usize {
        self.graph.len()
    }

    fn is_clique(&self, vertices: &[usize]) -> bool {
        for u in vertices {
            for v in vertices {
                if u < v && !self.graph[*u][*v] {
                    return false;
                }
            }
        }
        true
    }

    fn is_neighborhood_clique(&self, u: usize) -> bool {
        for u in self.graph[u].iter() {
            for v in self.graph[u].iter() {
                if u < v && !self.graph[u][v] {
                    return false;
                }
            }
        }
        true
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.graph[u][v]
    }

    fn is_simplicial(&self, u: usize) -> bool {
        self.is_neighborhood_clique(u)
    }

    fn is_almost_simplicial(&self, u: usize) -> bool {
        for ignore in self.graph[u].iter() {
            for u in self.graph[u].iter() {
                for v in self.graph[u].iter() {
                    if u < v && u != ignore && !self.graph[u][v] {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn vertices(&self) -> Box<dyn Iterator<Item = usize>> {
        Box::new(0..self.graph.len())
    }

    fn neighborhood(&self, u: usize) -> Box<dyn Iterator<Item = usize> + '_> {
        Box::new(self.graph[u].iter())
    }

    fn fill_in_count(&self, u: usize) -> usize {
        let mut cnt = 0;
        for u in self.graph[u].iter() {
            for v in self.graph[u].iter() {
                if u < v && !self.graph[u][v] {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    fn min_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize> {
        (0..self.graph.len()).min_by(cmp)
    }

    fn max_vertex_by<F: FnMut(&usize, &usize) -> Ordering>(&self, cmp: F) -> Option<usize> {
        (0..self.graph.len()).max_by(cmp)
    }

    fn min_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize> {
        self.graph[u].iter().min_by(cmp)
    }

    fn max_neighbor_by<F: FnMut(&usize, &usize) -> Ordering>(
        &self,
        u: usize,
        cmp: F,
    ) -> Option<usize> {
        self.graph[u].iter().max_by(cmp)
    }
}
