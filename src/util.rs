use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use crate::graph::tree_decomposition::{TreeDecomposition, TreeDecompositionError};
use std::borrow::Borrow;
use std::cmp::max;
use std::time::{Duration, SystemTime};

pub trait Stopper {
    fn stop(&mut self) -> bool;
    fn init(&mut self);
}

pub struct Timer {
    timer: SystemTime,
    duration: Duration,
}

impl Timer {
    pub fn new(duration: Duration) -> Self {
        Self {
            timer: SystemTime::now(),
            duration,
        }
    }
}

impl Stopper for Timer {
    fn stop(&mut self) -> bool {
        self.timer.elapsed().unwrap() > self.duration
    }

    fn init(&mut self) {
        self.timer = SystemTime::now();
    }
}

#[derive(Clone, Debug)]
pub struct EliminationOrder {
    data: Vec<usize>,
    width: u32,
}

impl EliminationOrder {
    pub(crate) fn new(data: Vec<usize>, width: u32) -> Self {
        EliminationOrder { data, width }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn order(&self) -> &[usize] {
        self.data.as_slice()
    }

    pub fn to_tree_decomposition<G: Graph>(
        &self,
        graph: &G,
    ) -> Result<TreeDecomposition, TreeDecompositionError> {
        let mut tree_decomposition = TreeDecomposition::with_capacity(self.data.len());
        let mut graph = HashMapGraph::from_graph(graph.borrow());

        let mut stack: Vec<(usize, Vec<usize>)> = Vec::with_capacity(self.data.len());
        self.data.iter().for_each(|v| {
            let bag: Vec<usize> = graph.neighborhood(*v).collect();
            stack.push((*v, bag));
            graph.eliminate_vertex(*v);
        });
        let first_bag = stack.last().unwrap().0;
        let (v, nb) = stack.pop().unwrap();
        let mut bag: Vec<usize> = nb.iter().cloned().collect();
        bag.push(v);
        tree_decomposition.add_bag(v, bag.as_slice());

        stack.iter().rev().for_each(|(v, nb)| {
            let found = tree_decomposition.bags().iter().find(|(_, b)| {
                for x in nb {
                    if !b.contains(x) {
                        return false;
                    }
                }
                return true;
            });

            let found = if found.is_some() {
                *found.unwrap().0
            } else {
                first_bag
            };

            let mut bag: Vec<usize> = nb.iter().cloned().collect();
            bag.push(*v);
            tree_decomposition.add_bag(*v, bag.as_slice());
            tree_decomposition.add_edge(*v, found);
        });

        match tree_decomposition.validate(&graph) {
            Err(e) => Err(e),
            Ok(_) => Ok(tree_decomposition),
        }
    }
}

pub fn get_width_virtual_elimination<G: Graph>(graph: &G, order: &[usize]) -> usize {
    let mut stack = Vec::with_capacity(graph.order());
    let mut width = 0;
    for (idx, v) in order.iter().enumerate() {
        let mut degree = 0;
        stack.push(v);
        let mut visited = vec![false; graph.order()];
        while let Some(x) = stack.pop() {
            for u in graph.neighborhood(*x) {
                if !visited[u] {
                    visited[u] = true;
                    match order[0..idx].iter().find(|x| **x == u) {
                        Some(u) => {
                            stack.push(u);
                        }
                        None => {
                            if u != *v {
                                degree += 1;
                            }
                        }
                    }
                }
            }
        }
        width = max(degree, width);
    }
    width
}

pub fn get_width<G: MutableGraph>(graph: &G, order: &[usize]) -> usize {
    let mut graph = graph.clone();
    order
        .iter()
        .map(|v| {
            let degree = graph.degree(*v);
            graph.eliminate_vertex(*v);
            degree
        })
        .max()
        .unwrap()
}

pub fn local_search<G: MutableGraph>(graph: &G, order: &EliminationOrder) -> EliminationOrder {
    let mut new_order = Vec::from(order.order());
    let mut cloned = graph.clone();

    // find largest vertex in elimination order
    let mut largest = None;
    for (idx, v) in order.order().iter().enumerate() {
        if cloned.degree(*v) == order.width() as usize {
            largest = Some((idx, v));
            break;
        }
        cloned.eliminate_vertex(*v);
    }
    let largest = largest.unwrap();
    //for (idx, v) in
    let best = order
        .order()
        .iter()
        .enumerate()
        .filter(|(_, v)| *v != largest.1)
        .map(|(idx, v)| {
            new_order.swap(idx, largest.0);
            let width = get_width(graph, new_order.as_slice());
            new_order.swap(idx, largest.0);
            (idx, v, width)
        })
        .filter(|(_, _, width)| *width <= order.width() as usize)
        .min_by(|u, v| u.2.cmp(&v.2));

    let (idx, _, width) = best.unwrap();
    new_order.swap(idx, largest.0);
    EliminationOrder::new(new_order, width as u32)
}

pub fn is_smaller_width<G: MutableGraph>(graph: &G, order: &[usize], old_width: usize) -> bool {
    let mut graph = graph.clone();
    for v in order {
        if graph.degree(*v) > old_width {
            return false;
        } else {
            graph.eliminate_vertex(*v);
        }
    }
    true
}
