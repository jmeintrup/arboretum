use crate::datastructures::BitSet;
use crate::graph::BaseGraph;
use fnv::{FnvHashMap, FnvHashSet};
use std::cmp::max;
use std::fmt;
use std::fmt::{Display, Formatter};

pub enum TreeDecompositionValidationError {
    HasCycle,
    NotConnected,
    MissingVertex(usize),
    MissingEdge((usize, usize)),
    NotInducingSubtree(usize),
}

impl Display for TreeDecompositionValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            TreeDecompositionValidationError::HasCycle => write!(f, "Has Cycle"),
            TreeDecompositionValidationError::NotConnected => write!(f, "Not Connected"),
            TreeDecompositionValidationError::MissingVertex(v) => {
                write!(f, "Missing Vertex: {}", v)
            }
            TreeDecompositionValidationError::MissingEdge((u, v)) => {
                write!(f, "Missing Edge: ({}, {})", u, v)
            }
            TreeDecompositionValidationError::NotInducingSubtree(v) => {
                write!(f, "Not Inducing Subtree: {}", v)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeDecomposition {
    pub bags: Vec<Bag>,
    pub root: Option<usize>,
    pub max_bag_size: usize,
}

impl Default for TreeDecomposition {
    fn default() -> Self {
        Self {
            bags: Default::default(),
            root: None,
            max_bag_size: 0,
        }
    }
}

impl TreeDecomposition {
    pub fn flatten(&mut self) {
        while let Some((parent, child)) = self.find_combinable() {
            self.reroute(child, parent);
            self.remove_bag(child);
        }
    }

    pub fn with_root(vertex_set: FnvHashSet<usize>) -> Self {
        let mut td = Self::default();
        td.add_bag(vertex_set);
        td
    }

    fn reroute(&mut self, old_bag: usize, parent_idx: usize) {
        let old_neighbors = self.bags[old_bag].neighbors.clone();

        self.bags[parent_idx].neighbors.extend(old_neighbors.iter());
        self.bags[parent_idx].neighbors.remove(&parent_idx);
        self.bags[parent_idx].neighbors.remove(&old_bag);

        let old_id = self.bags[old_bag].id;
        for neighbor_idx in old_neighbors {
            if neighbor_idx == parent_idx {
                continue;
            }
            assert!(!self.bags[neighbor_idx].neighbors.contains(&parent_idx));

            assert_eq!(self.bags[neighbor_idx].neighbors.remove(&old_id), true);
            assert_eq!(self.bags[neighbor_idx].neighbors.insert(parent_idx), true);
        }

        self.bags[old_bag].neighbors.clear();
    }

    fn remove_bag(&mut self, id: usize) {
        assert!(self.bags[id].neighbors.is_empty());
        if id == self.bags.len() - 1 {
            //println!(" Only need to pop!");
            self.bags.pop();
        } else {
            //println!(" Pre swap {} {:?}", id, self.bags[id]);
            let old_last = self.bags.swap_remove(id);
            //println!(" Post swap {} {:?}", id, self.bags[id]);
            assert!(old_last.neighbors.is_empty());
            self.bags[id].id = id;
            let old_last = self.bags.len();
            for neighbor in self.bags[id].neighbors.clone() {
                //println!(" Neighbor {} {:?}", neighbor, self.bags[neighbor]);
                assert_eq!(self.bags[neighbor].neighbors.remove(&old_last), true);
                assert_eq!(self.bags[neighbor].neighbors.insert(id), true);
            }
        }
    }

    fn find_combinable(&self) -> Option<(usize, usize)> {
        for b in &self.bags {
            if let Some(n) = b
                .neighbors
                .iter()
                .find(|n| self.bags[**n].vertex_set.is_subset(&b.vertex_set))
            {
                return Some((b.id, self.bags[*n].id));
            }
        }
        None
    }

    pub fn add_bag(&mut self, vertex_set: FnvHashSet<usize>) -> usize {
        let id = self.bags.len();
        if id == 0 {
            self.root = Some(id);
        }
        self.max_bag_size = max(self.max_bag_size, vertex_set.len());
        self.bags.push(Bag {
            id,
            vertex_set,
            neighbors: FnvHashSet::default(),
        });
        id
    }

    pub fn add_child_bags(&mut self, parent: usize, children: Vec<FnvHashSet<usize>>) -> Vec<usize> {
        assert!(self.bags.len() < parent);
        let mut ids = Vec::with_capacity(children.len());
        for c in children {
            let id = self.add_bag(c);
            ids.push(id);
            self.add_edge(parent, id);
        }
        ids
    }

    pub fn add_edge(&mut self, b1: usize, b2: usize) {
        assert!(b1 < self.bags.len());
        assert!(b2 < self.bags.len());
        assert_ne!(b1, b2);
        self.bags[b1].neighbors.insert(b2);
        self.bags[b2].neighbors.insert(b1);
    }

    pub fn dfs(&self) -> TreeDecompositionIterator {
        let mut visited = BitSet::new(self.bags.len());
        let stack = if self.root.is_some() {
            visited.set_bit(self.root.unwrap());
            vec![self.root.unwrap()]
        } else {
            vec![]
        };
        TreeDecompositionIterator {
            td: self,
            stack,
            visited,
        }
    }

    pub fn replace_bag_v2(&mut self, target_bag: usize, mut td: TreeDecomposition) {
        let mut separators: FnvHashMap<usize, FnvHashSet<usize>> = FnvHashMap::default();
        for neighbor in &self.bags[target_bag].neighbors {
            let key = self.bags[*neighbor].id;
            let value: FnvHashSet<_> = self.bags[target_bag]
                .vertex_set
                .intersection(&self.bags[*neighbor].vertex_set)
                .copied()
                .collect();
            separators.insert(key, value);
        }
        let offset = self.bags.len();

        for bag in &mut td.bags {
            bag.id += offset;
            bag.neighbors = bag.neighbors.iter().map(|i| *i + offset).collect();
        }

        self.bags.reserve(td.bags.len());
        for bag in td.bags {
            self.bags.push(bag)
        }

        for (_, separator) in separators {
            let new_neighbor = self.bags[offset..]
                .iter_mut()
                .find(|b| b.vertex_set.is_superset(&separator))
                .unwrap();

            assert_eq!(new_neighbor.neighbors.remove(&target_bag), true);
            assert_eq!(new_neighbor.neighbors.insert(new_neighbor.id), true);
        }
        self.bags.swap_remove(target_bag);
        self.max_bag_size = self.bags.iter().map(|b| b.vertex_set.len()).max().unwrap();
    }

    pub fn replace_bag(&mut self, target_bag: usize, mut td: TreeDecomposition) {
        let neighbors_of_target_bag = self.bags[target_bag].neighbors.clone();
        let vertices_of_target_bag = self.bags[target_bag].vertex_set.clone();
        let offset = self.bags.len();
        td.bags.iter_mut().for_each(|b| {
            b.id += offset;
            b.neighbors = b.neighbors.iter().map(|i| *i + offset).collect();
        });
        for neighbor_of_target_bag in neighbors_of_target_bag {
            let neighbor_of_target_bag = &mut self.bags[neighbor_of_target_bag];
            let intersection: FnvHashSet<_> = vertices_of_target_bag
                .intersection(&neighbor_of_target_bag.vertex_set)
                .copied()
                .collect();
            let new_neighbor_of_neighbor_of_target_bag = td
                .bags
                .iter_mut()
                .find(|b| b.vertex_set.is_superset(&intersection))
                .unwrap();
            new_neighbor_of_neighbor_of_target_bag
                .neighbors
                .insert(neighbor_of_target_bag.id);

            assert_eq!(neighbor_of_target_bag.neighbors.remove(&target_bag), true);
            assert_eq!(
                neighbor_of_target_bag
                    .neighbors
                    .insert(new_neighbor_of_neighbor_of_target_bag.id),
                true
            );
        }
        self.bags.extend_from_slice(&td.bags);
        let old_idx = self.bags.len() - 1;
        self.bags.swap(target_bag, old_idx);
        self.bags[target_bag].id = target_bag;
        for id in self.bags[target_bag].neighbors.clone() {
            assert_eq!(self.bags[id].neighbors.remove(&old_idx), true);
            assert_eq!(self.bags[id].neighbors.insert(target_bag), true);
        }
        self.bags.swap_remove(old_idx);
        self.max_bag_size = self.bags.iter().map(|b| b.vertex_set.len()).max().unwrap();
    }

    pub fn bags(&self) -> &[Bag] {
        &self.bags
    }

    pub fn combine_with_or_replace(&mut self, glue_point: usize, other: TreeDecomposition) {
        if self.bags.is_empty() || (self.bags.len() == 1 && self.bags[0].vertex_set.is_empty()) {
            *self = other;
        } else {
            self.combine_with(glue_point, other);
        }
    }

    pub fn combine_with(&mut self, glue_point: usize, mut other: TreeDecomposition) {
        assert!(glue_point < self.bags.len());
        self.max_bag_size = max(self.max_bag_size, other.max_bag_size);
        let offset = self.bags.len();
        for mut b in other.bags.iter_mut() {
            b.id += offset;
            b.neighbors = b.neighbors.iter().map(|n| *n + offset).collect();
        }
        let other_glue_point = other
            .bags
            .iter_mut()
            .find(|b| b.vertex_set.is_superset(&self.bags[glue_point].vertex_set))
            .unwrap();
        other_glue_point.neighbors.insert(glue_point);
        self.bags[glue_point].neighbors.insert(other_glue_point.id);
        self.bags.extend(other.bags.drain(..));
    }

    pub fn verify<G: BaseGraph>(&self, graph: &G) -> Result<(), TreeDecompositionValidationError> {
        if !self.is_connected() {
            return Err(TreeDecompositionValidationError::NotConnected);
        }

        if self.is_cyclic() {
            return Err(TreeDecompositionValidationError::HasCycle);
        }

        if let Some(v) = self.get_missing_vertex(graph) {
            return Err(TreeDecompositionValidationError::MissingVertex(v));
        }

        if let Some(e) = self.get_missing_edge(graph) {
            return Err(TreeDecompositionValidationError::MissingEdge(e));
        }

        if let Some(v) = self.get_vertex_not_inducing_subtree(graph) {
            return Err(TreeDecompositionValidationError::NotInducingSubtree(v));
        }

        Ok(())
    }

    fn is_connected(&self) -> bool {
        if self.bags.is_empty() {
            return true;
        }
        let mut visited = BitSet::new(self.bags.len());
        self.dfs().for_each(|b| {
            visited.set_bit(b.id);
        });
        visited.full()
    }

    fn is_cyclic(&self) -> bool {
        if self.bags.is_empty() {
            return true;
        }
        let mut visited = BitSet::new(self.bags.len());
        self.is_cyclic_rec(&mut visited, self.root.unwrap(), None)
    }

    fn is_cyclic_rec(&self, visited: &mut BitSet, v: usize, parent: Option<usize>) -> bool {
        visited.set_bit(v);
        for n in self.bags[v].neighbors.iter().copied() {
            if !visited[n] {
                if self.is_cyclic_rec(visited, n, Some(v)) {
                    return true;
                }
            } else if parent.is_some() && n != parent.unwrap() {
                return true;
            }
        }
        false
    }

    fn get_missing_vertex<G: BaseGraph>(&self, graph: &G) -> Option<usize> {
        let mut vertices: FnvHashSet<usize> = graph.vertices().collect();
        self.bags.iter().for_each(|b| {
            b.vertex_set.iter().for_each(|x| {
                vertices.remove(x);
            })
        });
        if vertices.is_empty() {
            None
        } else {
            Some(*vertices.iter().next().unwrap())
        }
    }

    fn get_missing_edge<G: BaseGraph>(&self, graph: &G) -> Option<(usize, usize)> {
        for u in graph.vertices() {
            for v in graph.vertices() {
                if u < v
                    && graph.has_edge(u, v)
                    && self
                        .bags
                        .iter()
                        .find(|b| b.vertex_set.contains(&u) && b.vertex_set.contains(&v))
                        .is_none()
                {
                    return Some((u, v));
                }
            }
        }
        None
    }

    fn get_vertex_not_inducing_subtree<G: BaseGraph>(&self, graph: &G) -> Option<usize> {
        for u in graph.vertices() {
            let mut inducing_bags: FnvHashSet<usize> = self
                .bags
                .iter()
                .filter(|b| b.vertex_set.contains(&u))
                .map(|b| b.id)
                .collect();

            let first = *inducing_bags.iter().next().unwrap();
            inducing_bags.remove(&first);
            let mut visited = BitSet::new(self.bags.len());
            visited.set_bit(first);
            let mut stack: Vec<usize> = vec![first];
            while let Some(c) = stack.pop() {
                for n in self.bags[c].neighbors.iter().copied() {
                    let bag = &self.bags[n];
                    if !visited[n] && bag.vertex_set.contains(&u) {
                        inducing_bags.remove(&bag.id);
                        stack.push(n);
                        visited.set_bit(n);
                    }
                }
            }
            if !inducing_bags.is_empty() {
                return Some(u + 1);
            }
        }
        None
    }
}

pub struct TreeDecompositionIterator<'a> {
    td: &'a TreeDecomposition,
    stack: Vec<usize>,
    visited: BitSet,
}

impl<'a> Iterator for TreeDecompositionIterator<'a> {
    type Item = &'a Bag;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.stack.pop()?;
        for c in self.td.bags[current].neighbors.iter().copied() {
            if !self.visited[c] {
                self.stack.push(c);
                self.visited.set_bit(c);
            }
        }
        Some(self.td.bags.get(current).unwrap())
    }
}

#[derive(Debug, Default, Clone)]
pub struct Bag {
    pub id: usize,
    pub vertex_set: FnvHashSet<usize>,
    pub neighbors: FnvHashSet<usize>,
}
