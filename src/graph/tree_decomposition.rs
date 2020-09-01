use crate::graph::graph::Graph;
use core::fmt;
use fnv::FnvHashSet;
use num::Num;
use std::borrow::Borrow;
use std::cmp::{max, Ordering};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Bag {
    vertices: FnvHashSet<u32>,
    id: u32,
}

impl PartialEq for Bag {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Bag {}

impl PartialOrd for Bag {
    fn partial_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for Bag {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl Bag {
    pub fn new(vertices: FnvHashSet<u32>, id: u32) -> Self {
        Self { vertices, id }
    }

    pub fn contains(&self, v: &u32) -> bool {
        self.vertices.get(v).is_some()
    }

    pub fn vertices(&self) -> &FnvHashSet<u32> {
        self.vertices.borrow()
    }
}

#[derive(Debug)]
pub struct TreeDecomposition {
    width: usize,
    bags: HashMap<usize, Vec<usize>>,
    neighbors: HashMap<usize, Vec<usize>>,
}

impl TreeDecomposition {
    pub fn new() -> Self {
        TreeDecomposition {
            width: 1,
            bags: HashMap::new(),
            neighbors: HashMap::new(),
        }
    }

    pub fn neighbors(&self, b: usize) -> &[usize] {
        self.neighbors.get(&b).unwrap().as_slice()
    }

    pub fn order(&self) -> usize {
        self.bags.len()
    }

    pub fn bag(&self, b: usize) -> &[usize] {
        self.bags.get(&b).unwrap().as_slice()
    }

    pub fn bags(&self) -> &HashMap<usize, Vec<usize>> {
        &self.bags
    }

    pub fn with_capacity(capacity: usize) -> Self {
        TreeDecomposition {
            width: 1,
            bags: HashMap::with_capacity(capacity),
            neighbors: HashMap::with_capacity(capacity),
        }
    }

    pub fn add_bag(&mut self, v: usize, bag: &[usize]) -> usize {
        self.bags.insert(v, Vec::from(bag));
        self.neighbors.insert(v, Vec::new());
        self.width = max(self.width, bag.len() - 1);
        self.bags.len()
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.add_endpoint(u, v);
        self.add_endpoint(v, u);
    }

    fn add_endpoint(&mut self, u: usize, v: usize) {
        self.neighbors.get_mut(&u).unwrap().push(v);
    }

    fn is_connected_and_acyclic(&self) -> Result<(), TreeDecompositionError> {
        let mut marked: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(Option<usize>, usize)> = Vec::with_capacity(self.bags.len());
        stack.push((None, *self.bags.keys().next().unwrap()));

        while let Some((from, v)) = stack.pop() {
            marked.insert(v);
            for u in self.neighbors.get(&v).unwrap().iter() {
                if from == None || from.unwrap() != *u {
                    if marked.contains(u) {
                        let msg = format!("Contains cycle between {} and {}.", u, v);
                        return Err(TreeDecompositionError::new(
                            TreeDecompositionErrorKind::ContainsCycleError,
                            &msg,
                        ));
                    }
                    stack.push((Some(v), *u));
                }
            }
        }
        match self.bags.keys().cloned().find(|e| !marked.contains(e)) {
            None => Ok(()),
            Some(v) => {
                let msg = format!("Vertex {} is not connected.", v);
                Err(TreeDecompositionError::new(
                    TreeDecompositionErrorKind::NotConnectedError,
                    &msg,
                ))
            }
        }
    }

    fn valid_connectivity<G: Graph>(&self, graph: &G) -> Result<(), TreeDecompositionError> {
        for v in graph.vertices() {
            let mut idx: Option<usize> = None;
            for (i, bag) in self.bags() {
                if bag.iter().find(|u| **u == v).is_some() {
                    idx = Some(*i);
                    break;
                }
            }
            if idx.is_none() {
                let msg = format!("Vertex {} is not contained in any bag.", v);
                return Err(TreeDecompositionError::new(
                    TreeDecompositionErrorKind::MissingVerticesError,
                    &msg,
                ));
            }

            let mut marked: HashSet<usize> = HashSet::with_capacity(self.bags.len());
            let mut stack: Vec<usize> = Vec::with_capacity(self.bags.len());

            stack.push(idx.unwrap());
            while let Some(x) = stack.pop() {
                marked.insert(x);
                for u in self.neighbors.get(&x).unwrap().iter() {
                    if !marked.contains(u) && self.bags.get(u).unwrap().contains(&v) {
                        stack.push(*u);
                    }
                }
            }

            for (i, bag) in self.bags.iter() {
                if !marked.contains(i) && bag.iter().find(|u| **u == v).is_some() {
                    let msg = format!("Vertex {} does not induce a sub-tree.", v);
                    return Err(TreeDecompositionError::new(
                        TreeDecompositionErrorKind::TreeInductionError,
                        &msg,
                    ));
                }
            }
        }
        Ok(())
    }

    fn contains_all_vertices<G: Graph>(&self, graph: &G) -> Result<(), TreeDecompositionError> {
        for v in graph.vertices() {
            let mut found = false;
            for bag in self.bags.values() {
                if bag.iter().find(|u| **u == v).is_some() {
                    found = true;
                    break;
                }
            }
            if !found {
                let msg = format!("Vertex {} is not contained in any bag.", v);
                return Err(TreeDecompositionError::new(
                    TreeDecompositionErrorKind::MissingVerticesError,
                    &msg,
                ));
            }
        }
        Ok(())
    }

    fn contains_all_edges<G: Graph>(&self, graph: &G) -> Result<(), TreeDecompositionError> {
        for i in graph.vertices() {
            for j in graph.vertices() {
                if i < j && graph.has_edge(i, j) {
                    let mut found = false;
                    for bag in self.bags.values() {
                        if bag.iter().find(|u| **u == i).is_some()
                            && bag.iter().find(|u| **u == j).is_some()
                        {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let msg = format!("Edge ({}, {}) is not contained in any bag.", i, j);
                        return Err(TreeDecompositionError::new(
                            TreeDecompositionErrorKind::MissingEdgesError,
                            &msg,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_width(&self) -> Result<(), TreeDecompositionError> {
        let max = self
            .bags
            .iter()
            .map(|(v, b)| (v, b.len()))
            .max_by(|(_, v_len), (_, u_len)| v_len.cmp(u_len))
            .unwrap();
        if max.1 - 1 != self.width {
            let msg = format!(
                "Invalid width, target width is {} but bag {} results in width {}.",
                self.width,
                max.0,
                max.1 - 1
            );
            return Err(TreeDecompositionError::new(
                TreeDecompositionErrorKind::InvalidWidthError,
                &msg,
            ));
        }
        Ok(())
    }

    pub fn validate<G: Graph>(&self, graph: &G) -> Result<(), TreeDecompositionError> {
        let res = self.is_connected_and_acyclic();
        if res.is_err() {
            return res;
        }
        let res = self.check_width();
        if res.is_err() {
            return res;
        }
        let res = self.contains_all_vertices(graph);
        if res.is_err() {
            return res;
        }
        let res = self.valid_connectivity(graph);
        if res.is_err() {
            return res;
        }
        let res = self.contains_all_edges(graph);
        if res.is_err() {
            return res;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TreeDecompositionError {
    kind: TreeDecompositionErrorKind,
    msg: String,
}

#[derive(Debug, Clone)]
pub enum TreeDecompositionErrorKind {
    ContainsCycleError,
    MissingVerticesError,
    MissingEdgesError,
    NotConnectedError,
    TreeInductionError,
    InvalidWidthError,
}

impl fmt::Display for TreeDecompositionErrorKind {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match *self {
            Self::ContainsCycleError => f.write_str("ContainsCycleError"),
            Self::MissingVerticesError => f.write_str("MissingVerticesError"),
            Self::NotConnectedError => f.write_str("NotConnectedError"),
            Self::TreeInductionError => f.write_str("TreeInductionError"),
            Self::MissingEdgesError => f.write_str("MissingEdgesError"),
            Self::InvalidWidthError => f.write_str("InvalidWidthError"),
        }
    }
}

impl TreeDecompositionError {
    fn new(kind: TreeDecompositionErrorKind, msg: &str) -> Self {
        Self {
            kind,
            msg: String::from(msg),
        }
    }
}

impl fmt::Display for TreeDecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl Error for TreeDecompositionError {
    fn description(&self) -> &str {
        &self.msg
    }
}
