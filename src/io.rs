use crate::datastructures::BitSet;
use crate::graph::bit_graph::BitGraph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use std::convert::TryFrom;
use std::io::{BufRead, BufReader, Read};
use std::{fs, io};

fn nums_error(res: &[Result<usize, std::num::ParseIntError>]) -> bool {
    res.len() != 2 || res[0].is_err() || res[1].is_err()
}

pub fn dimacs_p(line: &str) -> Result<(usize, usize), std::io::Error> {
    let nums: Vec<Result<usize, std::num::ParseIntError>> = line
        .trim_start_matches('p')
        .trim()
        .split(' ')
        .skip(1)
        .map(|s| s.parse())
        .collect();
    if nums_error(&nums) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid line",
        ));
    }
    let u = nums[0].as_ref().unwrap();
    let v = nums[1].as_ref().unwrap();
    Ok((*u, *v))
}

pub fn dimacs_e(line: &str) -> Result<(usize, usize), std::io::Error> {
    let nums: Vec<Result<usize, std::num::ParseIntError>> = line
        .trim_start_matches('e')
        .trim()
        .split(' ')
        .map(|s| s.parse())
        .collect();
    if nums_error(&nums) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid line",
        ));
    }
    let u = nums[0].as_ref().unwrap() - 1;
    let v = nums[1].as_ref().unwrap() - 1;
    Ok((u, v))
}

pub struct DimacsRead<T: BufRead>(pub T);

impl<T: BufRead> TryFrom<DimacsRead<T>> for HashMapGraph {
    type Error = std::io::Error;

    fn try_from(reader: DimacsRead<T>) -> Result<Self, Self::Error> {
        let mut reader = reader.0;
        let mut graph = HashMapGraph::new();
        for line in reader.lines().map(|line| line.unwrap()) {
            match line.chars().next() {
                Some('p') => {}
                Some('c') => {}
                _ => {
                    let (u, v) = dimacs_e(&line)?;
                    if u != v {
                        graph.add_edge(u, v);
                    }
                }
            };
        }
        return Ok(graph);
    }
}

impl<T: BufRead> TryFrom<DimacsRead<T>> for BitGraph {
    type Error = std::io::Error;

    fn try_from(reader: DimacsRead<T>) -> Result<Self, Self::Error> {
        let mut reader = reader.0;

        let mut graph = None;
        for line in reader.lines() {
            let line = line?;
            match line.chars().next() {
                Some('c') => {}
                Some('p') => {
                    let (n, _) = dimacs_p(line.as_str())?;
                    graph = Some(vec![BitSet::new(n); n]);
                }
                _ => {
                    let (u, v) = dimacs_e(line.as_str())?;
                    if u != v {
                        if graph.is_none() {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Invalid line",
                            ));
                        }
                        let mut graph = graph.as_mut().unwrap();
                        graph[u].set_bit(v);
                        graph[v].set_bit(u);
                    }
                }
            };
        }

        if graph.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid line",
            ));
        }
        Ok(graph.unwrap().into())
    }
}
