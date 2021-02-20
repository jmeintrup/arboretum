use crate::graph::BaseGraph;
use crate::graph::HashMapGraph;
use crate::graph::MutableGraph;
use crate::tree_decomposition::TreeDecomposition;
use std::convert::TryFrom;
use std::io::{BufRead, ErrorKind, Write};

pub struct PaceReader<T: BufRead>(pub T);

pub struct PaceWriter<'a, 'b: 'a, T: Write> {
    tree_decomposition: &'a TreeDecomposition,
    graph: &'b HashMapGraph,
    writer: T,
}

impl<'a, 'b: 'a, T: Write> PaceWriter<'a, 'b, T> {
    pub fn output(mut self) -> Result<(), std::io::Error> {
        let bag_count: usize = self.tree_decomposition.bags.len();
        let max_bag: usize = self.tree_decomposition.max_bag_size;
        writeln!(
            self.writer,
            "s td {} {} {}",
            bag_count,
            max_bag,
            self.graph.order()
        )?;
        for b in self.tree_decomposition.bags() {
            let mut tmp: Vec<_> = b.vertex_set.iter().copied().collect();
            tmp.sort_unstable();
            let vertices: Vec<_> = tmp.iter().map(|i| (i + 1).to_string()).collect();
            let vertices = vertices.iter().fold(String::new(), |mut acc, v| {
                acc.push_str(v.as_str());
                acc.push(' ');
                acc
            });
            writeln!(self.writer, "b {} {}", b.id + 1, vertices)?;
        }
        for b in self.tree_decomposition.bags() {
            for child in b.neighbors.iter().copied().filter(|i| *i > b.id) {
                writeln!(self.writer, "{} {}", b.id + 1, child + 1)?;
            }
        }
        Ok(())
    }

    pub fn new(
        tree_decomposition: &'a TreeDecomposition,
        graph: &'b HashMapGraph,
        writer: T,
    ) -> Self {
        Self {
            tree_decomposition,
            graph,
            writer,
        }
    }
}

impl<T: BufRead> TryFrom<PaceReader<T>> for HashMapGraph {
    type Error = std::io::Error;

    fn try_from(reader: PaceReader<T>) -> Result<Self, Self::Error> {
        let reader = reader.0;
        let mut graph: Option<HashMapGraph> = None;
        let mut order: Option<usize> = None;
        for line in reader.lines() {
            let line = line?;
            let elements: Vec<_> = line.split(' ').collect();
            match elements[0] {
                "c" => {
                    // who cares about comments..
                }
                "p" => {
                    order = Some(parse_order(&elements)?);
                    graph = Some(HashMapGraph::with_capacity(order.unwrap()));
                    (0..order.unwrap()).for_each(|v| graph.as_mut().unwrap().add_vertex(v));
                }
                _ => match graph.as_mut() {
                    Some(graph) => {
                        let u = parse_vertex(elements[0], order.unwrap())?;
                        let v = parse_vertex(elements[1], order.unwrap())?;
                        graph.add_edge(u, v);
                    }
                    None => {
                        return Err(std::io::Error::new(
                            ErrorKind::Other,
                            "Edges encountered before graph creation",
                        ));
                    }
                },
            };
        }
        match graph {
            Some(graph) => Ok(graph),
            None => Err(std::io::Error::new(
                ErrorKind::Other,
                "No graph created during parsing",
            )),
        }
    }
}

fn parse_vertex(v: &str, order: usize) -> Result<usize, std::io::Error> {
    match v.parse::<usize>() {
        Ok(u) => {
            if u == 0 || u > order {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid vertex label",
                ))
            } else {
                Ok(u - 1)
            }
        }
        Err(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid vertex label",
        )),
    }
}

fn parse_order(elements: &[&str]) -> Result<usize, std::io::Error> {
    if elements.len() < 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid line received starting with p",
        ));
    }
    match elements[2].parse::<usize>() {
        Ok(order) => Ok(order),
        Err(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid order of graph",
        )),
    }
}
