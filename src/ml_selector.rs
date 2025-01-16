use arboretum_td::graph::{BaseGraph, MutableGraph};
use arboretum_td::heuristic_elimination_order::{
    HeuristicEliminationDecomposer, MinDegreeSelector, MinFillSelector, Selector,
};
use arboretum_td::solver::AtomSolver;
use arboretum_td::{graph::HashMapGraph, io::PaceReader};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::BufWriter;
use std::time::Instant;
use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};
use std::{
    io::Write,
    process::{Command, Stdio},
};

pub struct MLSelector {
    graph: HashMapGraph,
    cache: Vec<i64>,
}

impl From<HashMapGraph> for MLSelector {
    fn from(graph: HashMapGraph) -> Self {
        let mut ml_selector = Self {
            cache: vec![0; graph.order()],
            graph,
        };

        ml_selector.update_cache();
        ml_selector
    }
}

impl MLSelector {
    fn update_cache(&mut self) {
        ml_values(&self.graph, &mut self.cache).expect("Error Updating Values");
    }
}

impl Selector for MLSelector {
    fn graph(&self) -> &HashMapGraph {
        &self.graph
    }

    fn value(&self, v: usize) -> i64 {
        self.cache[v]
    }

    fn eliminate_vertex(&mut self, v: usize) {
        self.graph.eliminate_vertex(v);
        self.update_cache();
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Tuple(i64, i64);

pub type MLDecomposer = HeuristicEliminationDecomposer<MLSelector>;

fn main() -> io::Result<()> {
    let file = File::create("output.csv")?;
    let buf_writer = BufWriter::new(file);

    let mut wtr = csv::Writer::from_writer(buf_writer);
    wtr.write_record([
        "file",
        "mindegree",
        "t_mindegree",
        "minfill",
        "t_minfill",
        "ml",
        "t_ml",
    ])?;

    let path = "./data";
    let entries = std::fs::read_dir(path).unwrap();
    for entry in entries {
        match entry {
            Ok(entry) => {
                let file = std::fs::File::open(entry.path());
                match file {
                    Ok(file) => {
                        let mut mindegree = 0;
                        let mut elapsed_mindegree = 0;
                        let mut minfill = 0;
                        let mut elapsed_minfill = 0;
                        let mut ml = 0;
                        let mut elapsed_ml = 0;

                        println!("Processing {:?}", file);
                        let reader = PaceReader(BufReader::new(file));
                        let graph = HashMapGraph::try_from(reader)?;

                        let now = Instant::now();
                        let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
                            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
                        let x = decomposer.compute().computed_tree_decomposition();
                        if let Some(decomposition_mindegree) = x {
                            let elapsed = now.elapsed();
                            println!(
                                "Min Degree: {:?}: {:?}",
                                decomposition_mindegree.max_bag_size, elapsed
                            );
                            mindegree = decomposition_mindegree.max_bag_size;
                            elapsed_mindegree = elapsed.as_millis();
                        }

                        let now = Instant::now();
                        let decomposer: HeuristicEliminationDecomposer<MinFillSelector> =
                            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
                        let x = decomposer.compute().computed_tree_decomposition();
                        if let Some(decomposition_minfill) = x {
                            let elapsed = now.elapsed();
                            println!(
                                "Min Fill: {:?}: {:?}",
                                decomposition_minfill.max_bag_size, elapsed
                            );
                            minfill = decomposition_minfill.max_bag_size;
                            elapsed_minfill = elapsed.as_millis()
                        }

                        let now = Instant::now();
                        let decomposer: HeuristicEliminationDecomposer<MLSelector> =
                            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
                        let x = decomposer.compute().computed_tree_decomposition();
                        if let Some(decomposition_ml) = x {
                            let elapsed = now.elapsed();
                            println!("ML: {:?}: {:?}", decomposition_ml.max_bag_size, elapsed);
                            ml = decomposition_ml.max_bag_size;
                            elapsed_ml = elapsed.as_millis()
                        }

                        // // (["file", "mindegree", "minfill", "ml", "time"]
                        let filename = entry.file_name().to_string_lossy().into_owned();
                        let mindegree_str = format!("{:?}", mindegree);
                        let minfill_str = format!("{:?}", minfill);
                        let ml_str = format!("{:?}", ml);

                        let elapsed_mindegree_str = elapsed_mindegree.to_string();
                        let elapsed_minfill_str = elapsed_minfill.to_string();
                        let elapsed_ml_str = elapsed_ml.to_string();

                        let record = &[
                            filename,
                            mindegree_str,
                            elapsed_mindegree_str,
                            minfill_str,
                            elapsed_minfill_str,
                            ml_str,
                            elapsed_ml_str,
                        ];
                        wtr.write_record(record).unwrap();
                        wtr.flush()?;
                        wtr.flush()?;
                    }
                    Err(e) => {
                        println!("read error: {:?}", e);
                    }
                }
            }
            Err(e) => {
                println!("read error: {:?}", e);
            }
        }
    }

    Ok(())
}

fn ml_values(graph: &HashMapGraph, cache: &mut [i64]) -> io::Result<()> {
    let mut child = Command::new("uv")
        .arg("run")
        .arg("--directory")
        .arg("/home/mhbr96/Python/tw_bnb/") // change this to correct path
        .arg("deserialize_msgpack.py")
        .stdin(Stdio::piped()) // write to stdin
        .stdout(Stdio::piped()) // read from stdout
        .spawn()
        .expect("Failed to start Python process");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&graph.serialize())?; // Write serialized graph data to stdin
        stdin.flush()?;
    }

    let mut output = Vec::new();
    if let Some(ref mut stdout) = child.stdout {
        stdout.read_to_end(&mut output)?; // Read all data from stdout
    }

    let results: Vec<Tuple> = rmp_serde::from_slice(&output).expect("Failed to deserialize output");

    let status = child
        .wait()
        .expect("Failed to wait for Python process to exit");

    if status.success() {
        for t in results.iter().cloned() {
            cache[t.0 as usize] = t.1
        }
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Python program failed",
        ))
    }
}
