use arboretum_td::graph::{BaseGraph, MutableGraph};
use arboretum_td::heuristic_elimination_order::{
    HeuristicEliminationDecomposer, MinDegreeSelector, MinFillSelector, Selector,
};
use arboretum_td::solver::AtomSolver;
use arboretum_td::{graph::HashMapGraph, io::PaceReader};
use fxhash::FxHashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::io::BufWriter;
use std::io::Write;
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
use std::time::Instant;
use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufReader, Read},
};

lazy_static! {
    static ref GLOBAL_UNIX_STREAM: Mutex<Option<UnixStream>> = Mutex::new(None);
}

const END_MARKER: &[u8] = b"<END>";

fn set_global_connection(socket_path: &str) {
    let stream = UnixStream::connect(socket_path).expect("Failed to connect to the server");
    let mut global_stream = GLOBAL_UNIX_STREAM.lock().unwrap();
    *global_stream = Some(stream);
}

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
    let socket_path = "/tmp/server_socket";
    set_global_connection(socket_path);

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

                        // let now = Instant::now();
                        // let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
                        //     HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
                        // let x = decomposer.compute().computed_tree_decomposition();
                        // if let Some(decomposition_mindegree) = x {
                        //     let elapsed = now.elapsed();
                        //     println!(
                        //         "Min Degree: {:?}: {:?}",
                        //         decomposition_mindegree.max_bag_size, elapsed
                        //     );
                        //     mindegree = decomposition_mindegree.max_bag_size;
                        //     elapsed_mindegree = elapsed.as_millis();
                        // }

                        // let now = Instant::now();
                        // let decomposer: HeuristicEliminationDecomposer<MinFillSelector> =
                        //     HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
                        // let x = decomposer.compute().computed_tree_decomposition();
                        // if let Some(decomposition_minfill) = x {
                        //     let elapsed = now.elapsed();
                        //     println!(
                        //         "Min Fill: {:?}: {:?}",
                        //         decomposition_minfill.max_bag_size, elapsed
                        //     );
                        //     minfill = decomposition_minfill.max_bag_size;
                        //     elapsed_minfill = elapsed.as_millis()
                        // }

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

fn read_until_marker(mut stream: &mut UnixStream) -> Vec<u8> {
    let mut reader = BufReader::new(&mut stream);
    let mut buffer = Vec::new();
    let mut chunk = [0; 4096];

    loop {
        let bytes_read = reader.read(&mut chunk).unwrap();
        if bytes_read == 0 {
            break;
        }

        // Append the read data into the main buffer
        buffer.extend_from_slice(&chunk[..bytes_read]);

        // Check if the end marker exists in the buffer
        if buffer
            .windows(END_MARKER.len())
            .any(|window| window == END_MARKER)
        {
            // Remove the end marker from the data
            let marker_pos = buffer
                .windows(END_MARKER.len())
                .position(|window| window == END_MARKER)
                .unwrap();
            buffer.truncate(marker_pos);
            break;
        }
    }

    buffer // Return the message buffer without the end marker
}

fn ml_values(graph: &HashMapGraph, cache: &mut [i64]) -> io::Result<()> {
    let mut global_stream = GLOBAL_UNIX_STREAM.lock().unwrap();
    if let Some(ref mut stream) = *global_stream {
        let mut serialized_graph = graph.serialize();
        serialized_graph.extend_from_slice(END_MARKER);

        stream.write_all(&serialized_graph)?;
        stream.flush()?;

        let output = read_until_marker(stream);
        // let mut output = Vec::new();
        // stream.read_to_end(&mut output)?;

        let results: Vec<Tuple> = rmp_serde::from_slice(&output).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Failed to deserialize output")
        })?;

        for t in results {
            cache[t.0 as usize] = t.1;
        }

        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "No global TCP stream available",
        ))
    }
}
