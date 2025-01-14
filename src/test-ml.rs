use arboretum_td::graph::{BaseGraph, MutableGraph};
use arboretum_td::heuristic_elimination_order::{
    HeuristicEliminationDecomposer, MinDegreeSelector, Selector,
};
use arboretum_td::solver::AtomSolver;
use arboretum_td::{graph::HashMapGraph, io::PaceReader};
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

pub type MLDecomposer = HeuristicEliminationDecomposer<MLSelector>;

fn main() -> io::Result<()> {
    let path = Path::new("data/ex192.gr");
    if path.exists() {
        // Open the file and wrap it in a BufReader
        let file = File::open(path)?;
        let reader = PaceReader(BufReader::new(file));

        // Attempt to construct the graph from the reader
        let graph = HashMapGraph::try_from(reader)?;

        let now = Instant::now();
        let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
        let x = decomposer.compute().computed_tree_decomposition();

        if let Some(decomposition) = x {
            let elapsed = now.elapsed();
            println!("{:?}: {:?}", decomposition.max_bag_size, elapsed);
        }

        let now = Instant::now();
        let decomposer: HeuristicEliminationDecomposer<MLSelector> =
            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
        let x = decomposer.compute().computed_tree_decomposition();
        if let Some(decomposition) = x {
            let elapsed = now.elapsed();
            println!("{:?}: {:?}", decomposition.max_bag_size, elapsed);
        }

        Ok(())
    } else {
        eprintln!("File does not exist: {:?}", path);
        Err(io::Error::new(io::ErrorKind::Other, "File not found"))
    }
}

fn ml_values(graph: &HashMapGraph, cache: &mut [i64]) -> io::Result<()> {
    let mut child = Command::new("uv")
        .arg("run")
        .arg("--directory")
        .arg("/home/mhbr96/Python/tw_bnb/")
        .arg("deserialize_msgpack.py")
        .stdin(Stdio::piped()) // write to stdin
        .stdout(Stdio::piped()) // read from stdout
        .spawn()
        .expect("Failed to start Python process");

    // println!("{:?}", graph);
    // println!("{:?}", graph.order());
    // let l = graph.serialize();
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&graph.serialize())?; // Write serialized graph data to stdin
        stdin.flush()?;
    }

    let mut output = Vec::new();
    if let Some(ref mut stdout) = child.stdout {
        stdout.read_to_end(&mut output)?; // Read all data from stdout
    }

    let status = child
        .wait()
        .expect("Failed to wait for Python process to exit");

    if status.success() {
        assert!(output.len() % 16 == 0); // Ensure input is a multiple of 16 bytes
        let pairs: Vec<(i64, i64)> = output
            .chunks_exact(16) // Iterator over 16-byte chunks (since each tuple is two i64s)
            .map(|chunk| {
                let node_bytes = <[u8; 8]>::try_from(&chunk[0..8]).expect("First 8 bytes invalid");
                let res_bytes = <[u8; 8]>::try_from(&chunk[8..16]).expect("Second 8 bytes invalid");
                let x = i64::from_le_bytes(node_bytes);
                let y = i64::from_le_bytes(res_bytes);
                (x, y) // Return tuple (x, y)
            })
            .collect();

        for (node, value) in pairs.iter().cloned() {
            cache[node as usize] = value
        }
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Python program failed",
        ))
    }
}
