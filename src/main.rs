use arboretum::graph::HashMapGraph;
use arboretum::io::{PaceReader, PaceWriter};
use arboretum::solver::{AlgorithmTypes, AtomSolverType, Solver, UpperboundHeuristicType};
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use structopt::StructOpt;

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "arboretum-cli",
    about = "Computes Tree Decompositions for a given input graph."
)]
struct Opt {
    /// Input file, using the graph format of the PACE 2021 challenge.
    /// `stdin` if not specified.
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file. `stdout` if not specified.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Mode. Heuristic or Exact. Defaults to exact.
    #[structopt(short, long)]
    heuristic: bool,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();

    let graph: HashMapGraph = match opt.input {
        Some(path) => {
            let file = File::open(path)?;
            let reader = PaceReader(BufReader::new(file));
            HashMapGraph::try_from(reader)?
        }
        None => {
            let stdin = stdin();
            let reader = PaceReader(stdin.lock());
            HashMapGraph::try_from(reader)?
        }
    };

    let td = if opt.heuristic {
        println!("c Running in default heuristic mode.");
        Solver::default_heuristic().solve(&graph)
    } else {
        println!("c Running in default exact mode.");
        Solver::default_exact().solve(&graph)
    };

    match opt.output {
        Some(path) => {
            let writer = OpenOptions::new().write(true).create(true).open(path)?;
            PaceWriter::new(&td, &graph, writer).output()
        }
        None => {
            let writer = stdout();
            PaceWriter::new(&td, &graph, writer).output()
        }
    }
}
