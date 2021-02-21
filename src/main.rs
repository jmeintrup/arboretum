use arboretum::graph::HashMapGraph;
use arboretum::io::{PaceReader, PaceWriter};
use arboretum::solver::Solver;
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use structopt::StructOpt;

#[cfg(log)]
use log::info;

use arboretum::SafeSeparatorLimits;
#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(feature = "jemallocator")]
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

    /// Mode. 'heuristic', 'exact' or 'auto'. Defaults to Exact.
    /// Any invalid input fails silently to 'exact'.
    #[structopt(short, long)]
    mode: Option<String>,

    /// Seed used for all rng. Unsigned 64bit integer value.
    #[structopt(short, long, default = "0")]
    seed: u64,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();
    let mode: &str = match opt.mode {
        Some(value) => match value.as_str() {
            "heuristic" => "heuristic",
            "auto" => "auto",
            "exact" => "exact",
            _ => "exact",
        },
        None => "exact",
    };

    #[cfg(feature = "handle-ctrlc")]
    arboretum::signals::initialize();

    #[cfg(feature = "log")]
    #[cfg(feature = "env_logger")]
    arboretum::log::build_pace_logger();

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

    let file = match opt.output {
        Some(path) => Some(OpenOptions::new().write(true).create(true).open(path)?),
        None => None,
    };

    let td = match mode {
        "heuristic" => {
            #[cfg(log)]
            info!("Running in default heuristic mode.");
            Solver::default_heuristic()
                .use_min_degree_for_minor_safe(true)
                .seed(Some(opt.seed))
                .solve(&graph)
        }
        "auto" => {
            #[cfg(log)]
            info!("Running in default auto mode.");
            Solver::auto(&graph)
                .use_min_degree_for_minor_safe(true)
                .seed(Some(opt.seed))
                .solve(&graph)
        }
        _ => {
            #[cfg(log)]
            info!("Running in default exact mode.");
            Solver::default_exact()
                .seed(Some(opt.seed))
                .safe_separator_limits(
                    SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                )
                .solve(&graph)
        }
    };

    match file {
        Some(file) => PaceWriter::new(&td, &graph, file).output(),
        None => {
            let writer = stdout();
            PaceWriter::new(&td, &graph, writer).output()
        }
    }
}
