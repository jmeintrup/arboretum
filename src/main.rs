use arboretum_td::graph::HashMapGraph;
use arboretum_td::io::{PaceReader, PaceWriter};
use arboretum_td::solver::{AlgorithmTypes, AtomSolverType, Solver};
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use structopt::StructOpt;

#[cfg(log)]
use log::info;

use arboretum_td::SafeSeparatorLimits;
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

    /// Seed used for all rng. Unsigned 64bit integer value. Defaults to '0' if missing.
    #[structopt(short, long)]
    seed: Option<u64>,

    /// Optional timeout value for heuristic algorithm. In heuristic mode
    /// the CLI stops on ctrl+c and outputs the current best solution. This might take a few seconds or minutes depending
    /// on the size of the input graph. When timeout is set, the algorithm tries to optimize a solution until the timeout is reached.
    #[structopt(short, long)]
    timeout: Option<u64>,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();
    let mode: &str = match opt.mode {
        Some(value) => match value.as_str() {
            "heuristic" => "heuristic",
            "auto" => "auto",
            "exact" => "exact",
            "bb" => "bb",
            _ => "exact",
        },
        None => "exact",
    };

    #[cfg(feature = "log")]
    #[cfg(feature = "env_logger")]
    arboretum_td::log::build_pace_logger();

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
            #[cfg(feature = "handle-ctrlc")]
            arboretum_td::signals::initialize();

            let timeout: Option<u64> = opt.timeout;
            let use_timeout = timeout.is_some();
            if let Some(timeout) = timeout {
                arboretum_td::timeout::initialize_timeout(timeout);
            }

            if use_timeout {
                #[cfg(log)]
                info!("Running in default timeout heuristic mode.");

                #[cfg(log)]
                info!("Initializing a timeout for {} seconds.", timeout);
                Solver::default_heuristic()
                    .safe_separator_limits(
                        SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                    )
                    .seed(opt.seed)
                    .algorithm_types(
                        AlgorithmTypes::default()
                            .atom_solver(AtomSolverType::TabuLocalSearchInfinite),
                    )
                    .solve(&graph)
            } else {
                #[cfg(log)]
                info!("Running in default heuristic mode.");
                Solver::default_heuristic()
                    .safe_separator_limits(
                        SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                    )
                    .seed(opt.seed)
                    .solve(&graph)
            }
        }
        "auto" => {
            #[cfg(log)]
            info!("Running in default auto mode.");
            #[cfg(feature = "handle-ctrlc")]
            arboretum_td::signals::initialize();

            let timeout: Option<u64> = opt.timeout;
            let use_timeout = timeout.is_some();
            if let Some(timeout) = timeout {
                #[cfg(log)]
                info!("Initializing a timeout for {} seconds.", timeout);

                arboretum_td::timeout::initialize_timeout(timeout);
            }

            let mut td = Solver::auto(&graph)
                .safe_separator_limits(
                    SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                )
                .seed(opt.seed)
                .solve(&graph);
            if use_timeout && !arboretum_td::timeout::timeout() {
                let td2 = Solver::default_heuristic()
                    .algorithm_types(
                        AlgorithmTypes::default()
                            .atom_solver(AtomSolverType::TabuLocalSearchInfinite),
                    )
                    .safe_separator_limits(
                        SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                    )
                    .seed(opt.seed)
                    .solve(&graph);
                if td2.max_bag_size < td.max_bag_size {
                    td = td2;
                };
            }
            td
        }
        "bb" => {
            #[cfg(log)]
            info!("Running in quickBB mode.");
            Solver::default_exact()
                .seed(opt.seed)
                .safe_separator_limits(
                    SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                )
                .algorithm_types(AlgorithmTypes::default().atom_solver(AtomSolverType::QuickBB))
                .solve(&graph)
        }
        _ => {
            #[cfg(log)]
            info!("Running in default exact mode.");
            Solver::default_exact()
                .seed(opt.seed)
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
