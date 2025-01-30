use arboretum_td::graph::BaseGraph;
use arboretum_td::heuristic_elimination_order::{
    set_socket_path, DegreeMLSelector, HeuristicEliminationDecomposer, MinDegreeMLSelector,
    MinDegreeSelector, MinFillMLSelector, MinFillSelector,
};
use arboretum_td::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use arboretum_td::solver::AtomSolver;
use arboretum_td::{graph::HashMapGraph, io::PaceReader};
use std::time::Instant;
use std::{
    convert::TryFrom,
    io::{self, BufReader},
};

fn main() -> io::Result<()> {
    let path = std::env::args().nth(1).expect("no path given");
    let epsilon: f64 = std::env::args()
        .nth(2)
        .expect("no pattern given")
        .parse()
        .unwrap();
    let c: f64 = std::env::args()
        .nth(3)
        .expect("no path given")
        .parse()
        .unwrap();

    let port: u32 = std::env::args()
        .nth(4)
        .expect("no port given")
        .parse()
        .unwrap();

    // Dirty hack to achive concurrent processing via multiple socket files
    let socket_path = format!("/tmp/tw_server_socket_{port}");
    set_socket_path(&socket_path);

    let mut mindegree = 0;
    let mut elapsed_mindegree = 0;
    let mut minfill = 0;
    let mut elapsed_minfill = 0;
    let mut ml_md = 0;
    let mut elapsed_ml_md = 0;
    let mut ml_mf = 0;
    let mut elapsed_ml_mf = 0;

    let file = std::fs::File::open(path).unwrap();
    let reader = PaceReader(BufReader::new(file));
    let graph = HashMapGraph::try_from(reader)?;
    let lb = MinorMinWidth::with_graph(&graph).compute();

    let now = Instant::now();
    let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
        HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
    let x = decomposer.compute().computed_tree_decomposition();
    if let Some(decomposition_mindegree) = x {
        let elapsed = now.elapsed();
        mindegree = decomposition_mindegree.max_bag_size;
        elapsed_mindegree = elapsed.as_nanos();
    }

    let now = Instant::now();
    let decomposer: HeuristicEliminationDecomposer<MinFillSelector> =
        HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
    let x = decomposer.compute().computed_tree_decomposition();
    if let Some(decomposition_minfill) = x {
        let elapsed = now.elapsed();
        minfill = decomposition_minfill.max_bag_size;
        elapsed_minfill = elapsed.as_nanos()
    }

    let now = Instant::now();
    let decomposer: MinDegreeMLSelector =
        HeuristicEliminationDecomposer::with_bounds(&graph, lb, graph.order());

    let x = decomposer.compute_order_and_decomposition();
    if let Some(decomposition_ml) = x {
        let elapsed = now.elapsed();
        ml_md = decomposition_ml.tree_decomposition.max_bag_size;
        elapsed_ml_md = elapsed.as_nanos()
    }

    let now = Instant::now();
    let decomposer: MinFillMLSelector =
        HeuristicEliminationDecomposer::with_bounds(&graph, lb, graph.order());

    let x = decomposer.compute_order_and_decomposition();
    if let Some(decomposition_ml) = x {
        let elapsed = now.elapsed();
        ml_mf = decomposition_ml.tree_decomposition.max_bag_size;
        elapsed_ml_mf = elapsed.as_nanos()
    }

    let data = serde_json::json!({
        "mindegree": mindegree,
        "elapsed_mindegree": elapsed_mindegree,
        "minfill": minfill,
        "elapsed_minfill": elapsed_minfill,
        "ml_md": ml_md,
        "elapsed_ml_md": elapsed_ml_md,
        "ml_mf": ml_mf,
        "elapsed_ml_mf": elapsed_ml_mf,
    });
    println!("{}", data);

    Ok(())
}
