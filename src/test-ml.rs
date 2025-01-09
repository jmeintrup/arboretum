use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufReader},
    path::Path,
};

use arboretum_td::{
    graph::{BaseGraph, HashMapGraph},
    heuristic_elimination_order::{HeuristicEliminationDecomposer, MinDegreeSelector},
    io::PaceReader,
    solver::AtomSolver,
};

fn main() -> io::Result<()> {
    let path = Path::new("data/ex008.gr");
    if path.exists() {
        let file = File::open(path)?;
        let reader = PaceReader(BufReader::new(file));
        let graph = HashMapGraph::try_from(reader)?;

        let decomposer: HeuristicEliminationDecomposer<MinDegreeSelector> =
            HeuristicEliminationDecomposer::with_bounds(&graph, 0, graph.order());
        let x = decomposer.compute().computed_tree_decomposition();

        if x.is_some() {
            println!("{:?}", x.unwrap().max_bag_size)
        }

        print!("{:?}", graph.serialize())
    }

    Ok(())
}
