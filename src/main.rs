use arboretum::graph::bag::TreeDecomposition;
use arboretum::graph::graph::Graph;
use arboretum::graph::hash_map_graph::HashMapGraph;
use arboretum::io::PaceReader;
use arboretum::solver::SolverBuilder;
use std::convert::TryFrom;
use std::io;
use std::io::stdin;

fn main() -> io::Result<()> {
    let graph: HashMapGraph = {
        let buffer = stdin();
        let reader = PaceReader(buffer.lock());
        HashMapGraph::try_from(reader)?
    };

    let solver = SolverBuilder::new().build();
    print_pace_td(&solver.solve(&graph), &graph);
    Ok(())
}

/*fn main() -> io::Result<()> {
    let graph: HashMapGraph = {
        let buffer = stdin();
        let reader = PaceReader(buffer.lock());
        HashMapGraph::try_from(reader)?
    };
    if graph.order() <= 2 {
        let mut td = TreeDecomposition::new();
        let vertices: FnvHashSet<_> = graph.vertices().collect();
        if vertices.len() > 0 {
            td.add_bag(vertices);
        }
        print_pace_td(&td, &graph);
        return Ok(());
    }

    let mut reducer = RuleBasedPreprocessor::new(&graph);
    let m: usize = graph.vertices().map(|v| graph.degree(v)).sum();
    println!("c reducing graph with n = {} m = {}", graph.order(), m);
    let now = SystemTime::now();
    reducer.preprocess();
    println!("c reduced in {}ms", now.elapsed().unwrap().as_millis());
    println!("c lb {}", reducer.lower_bound);
    let reduced_graph = reducer.graph();
    let m: usize = reduced_graph
        .vertices()
        .map(|v| reduced_graph.degree(v))
        .sum();
    println!(
        "c reduced graph has n = {} m = {}",
        reduced_graph.order(),
        m
    );

    let td = if reduced_graph.order() == 0 {
        reducer.into_td()
    } else {
        let framework = SafeSeparatorFramework::new(reduced_graph.clone(), 4);
        let now = SystemTime::now();
        let result = framework.compute();
        println!("c {:?}", result.decomposition_information);
        let tmp = result.tree_decomposition;
        if let Err(e) = tmp.verify(reduced_graph) {
            println!("c Not valid!: {}", e);
        } else {
            println!("c reduced is valid");
        }
        let after = now.elapsed().unwrap();
        println!("c solved reduced in: {}ms", after.as_millis());
        reducer.combine_into_td(tmp)
    };
    if let Err(e) = td.verify(&graph) {
        println!("c Not valid!: {}", e);
    }
    print_pace_td(&td, &graph);
    Ok(())
}*/

pub fn print_pace_td<G: Graph>(td: &TreeDecomposition, graph: &G) {
    let bag_count: usize = td.bags.len();
    let max_bag: usize = td.max_bag_size;
    println!("s td {} {} {}", bag_count, max_bag, graph.order());
    td.bags().iter().for_each(|b| {
        let mut tmp: Vec<_> = b.vertex_set.iter().copied().collect();
        tmp.sort();
        let vertices: Vec<_> = tmp.iter().map(|i| (i + 1).to_string()).collect();
        let vertices = vertices.iter().fold(String::new(), |mut acc, v| {
            acc.push_str(v.as_str());
            acc.push_str(" ");
            acc
        });
        println!("b {} {}", b.id + 1, vertices);
    });
    td.bags().iter().for_each(|b| {
        for child in b.neighbors.iter().copied().filter(|i| *i > b.id) {
            println!("{} {}", b.id + 1, child + 1);
        }
    });
}
