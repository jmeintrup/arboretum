use arboretum::treewidth::heuristics::{min_degree, min_fill};
use arboretum::util::elimination_order::{get_width, get_width_virtual_elimination};
use arboretum::util::io::Reader;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::fs::File;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn adjacency_matrix(c: &mut Criterion) {
    let file = File::open("data/david.col").unwrap();
    let reader = Reader::new(file);
    let graph = reader.read_graph_to_adjacency_matrix_graph().unwrap();

    let order = min_degree(&graph);

    c.bench_function("adjacency_matrix_min_degree", |b| {
        b.iter(|| min_degree(&graph))
    });

    c.bench_function("adjacency_matrix_min_fill", |b| b.iter(|| min_fill(&graph)));

    c.bench_function("adjacency_matrix_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });

    c.bench_function("adjacency_matrix_virtual_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });
}

fn array_matrix_combo_graph(c: &mut Criterion) {
    let file = File::open("data/david.col").unwrap();
    let reader = Reader::new(file);
    let graph = reader.read_graph_to_array_matrix_combo_graph().unwrap();

    let order = min_degree(&graph);

    c.bench_function("array_matrix_combo_graph_min_degree", |b| {
        b.iter(|| min_degree(&graph))
    });

    c.bench_function("array_matrix_combo_graph_min_fill", |b| {
        b.iter(|| min_fill(&graph))
    });

    c.bench_function("array_matrix_combo_graph_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });

    c.bench_function("array_matrix_combo_graph_virtual_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });
}

fn hashmap_graph(c: &mut Criterion) {
    let file = File::open("data/david.col").unwrap();
    let reader = Reader::new(file);
    let graph = reader.read_graph_to_hashmap_graph().unwrap();

    let order = min_degree(&graph);

    c.bench_function("hashmap_graph_min_degree", |b| {
        b.iter(|| min_degree(&graph))
    });

    c.bench_function("hashmap_graph_min_fill", |b| b.iter(|| min_fill(&graph)));

    c.bench_function("hashmap_graph_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });

    c.bench_function("hashmap_graph_virtual_elim", |b| {
        b.iter(|| get_width_virtual_elimination(&graph, &order.order()))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = hashmap_graph, array_matrix_combo_graph
}
criterion_main!(benches);
