use crate::exact::pid::PID;
use crate::exact::ExactSolver;
use crate::graph::bag::TreeDecomposition;
use crate::graph::graph::Graph;
use crate::graph::hash_map_graph::HashMapGraph;
use crate::graph::mutable_graph::MutableGraph;
use crate::lowerbound::{LowerboundHeuristic, MinorMinWidth};
use crate::upperbound::{HeuristicEliminationOrderDecomposer, MinDegreeStrategy, MinFillStrategy, UpperboundHeuristic, heuristic_elimination_decompose, MinFillSelector};
use fnv::FnvHashSet;
use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, RefCell};
use std::cmp::max;
use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

pub trait Preprocessor {
    fn preprocess(&mut self);

    fn combine_into_td(self, td: TreeDecomposition, graph: &HashMapGraph) -> TreeDecomposition;

    fn into_td(self) -> TreeDecomposition;

    fn graph(&self) -> &HashMapGraph;
}

#[inline]
fn eliminate(v: usize, graph: &mut HashMapGraph, stack: &mut Vec<FnvHashSet<usize>>) {
    let mut bag = graph.neighborhood_set(v).clone();
    bag.insert(v);
    stack.push(bag);
    graph.eliminate_vertex(v);
}

struct Cube {
    a: usize,
    b: usize,
    c: usize,
    x: usize,
    y: usize,
    z: usize,
    v: usize,
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Debug)]
enum SeparationLevel {
    Connected,
    BiConnected,
    TriConnected,
    Clique,
    AlmostClique,
    MinorSafeClique,
    Atomic,
}

impl SeparationLevel {
    pub fn increment(&self) -> Option<Self> {
        match self {
            Self::Connected => Some(Self::BiConnected),
            Self::BiConnected => Some(Self::TriConnected),
            Self::TriConnected => Some(Self::Clique),
            Self::Clique => Some(Self::AlmostClique),
            Self::AlmostClique => Some(Self::MinorSafeClique),
            Self::MinorSafeClique => Some(Self::Atomic),
            Self::Atomic => None,
        }
    }
}

pub struct RuleBasedPreprocessor {
    stack: Vec<FnvHashSet<usize>>,
    pub lower_bound: usize,
    partial_tree_decomposition: TreeDecomposition,
    processed_graph: HashMapGraph,
}

impl Preprocessor for RuleBasedPreprocessor {
    fn preprocess(&mut self) {
        self.lower_bound = 0;
        if self.processed_graph.order() == 0 {
            return;
        }

        self.remove_low_degree();
        if self.processed_graph.order() == 0 {
            self.process_stack();
            return;
        }
        /* self.remove_low_fill_in();
        if self.processed_graph.order() == 0 {
            self.process_stack();
            return;
        }*/
        while self.apply_rules() {}
        if self.processed_graph.order() == 0 {
            self.process_stack();
        }
    }

    fn combine_into_td(mut self, mut td: TreeDecomposition, graph: &HashMapGraph) -> TreeDecomposition {
        //println!("c Combining");
        //self.partial_tree_decomposition = td;
        let mut vertices = FnvHashSet::default();
        for bag in &td.bags {
            vertices.extend(bag.vertex_set.iter().copied())
        }
        let tmp = self.stack.clone();
        self.stack.push(vertices);
        self.process_stack();
        assert!(self.partial_tree_decomposition.verify(graph).is_ok());
        //self.partial_tree_decomposition.flatten();
        /*for bag in &self.partial_tree_decomposition.bags {
            println!("c id: {} len: {} lb: {}", bag.id, bag.vertex_set.len(), self.lower_bound)
        }*/
        self.partial_tree_decomposition.flatten();
        self.partial_tree_decomposition.replace_bag(0, td);
        /*self.stack = tmp;
        self.partial_tree_decomposition = td;
        self.process_stack();*/
        self.partial_tree_decomposition
    }

    fn into_td(mut self) -> TreeDecomposition {
        self.process_stack();
        self.partial_tree_decomposition
    }

    fn graph(&self) -> &HashMapGraph {
        &self.processed_graph
    }
}

impl RuleBasedPreprocessor {
    pub fn new(graph: &HashMapGraph) -> Self {
        Self {
            stack: vec![],
            lower_bound: 0,
            partial_tree_decomposition: TreeDecomposition::new(),
            processed_graph: graph.clone(),
        }
    }

    fn apply_rules(&mut self) -> bool {
        // islet
        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 0);
        if let Some(v) = found {
            let mut bag = FnvHashSet::with_capacity_and_hasher(1, Default::default());
            bag.insert(v);
            self.stack.push(bag);
            self.processed_graph.remove_vertex(v);
            return true;
        }
        // single-degree

        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 1);
        if let Some(v) = found {
            let mut bag = self.processed_graph.neighborhood_set(v).clone();
            bag.insert(v);
            self.stack.push(bag);
            self.processed_graph.remove_vertex(v);
            return true;
        }
        // series
        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.degree(*v) == 2);
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            self.lower_bound = max(self.lower_bound, 3);
            return true;
        }
        // triangle rule
        self.lower_bound = max(self.lower_bound, 4);

        let found = self
            .processed_graph
            .vertices()
            .filter(|v| self.processed_graph.degree(*v) == 3)
            .find(|v| {
                let tmp: Vec<_> = self
                    .processed_graph
                    .neighborhood_set(*v)
                    .iter()
                    .copied()
                    .collect();
                self.processed_graph.has_edge(tmp[0], tmp[1])
                    || self.processed_graph.has_edge(tmp[0], tmp[2])
                    || self.processed_graph.has_edge(tmp[1], tmp[2])
            });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        // buddy rule
        let found = self.processed_graph.vertices().find(|v| {
            self.processed_graph
                .vertices()
                .find(|u| {
                    v < u
                        && self.processed_graph.degree(*v) == 3
                        && self.processed_graph.degree(*u) == 3
                        && self.processed_graph.neighborhood_set(*u)
                            == self.processed_graph.neighborhood_set(*v)
                })
                .is_some()
        });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        // cube rule
        let cube: Option<Cube> = None;
        for v in self.processed_graph.vertices() {
            if self.processed_graph.degree(v) != 3 {
                continue;
            }
            let nb: Vec<_> = self
                .processed_graph
                .neighborhood_set(v)
                .iter()
                .copied()
                .collect();
            if nb
                .iter()
                .find(|x| self.processed_graph.degree(**x) != 3)
                .is_none()
            {
                continue;
            }
            let (x, y, z) = (nb[0], nb[1], nb[2]);
            let x_nb: Vec<_> = self
                .processed_graph
                .neighborhood_set(x)
                .iter()
                .copied()
                .collect();
            let mut a = if x_nb[0] == v { x_nb[2] } else { x_nb[0] };
            let mut b = if x_nb[1] == v { x_nb[2] } else { x_nb[1] };
            if !self.processed_graph.has_edge(y, a) || !self.processed_graph.has_edge(z, b) {
                std::mem::swap(&mut a, &mut b);
            }
            if !self.processed_graph.has_edge(y, a) || !self.processed_graph.has_edge(z, b) {
                continue;
            }
            let c_option = self
                .processed_graph
                .neighborhood(y)
                .find(|c| *c != v && self.processed_graph.has_edge(z, *c));
            if c_option.is_none() {
                return false;
            }
            let c = c_option.unwrap();

            let cube = Some(Cube {
                a,
                b,
                c,
                x,
                y,
                z,
                v,
            });
            break;
        }
        if let Some(cube) = cube {
            let mut bag = FnvHashSet::with_capacity_and_hasher(4, Default::default());
            bag.insert(cube.b);
            bag.insert(cube.z);
            bag.insert(cube.v);
            bag.insert(cube.c);
            self.processed_graph.eliminate_vertex(cube.z);
            self.processed_graph.add_edge(cube.a, cube.b);
            self.processed_graph.add_edge(cube.a, cube.c);
            self.processed_graph.add_edge(cube.a, cube.v);
            self.processed_graph.add_edge(cube.b, cube.c);
            self.processed_graph.add_edge(cube.b, cube.v);
            self.processed_graph.add_edge(cube.c, cube.v);
            self.stack.push(bag);

            return true;
        }

        let found = self
            .processed_graph
            .vertices()
            .find(|v| self.processed_graph.is_simplicial(*v));
        if let Some(v) = found {
            self.lower_bound = max(self.lower_bound, self.processed_graph.degree(v));
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        let found = self.processed_graph.vertices().find(|v| {
            self.lower_bound > self.processed_graph.degree(*v)
                && self.processed_graph.is_almost_simplicial(*v)
        });
        if let Some(v) = found {
            eliminate(
                v,
                self.processed_graph.borrow_mut(),
                self.stack.borrow_mut(),
            );
            return true;
        }
        false
    }

    fn process_stack(&mut self) {
        for new_bag in self.stack.drain(..).rev() {
            if let Some(old_bag) = self.partial_tree_decomposition.dfs().find(|old_bag| {
                new_bag
                    .iter()
                    .filter(|v| !old_bag.vertex_set.contains(*v))
                    .count()
                    <= 1
            }) {
                let old_id = old_bag.id;
                let id = self.partial_tree_decomposition.add_bag(new_bag);
                self.partial_tree_decomposition.add_edge(old_id, id);
            } else {
                self.partial_tree_decomposition.add_bag(new_bag);
            }
        }
    }

    fn remove_low_degree(&mut self) {
        // remove all low degree vertices. First remove all islets, then all islets and 1-degree,
        // then all islets, 1-degree and 2-degree
        for d in 0..3 {
            let mut visited: FnvHashSet<usize> = FnvHashSet::with_capacity_and_hasher(
                self.processed_graph.order(),
                Default::default(),
            );
            let mut queue = VecDeque::new();
            self.processed_graph
                .vertices()
                .filter(|v| self.processed_graph.degree(*v) <= d)
                .for_each(|v| {
                    visited.insert(v);
                    queue.push_back(v);
                });

            while let Some(v) = queue.pop_front() {
                let mut nb: FnvHashSet<_> = self.processed_graph.neighborhood(v).collect();
                if nb.len() > d {
                    continue;
                }
                self.processed_graph.eliminate_vertex(v);
                nb.iter().copied().for_each(|v| {
                    if self.processed_graph.has_vertex(v)
                        && self.processed_graph.degree(v) <= d
                        && !visited.contains(&v)
                    {
                        queue.push_back(v);
                        visited.insert(v);
                    }
                });
                nb.insert(v);
                self.lower_bound = max(self.lower_bound, nb.len() - 1);
                self.stack.push(nb);
            }
        }
    }
}

struct SearchState {
    graph: HashMapGraph,
    separation_level: SeparationLevel,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
}

impl SearchState {
    pub fn process_separator(&mut self, separator: &FnvHashSet<usize>) -> TreeDecomposition {
        let mut td = TreeDecomposition::new();
        let root = td.add_bag(separator.clone());
        if td.max_bag_size > 0 {
            let lb: &Cell<_> = self.lower_bound.borrow();
            lb.set(max(lb.get(), td.max_bag_size - 1));
        }
        for cc in self.graph.separate(&separator).iter() {
            let mut graph = self.graph.vertex_induced(cc);
            for v in separator.iter() {
                for u in separator.iter().filter(|u| v < *u) {
                    graph.add_edge(*v, *u);
                }
                for u in self
                    .graph
                    .neighborhood_set(*v)
                    .iter()
                    .filter(|u| cc.contains(*u))
                {
                    graph.add_edge(*v, *u);
                }
            }
            let mut search_sate = SearchState {
                graph,
                separation_level: self.separation_level,
                lower_bound: self.lower_bound.clone(),
                log_state: self.log_state.clone(),
            };
            let partial = search_sate.search();
            td.combine_with(root, partial);
            /*if td.max_bag_size > 0 {
                let lb: &Cell<_> = self.lower_bound.borrow();
                lb.set(max(lb.get(), td.max_bag_size - 1));
            }*/
        }
        return td;
    }

    pub fn search(&mut self) -> TreeDecomposition {
        let lowerbound = {
            let tmp: &Cell<_> = self.lower_bound.borrow();
            tmp.get()
        };
        if self.graph.order() <= lowerbound + 1 {
            let mut td = TreeDecomposition::new();
            td.add_bag(self.graph.vertices().collect());
            return td;
        }
        while self.separation_level != SeparationLevel::Atomic {
            if let Some(separator) = self.find_separator() {
                println!("c found safe separator: {:?}", self.separation_level);
                let mut log_state = self.log_state.get();
                log_state.increment(self.separation_level);
                self.log_state.set(log_state);
                return self.process_separator(&separator);
            } else {
                self.separation_level = self.separation_level.increment().unwrap();
            }
        }
        let mut log_state = self.log_state.get();
        log_state.increment(self.separation_level);
        self.log_state.set(log_state);
        //todo: add solver builder that returns a solver that is then called on the graph
        let upperbound = heuristic_elimination_decompose::<MinFillSelector>(self.graph.clone());
        let mmw = MinorMinWidth::compute(self.graph.borrow());
        {
            let lb: &Cell<_> = self.lower_bound.borrow();
            if mmw > lb.get() {
                println!("c Found new Lowerbound. Previous {} Now {}", lb.get(), mmw);
                lb.set(mmw);
            }
            println!(
                "c Atom has upperbound {}. Global lowerbound is {}",
                upperbound.max_bag_size - 1,
                lb.get()
            );
        }
        let lowerbound = {
            let tmp: &Cell<_> = self.lower_bound.borrow();
            tmp.get()
        };
        let mut solver =
            PID::with_bounds(self.graph.borrow(), lowerbound, upperbound.max_bag_size - 1);
        match solver.compute_exact() {
            Ok(td) => td,
            Err(_) => upperbound,
        }
    }

    pub fn find_separator(&self) -> Option<FnvHashSet<usize>> {
        match self.separation_level {
            SeparationLevel::Connected => self.graph.find_cut_vertex(),
            SeparationLevel::BiConnected => self.graph.find_safe_bi_connected_separator(),
            SeparationLevel::TriConnected => if self.graph.order() < 250 { self.graph.find_safe_tri_connected_separator() } else { None },
            SeparationLevel::Clique => self.graph.find_clique_minimal_separator(),
            SeparationLevel::AlmostClique => if self.graph.order() < 250 { self.graph.find_almost_clique_minimal_separator() } else { None },
            SeparationLevel::MinorSafeClique => self.graph.find_minor_safe_separator(),
            SeparationLevel::Atomic => None,
        }
    }
}

pub struct SafeSeparatorFramework {
    graph: HashMapGraph,
    lower_bound: Rc<Cell<usize>>,
    log_state: Rc<Cell<DecompositionInformation>>,
    search_state: SearchState,
}

impl SafeSeparatorFramework {
    pub fn new(graph: HashMapGraph, lower_bound: usize) -> Self {
        let lb = Rc::new(Cell::new(lower_bound));
        let log_state = Rc::new(Cell::new(DecompositionInformation::default()));
        Self {
            graph: graph.clone(),
            lower_bound: lb.clone(),
            log_state: log_state.clone(),
            search_state: SearchState {
                graph,
                separation_level: SeparationLevel::Connected,
                lower_bound: lb,
                log_state,
            },
        }
    }

    pub fn compute(mut self) -> DecompositionResult {
        let mut td = self.search_state.search();
        let mut lb = { self.lower_bound.get() };
        let mut log_state = { self.log_state.get() };

        DecompositionResult {
            tree_decomposition: td,
            decomposition_information: self.log_state.get(),
        }
    }
}

#[derive(Debug, Clone)]
struct AtomState {
    graph: HashMapGraph,
    target_bag: usize,
    tree_decomposition: Option<TreeDecomposition>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DecompositionInformation {
    n_separators: usize,
    n_cut_vertices: usize,
    n_degree_two_separators: usize,
    n_degree_three_separators: usize,
    n_clique_separators: usize,
    n_almost_clique_separators: usize,
    n_minor_safe_separators: usize,
    n_atoms: usize,
    max_atom: usize,
}

impl DecompositionInformation {
    fn increment(&mut self, separation_level: SeparationLevel) {
        match separation_level {
            SeparationLevel::Atomic => {}
            _ => {
                self.n_separators += 1;
            }
        }
        match separation_level {
            SeparationLevel::Connected => self.n_cut_vertices += 1,
            SeparationLevel::BiConnected => self.n_degree_two_separators += 1,
            SeparationLevel::TriConnected => self.n_degree_three_separators += 1,
            SeparationLevel::Clique => self.n_clique_separators += 1,
            SeparationLevel::AlmostClique => self.n_almost_clique_separators += 1,
            SeparationLevel::MinorSafeClique => self.n_minor_safe_separators += 1,
            SeparationLevel::Atomic => self.n_atoms += 1,
        }
    }

    fn set_max_atom(&mut self, max_atom: usize) {
        self.max_atom = max_atom;
    }
}

pub struct DecompositionResult {
    pub tree_decomposition: TreeDecomposition,
    pub decomposition_information: DecompositionInformation,
}
