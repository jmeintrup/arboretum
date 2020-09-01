use crate::datastructures::BitSet;
use crate::graph::bag::Bag;
use crate::graph::bag::TreeDecomposition;
use crate::graph::bit_graph::*;
use crate::graph::graph::Graph;
use fnv::{FnvHashMap, FnvHashSet};
use std::any::Any;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::collections::hash_set::Iter;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::iter::Filter;

type BlockCache = FnvHashMap<BitSet, Block>;
type OBlockCache = FnvHashMap<BitSet, OBlock>;
type IBlockCache = FnvHashMap<BitSet, IBlock>;

#[derive(Debug)]
struct Cache {
    pub o_block_cache: OBlockCache,
    pub i_block_cache: IBlockCache,
    pub block_cache: BlockCache,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HeapBitset {
    bit_set: BitSet,
}

impl Ord for HeapBitset {
    fn cmp(&self, other: &Self) -> Ordering {
        other.bit_set.cardinality().cmp(&self.bit_set.cardinality())
    }
}

impl PartialOrd for HeapBitset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.cmp(self))
    }
}

impl From<BitSet> for HeapBitset {
    fn from(bit_set: BitSet) -> Self {
        Self { bit_set }
    }
}

impl From<HeapBitset> for BitSet {
    fn from(bit_set: HeapBitset) -> Self {
        bit_set.bit_set
    }
}

pub struct PID<'a, G: Graph> {
    og_graph: &'a G,
    og_to_self: FnvHashMap<u32, u32>,
    self_to_og: Vec<u32>,
    graph: BitGraph,
    target_width: u32,
    upper_bound: u32,

    ready_queue: VecDeque<IBlock>,
    pending_endorsers: Vec<PMC>,
    i_blocks: HashSet<BitSet>,
    o_blocks: HashSet<BitSet>,
    buildable: HashSet<BitSet>,
    feasible: HashSet<BitSet>,
    solution: Option<PMC>,
    cache: Cache,
}

impl<'a, G: Graph> PID<'a, G> {
    pub fn new(og_graph: &'a G) -> Self {
        let mut og_to_self = FnvHashMap::default();
        let mut self_to_og = Vec::with_capacity(og_graph.order());

        for (idx, v) in og_graph.vertices().enumerate() {
            og_to_self.insert(v as u32, idx as u32);
            self_to_og.push(v as u32);
        }

        let graph = BitGraph::from_graph(og_graph, &og_to_self);
        Self {
            og_graph,
            og_to_self,
            self_to_og,
            graph,
            target_width: 0,
            upper_bound: (og_graph.order()) as u32,
            ready_queue: Default::default(),
            pending_endorsers: Default::default(),
            i_blocks: Default::default(),
            o_blocks: Default::default(),
            buildable: Default::default(),
            feasible: Default::default(),
            solution: None,
            cache: Cache {
                o_block_cache: OBlockCache::default(),
                i_block_cache: IBlockCache::default(),
                block_cache: BlockCache::default(),
            },
        }
    }

    pub fn with_bounds(og_graph: &'a G, lowerbound: u32, upperbound: u32) -> Self {
        let mut og_to_self = FnvHashMap::default();
        let mut self_to_og = Vec::with_capacity(og_graph.order());

        for (idx, v) in og_graph.vertices().enumerate() {
            og_to_self.insert(v as u32, idx as u32);
            self_to_og.push(v as u32);
        }

        let graph = BitGraph::from_graph(og_graph, &og_to_self);
        Self {
            og_graph,
            og_to_self,
            self_to_og,
            graph,
            target_width: lowerbound,
            upper_bound: upperbound,
            ready_queue: Default::default(),
            pending_endorsers: Default::default(),
            i_blocks: Default::default(),
            o_blocks: Default::default(),
            buildable: Default::default(),
            feasible: Default::default(),
            solution: None,
            cache: Cache {
                o_block_cache: OBlockCache::default(),
                i_block_cache: IBlockCache::default(),
                block_cache: BlockCache::default(),
            },
        }
    }

    pub fn decompose(&mut self) -> Result<TreeDecomposition, u32> {
        println!("c order: {}", self.graph.order());
        if self.graph.order() <= 2 {
            let mut td = TreeDecomposition::new();
            if self.graph.order() > 0 {
                let vertex_set: FnvHashSet<usize> = (0..self.graph.order())
                    .map(|i| self.self_to_og[i as usize] as usize)
                    .collect();
                td.add_bag(vertex_set);
            }
            return Ok(td);
        }
        let mut o_block_sieve = LayeredSieve::new(self.graph.order() as u32, self.target_width);

        while self.target_width < self.upper_bound {
            self.cache.o_block_cache = OBlockCache::default();
            o_block_sieve = LayeredSieve::new(self.graph.order() as u32, self.target_width);
            self.ready_queue = VecDeque::with_capacity(self.cache.i_block_cache.len());

            self.ready_queue = self.cache.i_block_cache.values().cloned().collect();

            for v in 0..self.graph.order() {
                let mut closed_neighborhood = self.graph.neighborhood_as_bitset(v).clone();
                closed_neighborhood.set_bit(v);

                if closed_neighborhood.cardinality() > (self.target_width + 1) as usize {
                    continue;
                }
                let mut blocks = separate_into_blocks(
                    &closed_neighborhood,
                    self.graph.borrow(),
                    self.cache.block_cache.borrow_mut(),
                    self.cache.i_block_cache.borrow_mut(),
                );
                let mut pmc = PMC::new(closed_neighborhood, &blocks, self.graph.borrow());

                if pmc.valid {
                    if pmc.ready(&mut self.cache.i_block_cache) {
                        pmc.endorse(
                            &self.graph,
                            &mut self.cache,
                            &mut self.ready_queue,
                            &mut self.solution,
                        );
                    } else {
                        self.pending_endorsers.push(pmc);
                    }
                }
            }

            loop {
                while !self.ready_queue.is_empty() {
                    let ready = self.ready_queue.pop_front().unwrap();
                    ready.process(
                        &mut o_block_sieve,
                        self.target_width as usize,
                        &self.graph,
                        &mut self.cache,
                        &mut self.ready_queue,
                        &mut self.pending_endorsers,
                        &mut self.solution,
                    );

                    if self.solution.is_some() {
                        return Ok(self.create_tree_decomposition());
                    }
                }

                let endorsers = std::mem::replace(&mut self.pending_endorsers, Default::default());
                for endorser in endorsers {
                    if endorser.ready(&self.cache.i_block_cache) {
                        endorser.endorse(
                            &self.graph,
                            &mut self.cache,
                            &mut self.ready_queue,
                            &mut self.solution,
                        );
                    } else {
                        self.pending_endorsers.push(endorser);
                    }
                    if self.solution.is_some() {
                        return Ok(self.create_tree_decomposition());
                    }
                }
                if self.ready_queue.is_empty() {
                    break;
                }
            }
            self.target_width += 1;
        }
        println!(
            "Err: target_width:{}, lower_bound:{}",
            self.target_width, self.upper_bound
        );
        Err(self.target_width)
        /*let mut td = TreeDecomposition::new();
        let bag = (0..self.graph.order()).map(|i| { self.self_to_og[i] as usize }).collect();
        td.add_bag(bag);
        return Ok(td);*/
    }

    fn create_tree_decomposition(&self) -> TreeDecomposition {
        let pmc = self.solution.as_ref().unwrap().clone();
        let mut td = TreeDecomposition::new();
        let vertex_set = self.translate_vertex_set(&pmc.vertex_set);
        let parent = td.add_bag(vertex_set);
        self.td_rec(parent, &mut td, &pmc);
        td
    }

    fn translate_vertex_set(&self, vertex_set: &BitSet) -> FnvHashSet<usize> {
        vertex_set
            .iter()
            .map(|v| self.self_to_og[v] as usize)
            .collect()
    }

    fn td_rec(&self, parent: usize, td: &mut TreeDecomposition, pmc: &PMC) -> () {
        let children = pmc.inbounds.clone();
        for c in children
            .iter()
            .map(|c| self.cache.i_block_cache.get(&c.component))
            .filter(|ib| ib.is_some())
            .map(|ib| &ib.unwrap().endorser)
        {
            let vertex_set = self.translate_vertex_set(&c.vertex_set);
            let new_parent = td.add_bag(vertex_set);
            td.add_edge(parent, new_parent);
            self.td_rec(new_parent, td, c);
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Block {
    outbound: Option<BitSet>,
    component: BitSet,
    separator: BitSet,
}

impl Ord for Block {
    fn cmp(&self, other: &Self) -> Ordering {
        self.component.cmp(&other.component)
    }
}

impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.component.cmp(&other.component))
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = if self.outbound.is_some() && self.outbound.as_ref().unwrap() == &self.component {
            "o"
        } else {
            "f/i"
        };
        let c: Vec<_> = self.component.iter().map(|i| i.to_string()).collect();
        let s: Vec<_> = self.separator.iter().map(|i| i.to_string()).collect();
        write!(f, "{}{{{}}}({{{}}})", t, c.join(", "), s.join(", "),)
    }
}

// todo: check if cmp is needed, cmp via firstsetbit idx
impl Block {
    pub fn log(&self, cache: &IBlockCache) -> String {
        let t = if self.outbound.is_some() && self.outbound.as_ref().unwrap() == &self.component {
            "o"
        } else {
            if cache.get(&self.component).is_some() {
                "f"
            } else {
                "i"
            }
        };
        let c: Vec<_> = self.component.iter().map(|i| i.to_string()).collect();
        let s: Vec<_> = self.separator.iter().map(|i| i.to_string()).collect();
        let o = if self.outbound.is_some() {
            let o: Vec<_> = self
                .outbound
                .as_ref()
                .unwrap()
                .iter()
                .map(|i| i.to_string())
                .collect();
            format!("o:({{{}}})", o.join(", "))
        } else {
            String::from("null")
        };

        format!("{}{{{}}}({{{}}}){}", t, c.join(", "), s.join(", "), o,)
    }

    pub fn new(component: BitSet, graph: &BitGraph, i_block_cache: &IBlockCache) -> Self {
        let separator = graph.exterior_border(&component);
        let mut rest = BitSet::new_all_set(graph.order());
        rest.and_not(&component);
        rest.and_not(&separator);

        let min_compo = component.get_first_set();
        let min_compo = min_compo.unwrap();

        let mut v = rest.get_first_set();
        let mut outbound = None;
        while v.is_some() {
            let v_value = v.unwrap();
            let mut c = graph.neighborhood_as_bitset(v_value).clone();
            let mut to_be_scanned = c.clone();
            to_be_scanned.and_not(&separator);
            c.set_bit(v_value);

            while !to_be_scanned.empty() {
                let mut save = c.clone();
                for w in to_be_scanned.iter() {
                    c.or(graph.neighborhood_as_bitset(w));
                }
                to_be_scanned = c.clone();
                to_be_scanned.and_not(&save);
                to_be_scanned.and_not(&separator);
            }
            if separator.is_subset_of(&c) {
                if v_value < min_compo {
                    let mut o = c.clone();
                    o.and_not(&separator);
                    outbound = Some(o);
                } else {
                    outbound = Some(component.clone());
                };
                break;
            }
            rest.and_not(&c);
            v = rest.get_next_set(v_value + 1);
        }
        let r = Self {
            component,
            separator,
            outbound,
        };
        r
    }

    pub fn outbound(&self) -> bool {
        self.outbound.is_some() && self.outbound.as_ref().unwrap() == &self.component
    }

    pub fn of_minimal_separator(&self) -> bool {
        self.outbound.is_some()
    }
}

#[derive(Clone, Eq, PartialEq)]
struct PMC {
    vertex_set: BitSet,
    inbounds: Vec<Block>,
    outbound: Option<Block>,
    valid: bool,
}

impl Debug for PMC {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let validity = if self.valid {
            "(valid):\n"
        } else {
            "(invalid):\n"
        };
        let s: Vec<_> = self.vertex_set.iter().map(|i| i.to_string()).collect();
        let o = if self.outbound.is_some() {
            format!("{:?}", self.outbound.as_ref().unwrap())
        } else {
            String::from("null")
        };
        let i: Vec<_> = self
            .inbounds
            .iter()
            .map(|b| format!("  inbound : {:?}\n", b))
            .collect();
        write!(
            f,
            "PMC{}  sep     : {{{}}}\n  outbound: {}\n{}",
            validity,
            s.join(", "),
            o,
            i.join(""),
        )
    }
}

impl PMC {
    pub fn new(vertex_set: BitSet, blocks: &[Block], graph: &BitGraph) -> Self {
        if vertex_set.empty() {
            return Self {
                vertex_set,
                inbounds: vec![],
                outbound: None,
                valid: false,
            };
        }
        let mut outbound: Option<Block> = None;
        let mut inbounds: Option<Vec<Block>> = None;
        for block in blocks {
            if block.outbound()
                && (outbound.is_none()
                    || outbound
                        .as_ref()
                        .unwrap()
                        .separator
                        .is_subset_of(&block.separator))
            {
                outbound = Some(block.clone());
            }
        }
        if outbound.is_none() {
            inbounds = Some(Vec::from(blocks));
        } else {
            let mut tmp: Vec<Block> = blocks
                .iter()
                .filter(|block| {
                    !block
                        .separator
                        .is_subset_of(&outbound.as_ref().unwrap().borrow().separator)
                })
                .cloned()
                .collect();
            inbounds = Some(tmp);
        }
        let mut pmc = Self {
            vertex_set,
            inbounds: inbounds.unwrap(),
            outbound,
            valid: true,
        };
        for b in &pmc.inbounds {
            if !b.of_minimal_separator() {
                pmc.valid = false;
                return pmc;
            }
        }

        for v in pmc.vertex_set.iter() {
            let mut rest = pmc.vertex_set.clone();
            rest.and_not(graph.neighborhood_as_bitset(v));
            rest.unset_bit(v);

            if pmc.outbound.is_some() && pmc.outbound.as_ref().unwrap().separator[v] {
                rest.and_not(&pmc.outbound.as_ref().unwrap().separator);
            }
            for b in pmc.inbounds.iter().filter(|b| b.separator[v]) {
                rest.and_not(&b.separator)
            }
            if !rest.empty() {
                pmc.valid = false;
                return pmc;
            }
        }
        pmc
    }

    fn ready(&self, i_block_cache: &IBlockCache) -> bool {
        for ib in &self.inbounds {
            if i_block_cache.get(&ib.component).is_none() {
                return false;
            }
        }
        true
    }

    pub fn endorse(
        &self,
        graph: &BitGraph,
        cache: &mut Cache,
        ready_queue: &mut VecDeque<IBlock>,
        solution: &mut Option<PMC>,
    ) {
        if self.outbound.is_some() {
            let mut target = self.vertex_set.clone();
            target.and_not(&self.outbound.as_ref().unwrap().separator);
            for b in &self.inbounds {
                target.or(&b.component);
            }
            if cache.i_block_cache.get(&target).is_none() {
                let block = get_or_create_block(
                    &target,
                    graph,
                    &mut cache.block_cache,
                    &cache.i_block_cache,
                );
                let i_block = IBlock::new(block.clone(), self.clone());
                cache.i_block_cache.insert(target, i_block.clone());
                ready_queue.push_back(i_block);
            }
        } else {
            *solution = Some(self.clone());
        }
    }

    pub fn process(
        &self,
        graph: &BitGraph,
        cache: &mut Cache,
        queue: &mut VecDeque<IBlock>,
        pending: &mut Vec<PMC>,
        solution: &mut Option<PMC>,
    ) {
        if self.ready(&cache.i_block_cache) {
            self.endorse(graph, cache, queue, solution);
        } else {
            pending.push(self.clone());
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
struct IBlock {
    block: Block,
    endorser: PMC,
}

impl Ord for IBlock {
    fn cmp(&self, other: &Self) -> Ordering {
        self.block.cmp(&other.block)
    }
}

impl PartialOrd for IBlock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Debug for IBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s: Vec<_> = self.block.separator.iter().map(|i| i.to_string()).collect();
        let c: Vec<_> = self.block.component.iter().map(|i| i.to_string()).collect();
        let o = if self.block.outbound.is_some() {
            let tmp: Vec<_> = self
                .block
                .outbound
                .as_ref()
                .unwrap()
                .iter()
                .map(|i| i.to_string())
                .collect();
            tmp.join(", ")
        } else {
            String::from("null")
        };
        write!(
            f,
            "IBlock:{{{}}}\n    in  :{{{}}}\n    out :{{{}}}\n",
            s.join(", "),
            c.join(", "),
            o
        )
    }
}

impl IBlock {
    pub fn new(block: Block, endorser: PMC) -> Self {
        Self { block, endorser }
    }

    pub fn process(
        &self,
        o_block_sieve: &mut LayeredSieve,
        target_width: usize,
        graph: &BitGraph,
        cache: &mut Cache,
        ready_queue: &mut VecDeque<IBlock>,
        pending_endorsers: &mut Vec<PMC>,
        solution: &mut Option<PMC>,
    ) {
        let o_block = cache.o_block_cache.get(&self.block.separator);
        if o_block.is_none() {
            let o_block = OBlock::new(
                self.block.separator.clone(),
                self.block.outbound.as_ref().unwrap().clone(),
            );
            cache
                .o_block_cache
                .insert(self.block.separator.clone(), o_block.clone());
            o_block_sieve.insert(
                self.block.outbound.as_ref().unwrap(),
                self.block.separator.clone(),
            );
            o_block.process(
                graph,
                target_width,
                cache,
                ready_queue,
                pending_endorsers,
                solution,
            );
        }

        let o_block_separators: Vec<&BitSet> =
            o_block_sieve.super_blocks(&self.block.component, &self.block.separator);
        let mut to_add: Vec<(BitSet, BitSet)> = Vec::new();
        for sep in o_block_separators {
            let tmp = std::mem::replace(
                cache.o_block_cache.get_mut(sep).unwrap(),
                OBlock {
                    separator: Default::default(),
                    open_component: Default::default(),
                },
            );
            match tmp.combine(
                self,
                target_width,
                graph,
                cache,
                ready_queue,
                pending_endorsers,
                solution,
            ) {
                Some(v) => {
                    to_add.push(v);
                }
                _ => {}
            };
            let o_block = cache.o_block_cache.get_mut(sep).unwrap();
            *o_block = tmp;
        }
        to_add.drain(..).for_each(|(k, v)| {
            o_block_sieve.insert(&k, v);
        })
    }
}

#[derive(Clone)]
struct OBlock {
    separator: BitSet,
    open_component: BitSet,
}

impl Debug for OBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s: Vec<_> = self.separator.iter().map(|i| i.to_string()).collect();
        let o: Vec<_> = self.open_component.iter().map(|i| i.to_string()).collect();
        write!(
            f,
            "TBlock:\n  sep :{{{}}}\n  open:{{{}}}",
            s.join(", "),
            o.join(", "),
        )
    }
}

impl OBlock {
    pub fn new(separator: BitSet, open_component: BitSet) -> Self {
        let tmp = Self {
            separator,
            open_component,
        };
        tmp
    }

    pub fn process(
        &self,
        graph: &BitGraph,
        target_width: usize,
        cache: &mut Cache,
        ready_queue: &mut VecDeque<IBlock>,
        pending_endorsers: &mut Vec<PMC>,
        solution: &mut Option<PMC>,
    ) {
        for v in self.separator.iter() {
            let mut new_separator = self.separator.clone();
            let mut nb = graph.neighborhood_as_bitset(v).clone();
            nb.and(&self.open_component);
            new_separator.or(&nb);

            if new_separator.cardinality() <= target_width + 1 {
                let blocks = separate_into_blocks(
                    &new_separator,
                    graph,
                    &mut cache.block_cache,
                    &mut cache.i_block_cache,
                );
                let mut pmc = PMC::new(new_separator, &blocks, graph);

                if pmc.valid {
                    if pmc.ready(&cache.i_block_cache) {
                        pmc.endorse(graph, cache, ready_queue, solution);
                    } else {
                        pending_endorsers.push(pmc);
                    }
                }
            }
        }
    }

    pub fn combine(
        &self,
        i_block: &IBlock,
        target_width: usize,
        graph: &BitGraph,
        cache: &mut Cache,
        ready_queue: &mut VecDeque<IBlock>,
        pending_endorsers: &mut Vec<PMC>,
        solution: &mut Option<PMC>,
    ) -> Option<(BitSet, BitSet)> {
        let mut new_separator = self.separator.clone();
        new_separator.or(i_block.block.separator.borrow());
        if new_separator.cardinality() > target_width + 1 {
            return None;
        }

        let blocks = separate_into_blocks(
            &new_separator,
            graph,
            &mut cache.block_cache,
            &mut cache.i_block_cache,
        );

        let mut full_block: Option<&Block> = None;
        for block in &blocks {
            if block.separator.cardinality() == new_separator.cardinality() {
                if full_block.is_some() {
                    return None;
                }
                full_block = Some(block);
            }
        }

        if full_block.is_none() {
            let pmc = PMC::new(new_separator, &blocks, graph);
            if pmc.valid {
                if pmc.ready(&cache.i_block_cache) {
                    pmc.endorse(graph, cache, ready_queue, solution);
                } else {
                    pending_endorsers.push(pmc);
                }
            }
        } else {
            if new_separator.cardinality() > target_width {
                return None;
            }
            let o_block = cache.o_block_cache.get(&new_separator);
            if o_block.is_none() {
                let o_block = OBlock::new(
                    new_separator.clone(),
                    full_block.as_ref().unwrap().component.clone(),
                );
                cache
                    .o_block_cache
                    .insert(new_separator.clone(), o_block.clone());
                o_block.process(
                    graph,
                    target_width,
                    cache,
                    ready_queue,
                    pending_endorsers,
                    solution,
                );
                return Some((
                    full_block.as_ref().unwrap().component.clone(),
                    new_separator,
                ));
            }
        }
        None
    }
}

fn separate_into_blocks(
    separator: &BitSet,
    graph: &BitGraph,
    block_cache: &mut BlockCache,
    i_block_cache: &mut IBlockCache,
) -> Vec<Block> {
    let mut rest = separator.clone();
    rest.not();

    let mut blocks = Vec::new();
    let mut v_option = rest.get_first_set();
    while v_option.is_some() {
        let v = v_option.unwrap();
        let mut c = graph.neighborhood_as_bitset(v).clone();
        c.and_not(separator);
        let mut to_be_scanned = c.clone();
        c.set_bit(v);

        while !to_be_scanned.empty() {
            let save = c.clone();
            for w in to_be_scanned.iter() {
                c.or(graph.neighborhood_as_bitset(w));
            }
            c.and_not(separator);
            to_be_scanned = c.clone();
            to_be_scanned.and_not(&save);
        }
        let block = get_or_create_block(&c, graph, block_cache, i_block_cache).clone();
        blocks.push(block);

        rest.and_not(&c);
        v_option = rest.get_next_set(v + 1);
    }
    return blocks;
}

fn get_or_create_block<'a>(
    component: &BitSet,
    graph: &BitGraph,
    block_cache: &'a mut BlockCache,
    i_block_cache: &'a IBlockCache,
) -> &'a Block {
    let is_none = block_cache.get(&component).is_none();
    if is_none {
        block_cache.insert(
            component.clone(),
            Block::new(component.clone(), graph, i_block_cache),
        );
    }
    block_cache.get(&component).unwrap()
}

struct LayeredSieve {
    n: u32,
    target_width: u32,
    sieves: Vec<BlockSieve>,
}

impl LayeredSieve {
    pub fn new(n: u32, target_width: u32) -> Self {
        let k = 33 - target_width.leading_zeros();
        let mut sieves = Vec::with_capacity(k as usize);
        for i in 0..k {
            let margin = (1 << i) - 1;
            sieves.push(BlockSieve::new(n, target_width, margin));
        }
        Self {
            n,
            target_width,
            sieves,
        }
    }

    pub fn insert(&mut self, vertices: &BitSet, neighbors: BitSet) {
        /*let v: Vec<_> = vertices.iter().map(|i| i.to_string()).collect();
        let n: Vec<_> = neighbors.iter().map(|i| i.to_string()).collect();
        println!("vertices{{{}}}\nneighbors{{{}}}",v.join(", "),n.join(", "));
        let v: Vec<_> = vertices.as_bit_vec().as_slice().iter().map(|i| *i as i64).map(|i| i.to_string()).collect();
        let n: Vec<_> = neighbors.as_bit_vec().as_slice().iter().map(|i| *i as i64).map(|i| i.to_string()).collect();
        println!("vertices_long[{}]\nneighbors_long[{}]",v.join(", "),n.join(", "));*/
        let ns: u32 = neighbors.cardinality() as u32;
        let margin: u32 = self.target_width + 1 - ns;
        let i = 32 - margin.leading_zeros();
        //println!("i: {}", i);
        self.sieves[i as usize].put(vertices, neighbors);
    }

    pub fn super_blocks<'a>(&'a self, vertices: &BitSet, neighbors: &BitSet) -> Vec<&'a BitSet> {
        let mut collector = Vec::new();
        for s in &self.sieves {
            s.super_blocks(vertices, neighbors, &mut collector);
        }
        collector
    }

    pub fn len(&self) -> usize {
        self.sieves.iter().map(|i| i.len()).sum()
    }
}

struct BlockSieve {
    root: Box<dyn BlockSieveNode>,
    last: u32,
    target_width: u32,
    margin: u32,
    size: u32,
    n: u32,
}

impl BlockSieve {
    pub const MAX_CHILDREN_SIZE: usize = 512;

    pub fn new(n: u32, target_width: u32, margin: u32) -> Self {
        Self {
            root: Box::new(NodeU64::new(0, 64, 0)),
            last: (n - 1) / 64,
            target_width,
            margin,
            size: 0,
            n,
        }
    }

    pub fn put(&mut self, key: &BitSet, value: BitSet) {
        let slice = key.as_slice();
        let mut node = &mut self.root;

        let mut i = 0;
        let mut j1 = 0;
        let mut bits: u64 = 0;
        loop {
            bits = 0;
            if i < slice.len() {
                bits = slice[i] as u64;
            }
            let mut j = node.index_of(bits);
            if j.is_err() {
                break;
            }
            let j = j.ok().unwrap();
            if node.is_leaf(self.last) {
                return;
            }
            node = &mut node.node_mut().children[j];
            i = node.node().index as usize;
            j1 = j;
        }
        if node.is_leaf(self.last) {
            node.add_value(bits, value);
        } else if node.is_last_in_interval() {
            node.add_child(bits, Self::new_path(i + 1, slice, value, self.last));
        } else {
            let ntz = node.node().ntz;
            let width = node.node().width;
            let mut header = new_node(i as u32, 64 - (ntz + width), ntz + width);
            if !header.is_leaf(self.last) {
                header.add_child(bits, Self::new_path(i + 1, slice, value, self.last));
            } else {
                header.add_value(bits, value);
            }
            node.add_child(bits, header);
        }
        self.size += 1;
        Self::try_resize(node, self.last);
    }

    fn new_path(idx: usize, slice: &[usize], value: BitSet, last: u32) -> Box<dyn BlockSieveNode> {
        let mut node = NodeU64::new(idx as u32, 64, 0);
        let mut bits = 0;
        if idx < slice.len() {
            bits = slice[idx] as u64;
        }
        if idx == last as usize {
            node.add_value(bits, value);
        } else {
            let path = Self::new_path(idx + 1, slice, value, last);
            node.add_child(bits, path);
        }

        Box::new(node)
    }

    fn try_resize(node: &mut Box<dyn BlockSieveNode>, last: u32) {
        if node.size() > Self::MAX_CHILDREN_SIZE {
            Self::resize(node, last);
            for child in node.node_mut().children.iter_mut() {
                Self::try_resize(child, last);
            }
        }
    }

    fn resize(node: &mut Box<dyn BlockSieveNode>, last: u32) {
        let w = node.node().width;
        let sz = node.size();

        let mut values = vec![0u64; sz];
        let mut m = node.get_mask();
        let ntz = m.trailing_zeros();
        let mut t = ntz + node.node().width;

        while values.len() > Self::MAX_CHILDREN_SIZE {
            t = (ntz + t) / 2;
            m = consecutive_one_bit(ntz, t);
            let mut p = 0;
            for i in 0..sz {
                let label = (node.get_label_at(i) & m) >> ntz;
                match values[0..p].binary_search(&label) {
                    Err(j) => {
                        let mut k = p as i64;
                        while k - 1 >= j as i64 {
                            values[k as usize] = values[(k - 1) as usize];
                            k -= 1;
                        }
                        values[j as usize] = label;
                        p += 1;
                    }
                    _ => {}
                }
            }
            values.resize(p, 0);
        }

        let mut new_children = Vec::with_capacity(values.len());
        let mask = node.get_mask() & !m;
        for _ in 0..values.len() {
            new_children.push(new_node(
                node.node().index,
                mask.count_ones(),
                mask.trailing_zeros(),
            ));
        }

        for i in 0..values.len() {
            let label = node.get_label_at(i);
            let j = values[..].binary_search(&((label & m) >> ntz)).unwrap();
            if node.is_leaf(last) {
                let tmp = node.node_mut().values.get_mut(i).unwrap();
                let value = std::mem::replace(tmp, Default::default());
                new_children[j].add_value(label, value);
            } else {
                let tmp = node.node_mut().children.get_mut(i).unwrap();
                let child = std::mem::replace(tmp, Box::new(PlaceHolderNode::default()));
                new_children[j].add_child(label, child);
            }
        }

        *node = new_node(node.node().index, m.count_ones(), m.trailing_zeros());
        for (i, c) in new_children.drain(..).enumerate() {
            node.add_child(values[i] << ntz, c);
        }
    }

    pub fn super_blocks<'a>(
        &'a self,
        vertices: &BitSet,
        neighbors: &BitSet,
        collector: &mut Vec<&'a BitSet>,
    ) {
        self.root.filter_superblocks(
            self,
            vertices.as_slice(),
            neighbors.as_slice(),
            0,
            collector,
        );
    }

    pub fn len(&self) -> usize {
        self.size as usize
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct PlaceHolderNode {}

impl BlockSieveNode for PlaceHolderNode {
    fn get_mask(&self) -> u64 {
        unimplemented!()
    }

    fn is_last_in_interval(&self) -> bool {
        unimplemented!()
    }

    fn is_leaf(&self, last: u32) -> bool {
        unimplemented!()
    }

    fn size(&self) -> usize {
        unimplemented!()
    }

    fn get_label_at(&self, i: usize) -> u64 {
        unimplemented!()
    }

    fn index_of(&self, label: u64) -> Result<usize, usize> {
        unimplemented!()
    }

    fn filter_superblocks(
        &self,
        owner: &BlockSieve,
        slice: &[usize],
        neighbors: &[usize],
        intersects: u32,
        collector: &mut Vec<&BitSet>,
    ) {
        unimplemented!()
    }

    fn node(&self) -> &BaseNode {
        unimplemented!()
    }

    fn node_mut(&mut self) -> &mut BaseNode {
        unimplemented!()
    }

    fn add_child(&mut self, label: u64, child: Box<dyn BlockSieveNode>) -> usize {
        unimplemented!()
    }

    fn add_value(&mut self, label: u64, value: BitSet) -> usize {
        unimplemented!()
    }

    fn add_label(&mut self, label: u64) -> usize {
        unimplemented!()
    }
}

struct BaseNode {
    index: u32,
    width: u32,
    ntz: u32,
    children: Vec<Box<dyn BlockSieveNode>>,
    values: Vec<BitSet>,
}

const fn consecutive_one_bit(i: u32, j: u32) -> u64 {
    (u64::MAX >> (64 - j)) & (u64::MAX << i)
}

trait BlockSieveNode {
    fn get_mask(&self) -> u64;

    fn is_last_in_interval(&self) -> bool;

    fn is_leaf(&self, last: u32) -> bool;

    fn size(&self) -> usize;

    fn get_label_at(&self, i: usize) -> u64;

    fn index_of(&self, label: u64) -> Result<usize, usize>;
    fn filter_superblocks<'a>(
        &'a self,
        owner: &BlockSieve,
        slice: &[usize],
        neighbors: &[usize],
        intersects: u32,
        collector: &mut Vec<&'a BitSet>,
    );

    fn node(&self) -> &BaseNode;
    fn node_mut(&mut self) -> &mut BaseNode;
    fn add_child(&mut self, label: u64, child: Box<dyn BlockSieveNode>) -> usize;
    fn add_value(&mut self, label: u64, value: BitSet) -> usize;
    fn add_label(&mut self, label: u64) -> usize;
}

fn new_node(index: u32, width: u32, ntz: u32) -> Box<dyn BlockSieveNode> {
    //return Box::new(LongNode::new(index, width, ntz));
    if width > 32 {
        return Box::new(NodeU64::new(index, width, ntz));
    } else if width > 16 {
        return Box::new(NodeU32::new(index, width, ntz));
    } else if width > 8 {
        return Box::new(NodeU16::new(index, width, ntz));
    }
    Box::new(NodeU8::new(index, width, ntz))
}

macro_rules! impl_node {
    ($struct_name:ident;$label_t:ty) => {
        struct $struct_name {
            node: BaseNode,
            labels: Vec<$label_t>,
        }

        impl $struct_name {
            pub fn new(index: u32, width: u32, ntz: u32) -> Self {
                //println!("index{}width{}ntz{}", index, width, ntz);
                Self {
                    node: BaseNode {
                        index,
                        width,
                        ntz,
                        children: Vec::with_capacity(BlockSieve::MAX_CHILDREN_SIZE),
                        values: Vec::with_capacity(BlockSieve::MAX_CHILDREN_SIZE),
                    },
                    labels: Vec::with_capacity(BlockSieve::MAX_CHILDREN_SIZE),
                }
            }
        }

        impl BlockSieveNode for $struct_name {
            fn get_mask(&self) -> u64 {
                consecutive_one_bit(self.node.ntz, self.node.ntz + self.node.width)
            }

            fn is_last_in_interval(&self) -> bool {
                self.node.ntz + self.node.width == 64
            }

            fn is_leaf(&self, last: u32) -> bool {
                self.node.index == last && (self.node.ntz + self.node.width == 64)
            }

            fn size(&self) -> usize {
                self.labels.len()
            }

            fn get_label_at(&self, i: usize) -> u64 {
                (self.labels[i] as u64) << self.node.ntz
            }

            fn index_of(&self, label: u64) -> Result<usize, usize> {
                let mut key = ((label & self.get_mask()) >> self.node.ntz) as $label_t;
                self.labels.binary_search(&key)
            }

            fn filter_superblocks<'a>(
                &'a self,
                owner: &BlockSieve,
                longs: &[usize],
                neighbors: &[usize],
                intersects: u32,
                collector: &mut Vec<&'a BitSet>,
            ) {
                let mask = self.get_mask();
                let index = self.node.index as usize;
                let ntz = self.node.ntz;
                let mut bits: u64 = 0;
                if index < longs.len() {
                    bits = ((longs[index] as u64 & mask) >> ntz);
                }

                let mut neighbor: u64 = 0;
                if index < neighbors.len() {
                    neighbor = (neighbors[index] as u64 & mask) >> ntz
                }

                let is_leaf = self.is_leaf(owner.last);
                for (idx, label) in self
                    .labels
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(idx, label)| (idx, *label as u64))
                {
                    if bits > label {
                        break;
                    }
                    if bits & !label == 0 {
                        let intersects = intersects + (label & neighbor).count_ones();
                        if is_leaf {
                            if intersects + self.node.values[idx].cardinality() as u32
                                <= owner.target_width + 1
                            {
                                collector.push(&self.node.values[idx]);
                            }
                        } else {
                            if intersects <= owner.margin {
                                let n = &self.node.children[idx];
                                n.filter_superblocks(
                                    owner, longs, neighbors, intersects, collector,
                                );
                            }
                        }
                    }
                }
            }

            fn node(&self) -> &BaseNode {
                &self.node
            }

            fn node_mut(&mut self) -> &mut BaseNode {
                &mut self.node
            }

            fn add_child(&mut self, label: u64, child: Box<dyn BlockSieveNode>) -> usize {
                let i = self.add_label(label);
                self.node.children.insert(i as usize, child);
                i as usize
            }

            fn add_value(&mut self, label: u64, value: BitSet) -> usize {
                let i = self.add_label(label);
                self.node.values.insert(i as usize, value);
                i as usize
            }

            fn add_label(&mut self, label: u64) -> usize {
                let res = self.index_of(label);
                let i = res.err().unwrap();
                let val = (label & self.get_mask()) >> self.node.ntz;
                self.labels.insert(i as usize, val as $label_t);
                i as usize
            }
        }
    };
}

impl_node!(NodeU64;u64);
impl_node!(NodeU32;u32);
impl_node!(NodeU16;u16);
impl_node!(NodeU8;u8);
