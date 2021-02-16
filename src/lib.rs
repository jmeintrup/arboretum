#![allow(unused)]
#![allow(dead_code)]
pub(crate) mod datastructures;
pub(crate) use datastructures::{BinaryQueue, BitSet, BitSetIterator};

pub mod heuristic_elimination_order;
pub mod exact;
pub mod graph;
pub mod io;
pub mod lowerbound;
pub mod solver;
pub mod tree_decomposition;

mod rule_based_reducer;
mod safe_separator_framework;
pub use rule_based_reducer::RuleBasedPreprocessor;
pub use safe_separator_framework::SafeSeparatorFramework;
