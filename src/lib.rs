#[macro_use]
pub(crate) mod macros {
    macro_rules! impl_setter {
        ($self:ident, $field:ident, $type:ty) => {
            pub fn $field(mut $self, $field: $type) -> Self {
                $self.$field = $field;
                $self
            }
        }
    }
}

pub(crate) mod datastructures;

pub mod exact;
pub mod graph;
pub mod heuristic_elimination_order;
pub mod io;
pub mod lowerbound;
pub mod meta_heuristics;
pub mod solver;
pub mod tree_decomposition;

mod rule_based_reducer;
mod safe_separator_framework;
pub use rule_based_reducer::RuleBasedPreprocessor;
pub use safe_separator_framework::{SafeSeparatorFramework, SafeSeparatorLimits};

#[cfg(feature = "handle-ctrlc")]
pub mod signals;

#[cfg(feature = "log")]
#[cfg(feature = "env_logger")]
pub mod log;
