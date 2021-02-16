<div style="text-align:center;">
	<img src="assets/logo_transparent.png" style="width: 240px;">
</div>

**arboretum** is a  graph library and CLI for computing tree decompositions.
Various state of the art preprocessing, graph reductions, exact and heuristic algorithms for obtaining tree decompositions are implemented.

*This repository is currently under development, the features and documentation presented here will be valid 
once the first version is released on [crates.io](https://crates.io/) The first release is planned for January 2021 and will include all the features mentioned here.*

# Features

* Well known fast heuristics such as min-degree and min-fill [[1]](#1)
* Metaheuristics such as tabu-local search and genetic algorithms [[3]](#3) and a novel algorithm based on the idea of artificial bee colony 
* Minor-min-width lowerbound heuristic [[4]](#4)
* Rule-based preprocessing for reducing graphs and obtaining tree-decompositions of graphs of treewidth ≤ 3 [[5]](#5)
* Graph decomposition based on the notion of safe separators [[6]](#6)
* State-of-the-art exact algorithms [[2]](#2)[[7]](#7)[[8]](#8)[[9]](#9)
* A classic branch-and-bound exact algorithm based on QuickBB [[10]](#10)

# The CLI
## Build
As **arboretum** is implemented in rust, the CLI can simply be built via cargo

```cargo build --release --features="cli"```

## Usage

Using a graph in [.gr format](https://pacechallenge.org/2021/) the program can be used as follows
```
cargo run --release --features="cli" < <graph.gr>
```
or
```
./target/release/arboretum < <graph.gr>
```
The CLI makes automated choices about which algorithms to use based on the input graph, but without the heuristic flag will always try to find an exact solution.

Available CLI arguments:

| Argument        | Description           |
| ------------- |:-------------|
| --help     | prints help |
| --verbose      | adds extensive logging and decomposition information      |
| --heuristic | sets the mode to heuristic     |
| --skip-preprocessing | skips running rule-based pre-processing    |
| --skip-safe-separator | skips running the safe-separator graph decomposition step |



```cargo build --release```

# The Library
## Usage

Simply add **arboretum** to your projects `cargo.toml` under dependencies and get started. For documentation refer to the [docs.rs](https://docs.rs/arboretum).

# References

<a id="1">[1]</a> 
Hans L. Bodlaender and Arie M. C. A. Koster. 2010. Treewidth computations I. Upper bounds. Inf. Comput. 208, 3 (March, 2010), 259–275. DOI:https://doi.org/10.1016/j.ic.2009.03.008

<a id="2">[2]</a> 
Bannach, Max & Berndt, Sebastian & Ehlers, Thorsten. (2017). Jdrasil: A Modular Library for Computing Tree Decompositions. 10.4230/LIPIcs.SEA.2017.28. 

<a id="3">[3]</a> 
Hammerl T., Musliu N., Schafhauser W. (2015) Metaheuristic Algorithms and Tree Decomposition. In: Kacprzyk J., Pedrycz W. (eds) Springer Handbook of Computational Intelligence. Springer Handbooks. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-43505-2_64

<a id="4">[4]</a> 
Bodlaender H.L., Koster A.M.C.A., Wolle T. (2004) Contraction and Treewidth Lower Bounds. In: Albers S., Radzik T. (eds) Algorithms – ESA 2004. ESA 2004. Lecture Notes in Computer Science, vol 3221. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-30140-0_56

<a id="5">[5]</a> 
Eijkhof, Frank & Bodlaender, Hans. (2002). Safe Reduction Rules for Weighted Treewidth. 176-185. 

<a id="6">[6]</a> 
Hans L. Bodlaender, Arie M.C.A. Koster,
Safe separators for treewidth,
Discrete Mathematics,
Volume 306, Issue 3,
2006,
Pages 337-350,
ISSN 0012-365X,
https://doi.org/10.1016/j.disc.2005.12.017.

<a id="7">[7]</a> 
Dell, Holger, Komusiewicz, Christian, Talmon, Nimrod, Weller, Mathias
"The PACE 2017 Parameterized Algorithms and Computational Experiments Challenge: The Second Iteration" (2018) DOI: 10.4230/LIPIcs.IPEC.2017.30

<a id="8">[8]</a> 
Tamaki, H.. “Positive-instance driven dynamic programming for treewidth.” ESA (2017).

<a id="9">[9]</a> 
Bannach, Max and Sebastian Berndt. “Positive-Instance Driven Dynamic Programming for Graph Searching.” WADS (2019).

# License
This Software is licensed under the MIT-License which can be found in the `LICENSE` file in this repository.

