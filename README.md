# PyOptEx

| | |
| --- | --- |
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/pyoptex.svg)](https://pypi.org/project/pyoptex/) [![PyPI Downloads](https://img.shields.io/pypi/dm/pyoptex.svg?label=PyPI%20downloads)](https://pypi.org/project/pyoptex/) |
| Meta | [![License - BSD 3-Clause](https://img.shields.io/pypi/l/pyoptex.svg)](https://github.com/mborn1/pyoptex/blob/main/LICENSE) [![docs](https://img.shields.io/readthedocs/pyoptex)](https://pyoptex.readthedocs.io/en/latest/) |


PyOptEx (or Python Optimal Experiments) is a package designed to create optimal design of experiments with Python. It is fully open source and can be used for any purpose.

The package is designed for both engineers, and design of experiment researchers. Engineers can use the precreated functions to generate designs for their problems,
especially the cost-optimal algorithms. Researchers can easily develop new metrics (criteria) and test them.

To generate experimental designs, there are two main options:

* **Fixed structure**: These designs have a fixed number of runs and fixed randomization
  structure, known upfront. Well-known designs include split-plot, strip-plot, and 
  regular staggered-level designs. A specialization is also included for splitk-plot
  designs using the update formulas as described in 
  [Born and Goos (2025)](https://www.sciencedirect.com/science/article/pii/S0167947324001129).

* **Cost-optimal designs**: These design generation algorithms follow a new 
  DoE philosophy. 
  Instead of fixing the number of runs and randomization structure, the algorithm 
  optimizes directly based on the underlying resource constraints. The user must only 
  specify a budget and a function which computes the resource consumption of a design. 
  Go to Creating a cost-optimal design for an example. The currently implemented 
  algorithm is CODEX.

**_NOTE:_**  This package does not have a release version yet and is still under active development.

## Main features

* The **first complete Python package for optimal design of experiments**. Model
  [everything](https://pyoptex.readthedocs.io/en/latest/_docs/doe/example_scenarios.html#example-scenarios) including continuous factors, categorical factors, 
  mixtures, blocked experiments, split-plot experiments, staggered-level experiments.

* **Intuitive design of experiments** with 
  [cost-optimal designs](https://pyoptex.readthedocs.io/en/latest/_docs/doe/quickstart.html#qc-cost) 
  for everyone. No longer requires expert statistical knowledge before creating
  experiments.

* Accounts for **any constraint** you require. Not only can you choose 
  the randomization structure 
  [manually](https://pyoptex.readthedocs.io/en/latest/_docs/doe/quickstart.html#qc-other-fixed), 
  or let the 
  [cost-optimal](https://pyoptex.readthedocs.io/en/latest/_docs/doe/quickstart.html#qc-cost) 
  design algorithms figure it out automatically, you can also specify the physically 
  possible factor combinations for a run.

* **Augmenting** designs was never easier. Simply read your initial design 
  to a pandas dataframe and augment it by passing it as a 
  [prior](https://pyoptex.readthedocs.io/en/latest/_docs/doe/customization.html#cust-augment).

* **Customize** any part of the algorithm, including the 
  [optimization criteria](https://pyoptex.readthedocs.io/en/latest/_docs/doe/customization.html#cust-metric) (metrics), 
  [linear model](https://pyoptex.readthedocs.io/en/latest/_docs/doe/customization.html#cust-model), 
  [encoding of the categorical factors](https://pyoptex.readthedocs.io/en/latest/_docs/doe/customization.html#cust-cat-encoding), 
  and much more.

* Directly optimize for **Bayesian** 
  [a-priori variance ratios](https://pyoptex.readthedocs.io/en/latest/_docs/doe/customization.html#cust-bayesian-ratio)
  in designs with hard-to-change factors.

* High-performance **model selection** using 
  [SAMS](https://pyoptex.readthedocs.io/en/latest/_docs/analysis/customization.html#a-cust-sams)
   (simulated annealing model selection)
  [(Wolters and Bingham, 2012)](https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157).

## Getting started

Install this package using pip

```
pip install pyoptex
```

## Documentation
The documentation for this package can be found at [here](https://pyoptex.readthedocs.io/en/latest/)

## Create your first design
See the documentation on [Your first design](https://pyoptex.readthedocs.io/en/latest/_docs/doe/quickstart.html)

## Analyze your first dataset
See the documentation on [Your first dataset](https://pyoptex.readthedocs.io/en/latest/_docs/analysis/quickstart.html)

## License
BSD-3 clause, meaning you can use and alter it for any purpose,
open-source or commercial!
However, any open-source contributions to this project are much
appreciated by the community.

## Contributing
Any ideas, bugs and features requests can be added as an [issue](https://github.com/mborn1/pyoptex/issues). Any direct code contributions can be added via pull requests.
