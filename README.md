# PyOptEx

| | |
| --- | --- |
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/pyoptex.svg)](https://pypi.org/project/pyoptex/) [![PyPI Downloads](https://img.shields.io/pypi/dm/pyoptex.svg?label=PyPI%20downloads)](https://pypi.org/project/pyoptex/) |
| Meta | [![License - BSD 3-Clause](https://img.shields.io/pypi/l/pyoptex.svg)](https://github.com/mborn1/pyoptex/blob/main/LICENSE) [![docs](https://img.shields.io/readthedocs/pyoptex)](https://pyoptex.readthedocs.io/en/latest/) |

Welcome! PyOptEx is a Python package to create experimental designs.
The focus is on accessibility for both engineers from industry using design
of experiments, and researchers willing to develop new criteria or
design structures.

The main contributions of the package are cost-optimal designs which shift
the philosophy of requiring expert knowledge and fixing the number of experiments,
number of plots in a split-plot experiment, to the an optimization algorithm which
determines the optimal number of runs and design structure based on the
underlying resource constraints. This means that engineers can simply specify
money and time as a resource and optimize for the available budget.

In case the design structure is not predetermined by the physicalities of the
experiment, cost-optimal experiments also generate significantly better designs.

**_NOTE:_**  This package does not have a release version yet and is still under active development.

## Getting started

Install this package using pip

```
pip install pyoptex
```

## Documentation
The documentation for this package can be found at [here](https://pyoptex.readthedocs.io/en/latest/)

## Create your first design
See the documentation on [Your first design](https://pyoptex.readthedocs.io/en/latest/quickstart.html)

## License
BSD-3 clause, meaning you can use and alter it for any purpose,
open-source or commercial!
However, any open-source contributions to this project are much
appreciated by the community.

## Contributing
Any ideas, bugs and features requests can be added as an [issue](https://github.com/mborn1/pyoptex/issues). Any direct code contributions can be added via pull requests.
