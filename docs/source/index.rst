Welcome to PyOptEx documentation!
===================================

**Date**: |date|

**Version**: |release|

PyOptEx (or Python Optimal Experiments) is a package designed to create
optimal designs for experiments with Python. The package designed to be
as intuitive as possible, making it accessible also to non-statisticians. 
The package is fully open source and can be used for any purpose.

The package is designed for both engineers, and design of experiment
researchers. Engineers can use the precreated functions
to generate designs for their problems. Researchers can easily develop
new metrics (criteria) and test them.

To generate experimental designs, there are two main options:

* **Fixed structure**: These designs have a fixed number of runs
  and fixed randomization structure known upfront. Well-known designs
  include split-plot, strip-plot, and regular staggered-level designs.
  A specialization is also included for split\ :sup:`k`\ -plot designs
  using the update formulas as described in 
  `Born and Goos (2025) <https://www.sciencedirect.com/science/article/pii/S0167947324001129>`_.
  Go to :ref:`qc_first_design` and :ref:`qc_splitk` for an example respectively.
* **Cost-optimal designs**: These design generation algorithms follow
  a new DoE philosophy. Instead of fixing the number of runs and randomization
  structure, the algorithm optimizes directly on the underlying resource
  constraints. The user must only specify a budget and a function which
  computes the resource consumption of a design. Go to :ref:`qc_cost` for
  an example. Currently, only one such algorithm is implemented: CODEX.

.. figure:: /assets/img/pyoptex_overview.svg
  :width: 400
  :alt: Pyoptex package overview
  :align: center

  The overview of the PyOptEx package.

See the :ref:`quickstart` for more information on how to generate different kinds of
designs. See :ref:`customization` for a more detailed explanation on how to tune and 
customize each algorithm. Reseachers can find more information here
on how to design custom criteria. Finally, see :ref:`performance` for some tips on how
to make the algorithm run faster.

.. note::

   This project is under active development. Analysis and model selection will follow.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   customization
   performance


.. toctree::
   :maxdepth: 2

   api
