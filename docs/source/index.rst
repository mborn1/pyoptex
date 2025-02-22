Welcome to PyOptEx documentation!
=================================

**Date**: |date|

**Version**: |release|

PyOptEx (or Python Optimal Experiments) is a package designed to create
optimal design of experiments with Python. 
It is fully open source and can be used for any purpose.

The package is designed for both engineers, and design of experiment
researchers. Engineers can use the precreated functions
to generate designs for their problems. Researchers can easily develop
new metrics (criteria) and test them. If you would like a refresher
on the concept of optimal design of experiments, see :ref:`doe`.

To generate experimental designs, there are two main options:

* **Fixed structure**: These designs have a fixed number of runs
  and fixed randomization structure, known upfront. Well-known designs
  include split-plot, strip-plot, and regular staggered-level designs.
  A specialization is also included for split\ :sup:`k`\ -plot designs
  using the update formulas as described in 
  `Born and Goos (2025) <https://www.sciencedirect.com/science/article/pii/S0167947324001129>`_.
  Go to :ref:`qc_first_design` and :ref:`qc_splitk` for an example respectively.
* **Cost-optimal designs**: These design generation algorithms follow
  a new DoE philosophy. Instead of fixing the number of runs and randomization
  structure, the algorithm optimizes directly based on the underlying resource
  constraints. The user must only specify a budget and a function which
  computes the resource consumption of a design. Go to :ref:`qc_cost` for
  an example. The currently implemented algorithm is CODEX.

.. figure:: /assets/img/pyoptex_overview.svg
  :width: 400
  :alt: Pyoptex package overview
  :align: center

  The overview of the PyOptEx package.

See the design of experiments :ref:`quickstart` for more information on how to generate different kinds of
designs. See :ref:`customization` for a more detailed explanation on how to tune and 
customize each algorithm. Reseachers can find more information here
on how to design custom criteria. The example scenarios are noted in :ref:`d_example_scenarios`.
Finally, see :ref:`performance` for some tips on how
to make the algorithm run faster.

To analyze the data after the experiment, have a look at the analysis :ref:`a_quickstart`.

Main features
-------------

* The **first complete Python package for optimal design of experiments**. Model 
  :ref:`everything <d_example_scenarios>` including continuous factors, categorical factors, mixtures, 
  blocked experiments, split-plot experiments, staggered-level experiments.
* **Intuitive design of experiments** with :ref:`cost-optimal designs <qc_cost>` for everyone. 
  No longer requires expert statistical knowledge before creating experiments.
* Accounts for **any constraint** you require. Not only can you choose the
  randomization structure :ref:`manually <qc_other_fixed>`, or let the :ref:`cost-optimal <qc_cost>` 
  design algorithms figure it out automatically, you can also specify the 
  physically :ref:`possible factor combinations <cust_constraints>` for a run.
* **Augmenting** designs was never easier. Simply read your initial design
  to a pandas dataframe and augment it by passing it as a :ref:`prior <cust_augment>`.
* **Customize** any part of the algorithm, including the :ref:`optimization criteria <cust_metric>` (metrics),
  :ref:`linear model <cust_model>`, :ref:`encoding of the categorical factors <cust_cat_encoding>`, and much more.
* Directly optimize for **Bayesian** :ref:`a-priori variance ratios <cust_bayesian_ratio>` in designs with 
  hard-to-change factors.
* High-performance **model selection** using :ref:`SAMS <a_cust_sams>` (simulated annealing model selection)
  `(Wolters and Bingham, 2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_.

Documentation
-------------

.. toctree::
   :maxdepth: 3

   general 
   doe
   analysis


.. toctree::
   :maxdepth: 2

   api
