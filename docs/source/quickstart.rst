.. |link-qc-pre| raw:: html

  <a href="https://github.com/mborn1/pyoptex/blob/

.. |link-qc-mid0| raw:: html

  /examples/quickstart/

.. |link-qc-mid1| raw:: html

  ">

.. |link-qc-post| raw:: html

  </a>

.. _quickstart:

Quickstart
==========

Installation
------------

To use PyOptEx, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyoptex


Terminology
-----------
This section summarizes some commonly used terminology and notations. You can
freely skip this and come back once a term is unclear.

* **Y**: the design matrix. When decoded, every factor is a single
  column. When encoded, the categorical factors are expanded to
  multiple dummy columns.
* **X**: the model matrix, or X = Y2X(Y), where Y is an encoded
  design matrix.
* **Z** or **Zs**: These are the grouping matrices for the random
  effects. Each Z assigns a group number to each run for a specific
  random effect. For example [0, 0, 1, 1] states that the first two
  runs are intercorrelated, and the last two are intercorrelated. 
* **V**: The V-matrix is the observation covariance matrix of Y.
  It is equal to :math:`V = \sum_{i=1}^k Z_i Z_i^T + I_N` with `k`
  the number of random effects (Zs), and `I` the identity matrix of
  size `N`.
* **Vinv**: The inverse of V.
* **M** : The information matrix: :math:`X^T V^{-1} X`. :math:`M^{-1}`
  is the covariance matrix of the parameter estimates in X.
* **encoded**: refers to a design matrix for which the categorical
  factors are dummy encoded.
* **decoded**: refers to a design matrix for which every factor,
  including the categorical factors, is a single column. The
  categorical factors are generally numbers from 0 up to the number
  of levels.
* **normalized**: refers to the design matrix being normalized between
  -1 and 1. A normalized design matrix is always encoded.
* **denormalized**: refers to the design matrix with each column representing
  one factor, denormalized to their original levels and units.
  A continuous factor will be between its own min and max, a categorical factor
  is a column of strings representing the level name. A denormalized
  design matrix is always decoded.
* **plot** or **stratum**: A group of runs that are correlated and are modeled
  with a random effect.
* **metric** or **criterion**: The optimization objective for the
  algorithm.
* **continuous** or **quantitative**: Refers to a factor having a value on
  a continuous, measureable scale. The values are comparable and sortable.
* **categorical** or **qualitative**: Refers to a factor having a predetermined
  set of possible levels. The values are not comparable, but not sortable.
* **cost function**: The function which computes the resource consumption of the
  design matrix.
* **cost** or **resource consumption**: The cost or amount of resources consumed
  for the design.

.. _qc_first_design:

Create your first design
------------------------

.. note::
  If you would like a refresher on optimal design of experiments, see
  :ref:`doe`.

We will start by creating a fully randomized D-optimal design 
with 20 runs, one categorical and two continuous factors, 
using the coordinate-exchange algorithm. We are using the
:py:mod:`fixed_structure <pyoptex.doe.fixed_structure>` submodule 
for this.

.. note::
  The complete Python script for the generation of such a design can be
  found in |link-qc-pre|\ |version|\ |link-qc-mid0|\ example_randomized_fs.py\ |link-qc-mid1|\ example_randomized_fs.py\ |link-qc-post|.

Start by importing the necessary modules

>>> # Python imports
>>> import os
>>> import time
>>> 
>>> # PyOptEx imports
>>> from pyoptex._seed import set_seed
>>> from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
>>> from pyoptex.doe.fixed_structure import (
>>>     Factor, create_fixed_structure_design, create_parameters, default_fn
>>> )
>>> from pyoptex.doe.fixed_structure.metric import Dopt

We define the number of runs

>>> nruns = 20

Next, we define the factors for our experiment. We have one categorical
factor A with levels L1, L2, and L3. Next we also define two continuous
factors B, and C. By default, factor B is in the range [-1, 1]. However,
by specifying the `min` and `max` properties, we can define C in the
range [2, 5].

>>> factors = [
>>>     Factor('A', type='categorical', levels=['L1', 'L2', 'L3']),
>>>     Factor('B', type='continuous'),
>>>     Factor('C', type='continuous', min=2, max=5),
>>> ]

.. note::
   By default, a continuous factor is discretised to three points 
   [low, mid, high]. If a higher degree of discretization is desired,
   see :ref:`cust_disc_num`.

.. note::
   The encoding of the categorical factors can also be customized
   using the `coords` parameter. See :ref:`cust_cat_encoding`
   for more information.

Finally, we must define a model. We define a full response surface model
with 9 parameters, including the intercept, all three main effects,
three two-factor interactions, and two quadratic effects of the factors
B and C. The first command creates a matrix representation of the model,
the second converts this matrix representation to a callable function
which transforms a design matrix (Y) to a model matrix (X).

>>> model = partial_rsm_names({
>>>     'A': 'tfi',
>>>     'B': 'quad',
>>>     'C': 'quad',
>>> })
>>> Y2X = model2Y2X(model, factors)

.. note::
   Any custom linear model can be used. See :ref:`cust_model`
   for more information.

We must also specify the metric which we want to optimize.
In this case, we optimize for D-optimality (namely accurate
parameter estimates).

>>> metric = Dopt()

.. note::
   Metrics can also be fully customized. See :ref:`cust_metric`
   for more information.

Finally, we are ready to generate a design using the following
code snippet.

>>> # Parameter initialization
>>> n_tries = 10
>>> 
>>> # Create the set of operators
>>> fn = default_fn(metric, Y2X)
>>> params = create_parameters(factors, fn, nruns)
>>> 
>>> # Create design
>>> start_time = time.time()
>>> Y, state = create_fixed_structure_design(params, n_tries=n_tries)
>>> end_time = time.time()

The function :py:func:`create_fixed_structure_design <pyoptex.doe.fixed_structure.wrapper.create_fixed_structure_design>` 
returns a dataframe `Y` containing the design, and the final internal
state of the algorithm which contains the encoded design matrix, model matrix,
and metric value.

We can write the design to a csv

>>> root = os.path.split(__file__)[0]
>>> Y.to_csv(os.path.join(root, 'example_randomized_fs.csv'), index=False)

And we can print the final metric, execution time and design to the
console.

>>> print('Completed optimization')
>>> print(f'Metric: {state.metric:.3f}')
>>> print(f'Execution time: {end_time - start_time:.3f}')
>>> print(Y)

More information on how to evaluate the design in :ref:`qc_evaluation`.

.. note::
  A split-plot design with only one stratum, the easy-to-change stratum
  is also a fully randomized design. Because of the update formulas,
  creating a randomized design with the
  :py:func:`create_splitk_plot_design <pyoptex.doe.fixed_structure.splitk_plot.wrapper.create_splitk_plot_design>`
  may be faster.
  Such an example script may be found in
  |link-qc-pre|\ |version|\ |link-qc-mid0|\ example_randomized_sp.py\ |link-qc-mid1|\ example_randomized_sp.py\ |link-qc-post|


.. _qc_splitk:

Creating a split\ :sup:`k`\ -plot design
----------------------------------------

What if the factor A was actually a component that was hard-to-change?
In such a scenario, design of experiments literature recommends
the use of a split-plot design, where the factor A is no longer
reset with every run. We will create a split-plot design
with 5 whole plots and 4 runs per whole plot.

.. note::
  The Python script for the generation of such a design can be
  found in 
  |link-qc-pre|\ |version|\ |link-qc-mid0|\ example_splitplot_sp.py\ |link-qc-mid1|\ example_splitplot_sp.py\ |link-qc-post|.

To create a split-plot design, first,
we require the imports again.

>>> # Python imports
>>> import os
>>> import time
>>> import numpy as np
>>> 
>>> # PyOptEx imports
>>> from pyoptex._seed import set_seed
>>> from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
>>> from pyoptex.doe.fixed_structure import Factor
>>> from pyoptex.doe.fixed_structure.splitk_plot import (
>>>     create_splitk_plot_design, default_fn, create_parameters, Plot
>>> )
>>> from pyoptex.doe.fixed_structure.splitk_plot.metric import Dopt

Note that we now import most from :py:mod:`splitk_plot <pyoptex.doe.fixed_structure.splitk_plot>`
instead of :py:mod:`fixed_structure <pyoptex.doe.fixed_structure>`.
Next, we define the hard-to-change and easy-to-change plot (or stratum).

>>> etc = Plot(level=0, size=4)
>>> htc = Plot(level=1, size=5, ratio=0.1)
>>> plots = [etc, htc]
>>> nruns = np.prod([p.size for p in plots])

.. note::
   Split-plot designs require the user to specify an estimate of 
   the ratio between the variance of the random effect and the random error,
   here noted on line 2. Generally, a value of `1` is a good estimate,
   however, a Bayesian approach is also possible. See :ref:`cust_bayesian_ratio`
   for more information.

We specify the factors with the stratum they are in.

>>> factors = [
>>>     Factor('A', htc, type='categorical', levels=['L1', 'L2', 'L3']),
>>>     Factor('B', etc, type='continuous'),
>>>     Factor('C', etc, type='continuous', min=2, max=5),
>>> ]

And like in :ref:`qc_first_design`, we define the optimization metric
as D-optimality

>>> metric = Dopt()

Finally, we generate the split-plot design.

>>> # Parameter initialization
>>> n_tries = 10
>>> 
>>> # Create the set of operators
>>> fn = default_fn(metric, Y2X)
>>> params = create_parameters(factors, fn)
>>> 
>>> # Create design
>>> start_time = time.time()
>>> Y, state = create_splitk_plot_design(params, n_tries=n_tries)
>>> end_time = time.time()

.. note::
   Adding more plots is as easy as specifying higher levels and assigning
   factors to them. For example, the very-hard-to-change factors in a 
   split-split-plot design would have a 
   
   >>> `vhtc = Plot(level=2)`.

More information on how to evaluate the design in :ref:`qc_evaluation`.

.. note::
  While a split-plot design can also be created using
  :py:func:`create_fixed_structure_design <pyoptex.doe.fixed_structure.wrapper.create_fixed_structure_design>`,
  using :py:func:`create_splitk_plot_design <pyoptex.doe.fixed_structure.splitk_plot.wrapper.create_splitk_plot_design>`
  is generally faster due to the update formulas.

.. _qc_other_fixed:

Creating other fixed structure designs
--------------------------------------

Not every design is either randomized or a split-plot design.
For instance, a strip-plot design defines multiple non-sequential runs
to be grouped together. For any scenario where the randomization
structure does not depend on the design and the number of runs is fixed,
you can use the :py:func:`create_fixed_structure_design <pyoptex.doe.fixed_structure.wrapper.create_fixed_structure_design>`.

Let's create a simple strip-plot design with 5 plots and 4 runs per plot.

.. note::
  The Python script for the generation of such a design can be
  found in 
  |link-qc-pre|\ |version|\ |link-qc-mid0|\ example_strip_plot_fs.py\ |link-qc-mid1|\ example_strip_plot_fs.py\ |link-qc-post|.

Like all previous examples, we start with the imports

>>> # Python imports
>>> import os
>>> import time
>>> import numpy as np
>>> 
>>> # PyOptEx imports
>>> from pyoptex._seed import set_seed
>>> from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
>>> from pyoptex.doe.fixed_structure import (
>>>     Factor, RandomEffect, create_fixed_structure_design, 
>>>     create_parameters, default_fn
>>> )
>>> from pyoptex.doe.fixed_structure.metric import Dopt

Next, we define the random effect for a strip-plot design.

>>> nruns = 20
>>> nplots = 5
>>> re = RandomEffect(np.tile(np.arange(nplots), nruns//nplots), ratio=0.1)

Next, define the factors. Note that we assign A to the first
random effect.

>>> factors = [
>>>     Factor('A', re, type='categorical', levels=['L1', 'L2', 'L3']),
>>>     Factor('B', type='continuous'),
>>>     Factor('C', type='continuous', min=2, max=5),
>>> ]

Finally, we compute the design

>>> # Create a partial response surface model
>>> model = partial_rsm_names({
>>>     'A': 'tfi',
>>>     'B': 'quad',
>>>     'C': 'quad',
>>> })
>>> Y2X = model2Y2X(model, factors)
>>> 
>>> # Define the metric
>>> metric = Dopt()
>>> 
>>> # Parameter initialization
>>> n_tries = 10
>>> 
>>> # Create the set of operators
>>> fn = default_fn(metric, Y2X)
>>> params = create_parameters(factors, fn, nruns)
>>> 
>>> # Create design
>>> start_time = time.time()
>>> Y, state = create_fixed_structure_design(params, n_tries=n_tries)
>>> end_time = time.time()

You will now notice that the resulting design
has the same setting of factor A for runs
[1, 5, 9, 13, 17], the first plot of the strip-plot design

.. note::
  If you want to force certain level constraints like in a
  strip-plot design, but you do not want any random effect
  associated, simply set the ratio of the random effect
  to zero.

.. _qc_cost:

Creating a cost-optimal design
------------------------------

Why use cost-optimal designs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cost optimal designs shift the philosphy of creating designs.
Historically, an experiment was always analyzed by a statistician 
who determines whether to use a randomized design, a split-plot design,
a split-split-plot design, etc. That person would then proceed to 
make an estimation about the number of runs that could be performed,
the sizes of the plots in a split\ :sup:`k`\ -plot design, etc.

.. figure:: /assets/img/classical_procedure.svg
  :width: 100%
  :alt: classical procedure
  :align: center

  Classic optimal design procedure.

All these estimations require expert knowledge in the field of
design of experiments, which most often engineers do not possess.
In case the experiment is very complicated, any estimation made by
the statistician may not even be optimal.

Cost optimal designs avoid these issues by directly optimizing based
on the underlying resource constraints. These constraints can be
time (when dealing with hard-to-change factors), money, availability of
certain components or ingredients in stock, etc. The algorithm proceeds
to automatically determine the optimal number of runs, run order, etc.
Most often, this approach yields better designs, while
simulatneously making it easier, more comprehensible, and faster 
for engineers to create designs. They spend less time on researching the
best design, and can spend more time actually executing their design and analyzing
the data.

.. figure:: /assets/img/cost_optimal_procedure.svg
  :width: 100%
  :alt: cost optimal procedure
  :align: center

  Cost optimal design procedure.

The generalized staggered-level design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The design generated by this algorithm is a generalized staggered-level design.
Mathematically, the design assumes any hard-to-change factor is only reset
if the factor changes its level. In constrast to split-plot designs and
regular staggered-level designs which assume a reset at fixed locations in 
the design. The figure below depicts the difference in interpretation.
Both left and right are the same design, however, the runs are grouped
differently in the middle column. The split-plot design requires a reset
in for the second factor whenever the first resets. The generalized-staggered
level design only resets when the factor level changes.


.. list-table::
  :align: center
  :widths: 1 1

  * - .. figure:: /assets/img/interpretation_splitk_plot.svg
        :width: 100%
        :alt: Splitk-plot interpretation
        :align: center

        Split\ :sup:`k`\ -plot interpretation.

    - .. figure:: /assets/img/interpretation_stagg_level.svg
        :width: 100%
        :alt: (Generalized) Staggered-level interpretation
        :align: center

        (Generalized) Staggered-level interpretation.

The problem with resets at fixed locations is that when, by accident, both
consecutive levels are the same, the technician may refrain from resetting
the factor. For example, if this factor is a mechanical component of a product, 
a technician may not want to dissassemble and reassemble the product the
exact same way. This leads to a mismatch between what the experimenter desired,
and what was actually executed.

.. _qc_codex:

An example (CODEX)
^^^^^^^^^^^^^^^^^^

Let's create a design with one categorical factor and three continuous
factors. The categorical factor A is hard-to-change and has four levels
L1, L2, L3, and L4. The three continuous factors, E, F, and G, are easy-to-change. We will
optimize for I-optimality with a full response surface model.

As we are dealing with hard-to-change factors, our limiting resource
is time. We will be using 3 days of 4 hours each, for a total of 720 minutes.
To reset factor A, we require 2 hours. To reset any of the factors E, F, or G,
we require only a single minute (they are easy-to-vary). The execution cost of a single
run is 5 minutes. Some times, multiple factors are reset simultaneously. In this
case, we assume that the transition cost is determined by the most-hard-to-change factor.
Such a scenario arises when multiple workers or technicians can work in parallel on their
own task.

.. note::
  The Python script for the generation of such a design can be
  found in 
  |link-qc-pre|\ |version|\ |link-qc-mid0|\ example_cost_optimal_codex.py\ |link-qc-mid1|\ example_cost_optimal_codex.py\ |link-qc-post|.


First start with the necessary imports

>>> # Python imports
>>> import time
>>> import os
>>> 
>>> # PyOptEx imports
>>> from pyoptex._seed import set_seed
>>> from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
>>> from pyoptex.doe.cost_optimal import Factor
>>> from pyoptex.doe.cost_optimal.metric import Iopt
>>> from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
>>> from pyoptex.doe.cost_optimal.codex import (
>>>     create_cost_optimal_codex_design, default_fn, create_parameters
>>> )

Then we define the factors. We define factor A as categorical, and the other
three factors E, F, G are continuous and easy-to-vary by setting the `group` 
parameter to `False`. Easy-to-change parameters are assumed to be reset
with every run, no matter the factor level. 
Factor F is also considered to be between [2, 5] instead
of the default [-1, 1].

>>> factors = [
>>>     Factor('A', type='categorical', levels=['L1', 'L2', 'L3', 'L4']),
>>>     Factor('E', type='continuous', grouped=False),
>>>     Factor('F', type='continuous', grouped=False, min=2, max=5),
>>>     Factor('G', type='continuous', grouped=False),
>>> ]

.. note::
   Every hard-to-change factor has a random effect associated with itself.
   The ratio can be specified using a `ratio` parameter and is set to `1`
   by default, which is generally a good estimate. In addition, the user can also opt 
   for a Bayesian approach. See :ref:`cust_bayesian_ratio` for more information.

Next, we define the response surface model. Every continuous factor is
added with their main effect, two-factor interactions, and quadratic effect.
The categorical factor is only added as a main effect and two-factor interaction.
Similar to :ref:`qc_first_design`, the second command converts the matrix of the
model to a callable.

>>> model = partial_rsm_names({
>>>     'A': 'tfi',
>>>     'E': 'quad',
>>>     'F': 'quad',
>>>     'G': 'quad'
>>> })
>>> Y2X = model2Y2X(model, factors)

.. note::
   Any linear model can be used. See :ref:`cust_model` for more information.

We must also specify the optimization criterion. In this case, I-optimality.

>>> metric = Iopt()

.. note::
   Any optimization metric can be used. See :ref:`cust_metric` for more information.

We create the cost function using the
:py:func:`parallel_worker_cost <pyoptex.doe.cost_optimal.cost.parallel_worker_cost>`
helper function. This cost function defines that the cost of transition between two
consecutive runs is equal to the transition cost of the most-hard-to-change factor.
Such a scenario arises when multiple workers or technicians can work in parallel on their
own task.

>>> max_transition_cost = 3*4*60
>>> transition_costs = {
>>>     'A': 2*60,
>>>     'E': 1,
>>>     'F': 1,
>>>     'G': 1
>>> }
>>> execution_cost = 5
>>> cost_fn = parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

.. note::
   The power of the algorithm is in the possibility to define your own
   cost function. For more information, see :ref:`cust_cost`.

Finally, we can generate the design

>>> # Simulation parameters
>>> nsims = 10
>>> nreps = 1
>>> fn = default_fn(nsims, cost_fn, metric, Y2X)
>>> params = create_parameters(factors, fn)
>>> 
>>> # Create design
>>> start_time = time.time()
>>> Y, state = create_cost_optimal_codex_design(
>>>     params, nsims=nsims, nreps=nreps
>>> )
>>> end_time = time.time()

Similar to :ref:`qc_first_design`, 
:py:func:`create_cost_optimal_codex_design <pyoptex.doe.cost_optimal.codex.wrapper.create_cost_optimal_codex_design>`
returns the design `Y` and the corresponding internal state
with the encoded design matrix, model matrix, metric, cost, etc.

We can write the design to a csv

>>> root = os.path.split(__file__)[0]
>>> Y.to_csv(os.path.join(root, f'example_cost_optimal_codex.csv'), index=False)

And we can print the resulting metric, cost, number of experiments and
execution time to the console.

>>> print('Completed optimization')
>>> print(f'Metric: {state.metric:.3f}')
>>> print(f'Cost: {state.cost_Y}')
>>> print(f'Number of experiments: {len(state.Y)}')
>>> print(f'Execution time: {end_time - start_time:.3f}')


.. _qc_evaluation:

Evaluation
----------

Evaluating the resulting design is just as important as correctly
generating them. In order to ease the evaluation, some common
functions have been pre-implemented.

First, we can do a generic evaluation. The first command imports the necessary
functions, the second plots the design graphically, and the last command
plots the color map on correlations for the design.

>>> from pyoptex.doe.utils.evaluate import design_heatmap, plot_correlation_map
>>> design_heatmap(Y, factors).show()
>>> plot_correlation_map(Y, factors, fn.Y2X, model=model).show()

.. list-table::
  :align: center
  :widths: 1 1

  * - .. figure:: /assets/img/heatmap.svg
        :width: 100%
        :alt: The design heatmap
        :align: center

        A heatmap of the design, |br|
        to be executed from top to bottom.

    - .. figure:: /assets/img/corrmap.svg
        :width: 100%
        :alt: Color map on correlations
        :align: center

        The color map on correlations |br|
        between the different factors.

The next evaluations depend on how the design should be interpreted.
Is it a fixed structure design, or a cost-optimal design
(generalized staggered-level design).
Depending on the type, the imports are different.

For a fixed structure design

>>> from pyoptex.doe.fixed_structure.evaluate import (
>>>     evaluate_metrics, plot_fraction_of_design_space, 
>>>     plot_estimation_variance_matrix, estimation_variance
>>> )

For a cost-optimal design (generalized staggered-level design) 

>>> from pyoptex.doe.cost_optimal.evaluate import (
>>>     evaluate_metrics, plot_fraction_of_design_space, 
>>>     plot_estimation_variance_matrix, estimation_variance
>>> )

Once imported, we can evaluate the design. The first command prints the metric value
for the different provided metrics to the console. The second command
plots a fraction of design space plot. The third command plots the covariance
matrix of the parameter estimates. Finally, the last commands prints the variances
of the parameter estimates to the console. 

The `params` are the simulation parameters which are passed to the
design generation functions.

>>> print(evaluate_metrics(Y, params, [metric, Dopt(), Iopt(), Aopt()]))
>>> plot_fraction_of_design_space(Y, params).show()
>>> plot_estimation_variance_matrix(Y, params, model).show()
>>> print(estimation_variance(Y, params))

.. list-table::
  :align: center
  :widths: 1 1
  :class: align-top

  * - .. figure:: /assets/img/fraction_design_space.svg
        :width: 100%
        :alt: The fraction of design space plot
        :align: center

        The fraction of design space plot, |br|
        for each set of variance ratios.

    - .. figure:: /assets/img/estimation_var.svg
        :width: 100%
        :alt: The covariance of the parameter estimates
        :align: center

        The covariance matrix of the parameter estimates.
