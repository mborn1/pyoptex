.. _doe:

Optimal design of experiments (DoE)
===================================

What is optimal design of experiments?
--------------------------------------

Design of experiments is a branch of statistics concerned with finding or computing
the ideal combinations of different effects, also known as the independent variables,
to explain the variation in the later observed data. For example, in linear regression,
we want to optimize the value of :math:`X`, to later optimally estimate :math:`\beta`, 
:math:`Y` or something else.

.. math::

    Y = X \beta + \epsilon

These exact values of :math:`X` are commonly computed using a computer with the point-
(`Fedorov, 1972 <https://www.google.be/books/edition/Theory_Of_Optimal_Experiments/PwUz-uXnImcC?hl=en&gbpv=1&printsec=frontcover>`_) 
or coordinate-exchange (`Meyer and Nachtsheim, 1995 <https://www.jstor.org/stable/1269153>`_) algorithm.

Why use design of experiments?
------------------------------

Today, we are in the era of big data. Big machine learning models and advanced 
data mining techniques can make sense of large amounts of unstructured data. 
However, often, data is not so cheap to come by. In this case, we must be 
selective on the tests we perform. In addition, sometimes, targeted 
experimentation is much more informative and faster than simply gathering 
random data without a clear objective in mind.

Let us first look at how historically experiments were performed. Assume
you want to make a cake from a premixed dough and want to determine the ideal oven temperature
and time in the oven. Both the `temperature` and `time` are called factors.
You only have the ingredients to make five cakes.

You first start with what you think it best, you start at 180°C and 10 minutes.
You note the quality of the cake after tasting and decide you want change the 
temperature. You bake two more cakes at 200°C and 220°C, both for 10 minutes.
Suppose you see 220°C is the best for such a small amount of time. The other cakes
are not fully baked. You now decide to continue your experiment by
increasing the time to 15 minutes and 20 minutes respectively. You finally note
that both at 15 minutes and 20 minutes, the cake looks burned, so you decide
220°C and 10 minutes is the best.

However, you came to a bad conclusion, because, it turns out, 200°C and 15 minutes
is actually optimal. But you do not have any information on this.
The previous form of experimentation is called "one-factor-at-a-time" and
is still extremely common in practice because researchers can clearly see
the effect of changing either the temperature or the baking time.

The problem is that the one-factor-at-a-time methodology does not gather
information optimally, and additionally does not gather any information
on the interaction effect between temperature and time. For example, how does the effect
of temperature change for different baking times. We only tested the effect of
temperature for 10 minutes of baking time.

Design of experiments allows you to gather much more information with the 
same budget, namely the five cakes. You compute upfront which temperature 
and baking time combinations you need to use. You bake the cakes, and 
then you fit a model, commonly
a regression model. From this model, you can decide much more accurately what 
the optimal temperature and baking times are.

Some terminology
----------------

Design of experiments has some specific terminology which we explain here:

* A **test** or **run** is, from the example above, the baking of a single 
  cake.
* A **design** or **experiment** is the entire collection of the tests.
* A **factor**, **variable**, **component** are, from the example above,
  the temperature of the oven and the baking time.
* A **coordinate** or **factor level** is the setting of one of the factors,
  in one of the runs. For example, the `10 minutes` in one of the runs.
* A **metric** or **criterion** is a mathematical representation of the desired
  objective. Many common criteria exist such as D-, I- and A-optimality.

Optimization criteria
---------------------

In order to generate an optimal design, we need to define what `optimal`
is. The optimization criterion defines mathematically what is optimal for your experiment.
Many different criteria exist. The most common are:

* **D-optimality** which was designed for model selection. If you have 
  many potential factors, but are not sure which ones actually affect the outcome
  of the experiment, you need to be able to estimate each parameter as
  individually as possible.
* **A-optimality** which was also designed for model selection. However, the 
  difference with D-optimality is that A-optimality only looks at the
  variances of the parameters estimates, not the covariances between
  different parameters. Some say use A-optimality, others say use D-optimality for
  model selection.
* **I-optimality** which focuses on the predictions of the corresponding model.
  It minimizes the uncertainty of future predictions, meaning it is the
  ideal criterion to use when optimizing a process.

How to use design of experiments?
---------------------------------

In this section, we will reuse the example of baking cake in an oven.
We first provide the flowcharts and then explain them in the sections
below.

.. list-table::
  :align: center
  :widths: 1 1
  :class: align-top no-border

  * - .. figure:: /assets/img/doe_flowchart.svg
        :width: 100%
        :alt: Fixed structure flow
        :align: center

        The flow when manually selecting a |br|
        randomization structure (classic optimal design).

    - .. figure:: /assets/img/doe_flowchart_cost_optimal.svg
        :width: 100%
        :alt: Cost optimal flow
        :align: center

        The flow when automatically selecting |br|
        a randomization structure based on the |br|
        resource constraints (cost-optimal design).

The factors
^^^^^^^^^^^
To start with design of experiments, you must think about the process
you want to investigate and note down all potential effects on the outcome.
Next, divide these effects or factors in the three following categories:

* Observable and controllable
* Observable but not controllable
* Not observable

The first category can be used in the design as factors. Decide which
factors you wish to investigate and keep all other factors constant.
For example, we wish to investigate temperature and baking time, but not 
the number of added eggs. In this case, we should always use the same
number of eggs for each run.
The more factors you wish to investigate, the more resources (e.g. runs) 
you will require. Next, the second category is not controllable, but it 
is observable. If you can provide an estimate upfront, do so. The factors for which we have an estimate 
are called `covariates`. For example, if you can bake five cakes, two from 
one brand, and three from the other. The exact dough mixtures are as close 
as possible, but not the same. You can add these as non-controllable parameters
as they may influence the quality of the cake. If you cannot provide an estimate, 
keep them as constant as possible. Note again that adding covariates
results in more resources. Finally, the third category is 
the unknowns. For example, the ambient temperature if we do not 
have a thermometer available. Try to keep them as constant as possible by 
performing all runs after each other, on the same day.

The model
^^^^^^^^^
Next, based on the selected factors and covariates you need to determine 
the model :math:`X`. The model specifies which effects are investigated.
Do you want to investigate only the main effects and an intercept?

.. math::
    Y ~ a_0 + a_1 \cdot T + a_2 \cdot time

Do you want to include the 
interaction between time and temperature?

.. math::
    Y ~ a_0 + a_1 \cdot T + a_2 \cdot time + a_3 \cdot T \cdot time

Do you also want quadratic terms?

.. math::
    Y ~ a_0 + a_1 \cdot T + a_2 \cdot time + a_3 \cdot T \cdot time + a_4 \cdot T^2 + a_5 \cdot time^2

Do you have an exponential effect instead of interactions and quadratics?

.. math::
    Y ~ a_0 + a_1 \cdot T + a_2 \cdot time + a_3 \cdot e^{-T}

You must decide, based on prior knowledge, which model to choose. If you
do not have prior knowledge, a response surface design including all main effects, interactions
and quadratic effects is generally decent as a starting point if you have the resources.
Otherwise, look into screening designs. Note
that larger model have more flexibility, but also have more coefficients :math:`a_i`,
and therefore require more runs and resources.

.. note::
    When including categorical factors, a square does not make sense. The highest
    order in a response surface model for these factors is two-factor interaction.

.. note::
    We only specified linear models as these are most common. Non-linear models
    are also heavily researched, but are not possible in this package (yet).

The randomization structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you chose a model, you can determine which randomization structure you require.
Assume for example that you are baking cakes and multiple, consectutive runs have the 
same temperature. Or maybe you thought you were smart by reordering the runs so that 
you do not have to change the temperature of the oven too much. This is wrong.

Temperature is what we call a **hard-to-change** factor. It takes some time to 
preheat the oven, or to let it cool down for a lower temperature. We cannot 
(or do not want to) let the oven cool down completely and set the temperature
again from the beginning with every run. When not resetting the temperature with every run, the 
runs are no longer independent. Namely, assume there is a measurement error in the 
oven and the true temperature is not 180°C, but 185°C. If we do not reset the 
temperature, all consecutive runs at 180°C will actually be performed at 185°C. The 
next batch, after a reset, may be run at 179°C instead of 180°C. Not resetting 
a factor may result in additional correlations between the runs.

A more accurate statistical model is not OLS (ordinary least squares), which only 
includes a random error :math:`\epsilon`, but rather a linear mixed model, which includes one or more random effects
:math:`\gamma`. In this case, the randomization structure could be a split-plot design
instead of a randomized design.

.. note::
    Choosing the randomization structure may seem complicated. It generally, requires
    expert knowledge and some trial-and-error to find the optimal structure. 
    :ref:`Cost-optimal design algorithms <doe_cost>` 
    find the optimal randomization structure for you. No more expert knowledge required!


The design of experiments parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the randomization structure has been chosen, you need to select the corresponding parameters.
Most often, the first parameter is the number of runs. How many tests can we do with the 
provided resources. The other parameters depend on the chosen structure. For example, in 
a split-plot design, we must choose how often we wish to reset the hard-to-change factors.
We must decide how often we change the temperature of the oven.

The optimization
^^^^^^^^^^^^^^^^

Once the factors, model, randomization structure, and corresponding parameters have been chosen,
we can optimize the design. This is most often done using the coordinate-exchange algorithm.
See the :ref:`quickstart` on how to practically generate a design with the package.

.. _doe_cost:

The cost-optimal design
^^^^^^^^^^^^^^^^^^^^^^^

Newcomers and non-statisticians may have difficulty with all of the different subtleties and
required knowledge such as choosing the randomization structure (randomized, split-plot, staggered-level, etc.)
and the corresponding parameters such as the number of runs.

What if for the cake baking experiment the time is a limiting resource. Everything must be
completed in two hours. But resetting the temperature takes time (the transition between
two runs), and the baking time itself is also a factor (associated to the experiment execution).
The optimal number of runs depends on how many times we reset the temperature, and 
on the baking time for each run. It is not clear upfront what the optimal number of runs is.

Cost-optimal designs change the philosophy in design of experiments. The user no longer
specifies the randomization structure or the associated parameters, but rather specifies
a single function which computes the resources of a design. The input of the function 
is a proposed design, and the output, in this example, should be the total time required to 
execute the proposed design. The algorithm automatically searches for an optimal structure
and associated parameters. These structures may even be completely new and tailored to your
problem.

Advantages of using cost-optimal designs:

* Reduces requires expert knowledge, therefore increasing accessibility 
  to the domain of design of experiments.
* Reduces development time. It is no longer necessary to try multiple 
  randomization structures or parameters for a design.
* Increases performance significantly in most non-trivial scenarios
  by more efficiently ordering the runs and determining the ideal trade-off
  between the factor levels and number of runs.

However, while there are many advantages, it also comes at a small price. In case
the scenario is **exactly** defined by a predetermined randomization structure with 
predetermined parameters, fixing the structure generally leads to more optimal
solutions. However, often, the problem is more complex. For example, in our cake 
baking experiment, heating the oven is faster than cooling it down. The oven has 
an active heating element, but no active cooling element. The time to heat the oven 
also depends on the change in temperature. Heating from 180°C to 220°C takes longer
than from 180°C to 200°C. In these scenarios, cost-optimal designs can significantly 
increase the amount of information for the same resources. Or otherwise, can 
gather the same information with less resources.
