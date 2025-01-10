.. _a_customization:

Customization
=============

Custom model
------------

See :ref:`cust_model` for more information.

Custom categorical encoding
---------------------------

See :ref:`cust_cat_encoding` for more information.

Custom regressors
-----------------

To create your own regressor, create a new class extending the
:py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`
for single model regressors, and
:py:class:`MultiRegressionMixin <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin>`
for regressors outputting multiple models. These mixins automatically
extend the sklearn interfaces so that you can also use them in their pipelines.

>>> class MyRegressor(RegressionMixin):
>>>     def _fit(self, X, y):
>>>         # Your fit code
>>>         pass
>>> 
>>>     def _predict(self, X):
>>>         # Optional, if you require a custom prediction
>>>         # Defaults to
>>>         return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1) \
>>>                       * self.y_std_ + self.y_mean_


One function must be implemented: 
:py:func:`_fit <pyoptex.analysis.mixins.fit_mixin.RegressionMixin._fit>` which takes
the encoded and normalized model matrix of the data, `X`, and the normalized
outputs, `y`. See the :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`
documentation for more information about which other parameters are available
during fitting and model selection. If desired, the user could overwrite the default 
:py:func:`_predict <pyoptex.analysis.mixins.fit_mixin.RegressionMixin._predict>`,
however, maybe the other attributes and functions must also be updated.

.. note::
    It is important that the constructor of the regressor only sets the variables,
    and not adjust or validate them. Validation and any adjustments should be done 
    during fitting in the 
    :py:func:`_regr_params <pyoptex.analysis.mixins.fit_mixin.RegressionMixin._regr_params>`,
    :py:func:`_compute_derived <pyoptex.analysis.mixins.fit_mixin.RegressionMixin._compute_derived>`
    and :py:func:`_validate_X <pyoptex.analysis.mixins.fit_mixin.RegressionMixin._validate_X>`.

During fitting, certain attributes must be set. For
:py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`,
the user must set (all parameters are also indicated as the attributes of the mixin):

* terms\_ : a 1-D numpy integer array with the indices of the selected terms in the
  encoded model matrix.
* coef\_ : the coefficients of the terms for a normalized model matrix.
* scale\_ : the scale (or variance of the random errors) of the data.
* vcomp\_ : a 1-D numpy floating point arry with the estimated variance components in a mixed model.
* fit\_ : optionally the fit object, returned by `fit_fn\_`. If specified, the used can call
  `.summary()` which is forwarded here.

Have a look at the source code of 
:py:class:`SimpleRegressor <pyoptex.analysis.estimators.simple_model.SamsRegressor>` for a
simple example.

For a :py:class:`MultiRegressionMixin <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin>`,
only three attributes must be set:

* models\_ : A list of 1-D numpy integer arrays, similar to `terms\_` above. The models should be sorted
  by the selection metric, maximum first.
* selection_metrics\_ : The values of the selection metric as a 1-D numpy floating point array. a
  higher selection metric indicates a better model.
* metric_name\_ : The name of the selection metric as a string. Used for interpretation.

.. _a_cust_sams:

Simulated annealing model selection (SAMS)
------------------------------------------

Simulated annealing model selection, or SAMS, was devised by
`Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_.
It is a model selection algorithm, which instead of looking at the statistical
significance, like is most commonly used, simulates multiple models and looks at what the
good fitting models have in common. The algorithm works in three stages:

* The simulation stage: here the algorithm simulates many models of a fixed size using simulated annealing,
  and sorts them by their
  :math:`R^2`. Commonly it simulates 10000 or 20000 models, however it depends on the
  problem at hand.
* The reduction stage: here the algorithm takes the simulated models and looks what the
  most common 1-factor, 2-factor, 3-factor, etc. combinations are. In other words, it looks
  at which submodel of size k occurs most frequently in the good fitting models for multiple 
  values of k.
* The selection stage: here the algorithm takes the most occuring submodels of each size and
  compares them to determine an ordering. The ordering is based on the entropy which is explained
  later.

As you may notice, the algorithm does not output just one model. It outputs multiple models,
ordered by which model it has the most confidence in. The last two stages of the algorithm
use the result of the first stage to automatically determine an ordering, however, the user
may also manually look at a raster plot of the results which looks as follows:

.. figure:: /assets/img/raster_plot.png
  :width: 100%
  :alt: The raster plot. Every row is a model, every column is a term. The color indicates the coefficient magnitude.
  :align: center

Each row is a model, each column is a potential term in the model, and the color indicates the
coefficient of the term. This means that any term not in the model has a coefficient of zero, which is
plotted in white. By looking at largely colored columns, we can determine which
submodels occur most often (here :math:`x_1`, :math:`x_3` and :math:`x_7`).

.. note::
  In some events, multiple distinct models may perform equally well. Such a scenario is
  difficult to detect in the raster plot, and also by the entropy criterion. Luckily, we
  can also cluster the results in the raster plot making them more visible. The different
  terms in each model are binary encoded if the effect is present or not. On this representation,
  a kmeans clustering is run. This technique was also devised by
  `Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_.

See :py:class:`SamsRegressor <pyoptex.analysis.estimators.sams.estimator.SamsRegressor>` for information
on the parameters.

.. _a_cust_sams_entropy:

Entropy calculations
^^^^^^^^^^^^^^^^^^^^

The entropy is the most effective addition of the algorithm to perform automated model
selection. The entropy is computed as

.. math::

    e = f_{o} * log_2(f_{o} / f_{t}) + (1 - f_{o}) * log_2((1 - f_{o}) / (1 - f_{t}))

where :math:`f_{o}` is the observed frequency of the submodel in the simulation phase, and
:math:`f_{t}` is the theoretical frequency this submodel would occur when randomly sampling
hereditary models.

In `Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_,
the authors performed some simulations on screening designs for different model selection algorithms.
The oracle method requires prior knowledge about the true model, and each term is tested for significance.
The AICc method is Akaike's Information Criterion (corrected). The authors noted that
a search through the hereditary models was performed, from which the best according to the AICc was
selected. This, together with the Bayes Information Criterion (BIC) is commonly applied in practice.
The last method is the new SAMS method with entropy selection.

.. list-table:: Part of the simulations results from Wolters and Bingham (2012)
  :align: center
  :widths: 1 1 1 1 1

  * - Method 
    - Correct
    - Underfitted
    - Overfitted
    - (Partialy) Wrong
  * - Oracle
    - 62.8
    - 37.2
    - 0
    - 0
  * - AICc
    - 7.2
    - 0.7
    - 53.8
    - 38.3
  * - SAMS
    - 43.3
    - 16.2
    - 15.8
    - 24.7

The SAMS method with entropy significantly outperforms any other method with 43.3% of models
found to be correct. In addition, the oracle method, which has prior knowledge about the true
model, also only found 62.8% of the models. AICc only found about 7.2% of the models making it 
not very suitable for this scenario.
  
.. _samplers_sams:

There is one downside to the entropy criterion. Only in the specific case where the model
is a (partial) response surface model with weak heredity can :math:`f_{t}`
be computed exactly. To make sure the algorithm is generic enough, a fallback was implemented
to compute an approximation of the entropy using a model sampler. Three different
samplers are implemented:
:py:func:`sample_model_dep_onebyone <pyoptex.utils.model.sample_model_dep_onebyone>`,
:py:func:`sample_model_dep_mcmc <pyoptex.analysis.estimators.models.model.sample_model_dep_mcmc>`
and :py:func:`sample_model_dep_random <pyoptex.utils.model.sample_model_dep_random>`.

For each of these samplers, we ran similar simulations to
`Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_.
We start from a PB12 design (Plackett-Burman). Next, we generate a random hereditary model
by sampling 1 to 4 main effects, :math:`n_{main}`, and sequentially sampling :math:`4 - n_{main}`
interaction effects. Note that this is a weak heredity submodel of a partial response surface
design where each factor has linear effects and two-factor interactions. 

The results are

.. list-table:: Simulations of different entropy approximations
  :align: center
  :widths: 1 1 1 1 1

  * - Method 
    - Correct
    - Underfitted
    - Overfitted
    - (Partialy) Wrong
  * - Exact entropy
    - 43.7
    - 30.3
    - 10.5
    - 15.5
  * - One-by-one
    - 37.3
    - 12.6
    - 23.8
    - 26.3
  * - Markov-chain Monte carlo (mcmc)
    - 38.8
    - 12.3
    - 23.3
    - 25.6
  * - Random 
    - 36.8
    - 10.1
    - 26.1
    - 27.0

The first row is the exact entropy method as used in
`Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_.
Note that all three samplers, even though they perform worse than the exact entropy based on the percentage
of correct models, still perform significantly better than AICc. When loosening the classification by
also classifying models underfitted or overfitted by one term as correct, the exact entropy method
has 61.1% accuracy, the one-by-one has 59.1%, the mcmc has 59.3%, and the random has 56.5%. Only
a 2% difference for the one-by-one and mcmc samplers.

By default, the one-by-one 
sampler is used as it performs almost equally as good as the mcmc method, but computes faster.

.. _warning_sams:

.. warning::
  The implementation of SAMS uses the samplers by default, however, the exact method
  may be used by specifying the `entropy_model_order` parameter in
  :py:class:`SamsRegressor <pyoptex.analysis.estimators.sams.estimator.SamsRegressor>`.
  However, a large warning should be given to this parameter as it comes with certain
  assertions (which are covered in many, but not all scenarios).

  First, the heredity mode must be 'weak', otherwise the sampling method is still
  applied. Second, the model must be generated using
  :py:func:`partial_rsm_names <pyoptex.utils.model.partial_rsm_names>` followed by
  :py:func:`model2Y2X <pyopytex.utils.model.model2Y2X>`. Third, the factors must
  be ordered: first all factors which can have a quadratic effect, second 
  all factors which can not have quadratic effects, but can have two-factor interactions,
  and third all factors which can only have a main effect. Finally, the dependency
  matrix must be generated using
  :py:func:`order_dependencies <pyoptex.utils.model.order_dependencies>`.

  As an example. Create three factors

  >>> factors = [
  >>>   Factor('A'), Factor('B'), Factor('C')
  >>> ]

  Next, create the model orders. The order of the factor names in the dictionary
  **must** be the same as those in the list of factors. They also must be
  ordered `quad` - `tfi` - `lin`.

  >>> entropy_model_order = {'A': 'quad', 'B': 'tfi', 'C': 'lin'}
  
  Create the model using :py:func:`partial_rsm_names <pyoptex.utils.model.partial_rsm_names>`.
  Note that the `quad` elements are first, then the `tfi`, and finally the `lin` elements.
  The dictionary parameters **must** be in the same order as the factors.

  >>> model = partial_rsm_names(entropy_model_order)
  >>> Y2X = model2Y2X(model, factors)

  Next, create the dependencies from the model

  >>> dep = order_dependencies(model, factors)

  Finally, we can fit SAMS using the exact entropy formula

  >>> regr = SamsRegressor(
  >>>     factors, Y2X, 
  >>>     mode='weak', dependencies=dep,
  >>>     forced_model=np.array([0], np.int\_),
  >>>     entropy_model_order=entropy_model_order
  >>> )

  

  
