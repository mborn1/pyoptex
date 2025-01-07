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
however, care should be taken when doing so for the other attributes and functions.

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

For a :py:class:`MultiRegressionMixin <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin>`,
only three attributes must be set:

* models\_ : A list of 1-D numpy integer arrays, similar to `terms\_` above. The models should be sorted
  by the selection metric, maximum first.
* selection_metrics\_ : The values of the selection metric as a 1-D numpy floating point array. a
  higher selection metric indicates a better model.
* metric_name\_ : The name of the selection metric as a string. Used for interpretation.
