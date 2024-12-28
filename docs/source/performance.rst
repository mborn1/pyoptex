.. _performance:

Performance
===========

.. _perf_cost:

Costs
-----

Cost functions by default provide the user with a 
denormalized design matrix. However, the constant denormalization
of the design can form a computational bottleneck.

If the denormalization is unnecessary, the user can specify `denormalize=False`
in the decorator as follows

>>> @cost_fn(denormalize=False)
>>> def cost(Y)
>>>     # Your cost function here
>>>     return []

In this case, the design is no longer denormalized, but only decoded. This means
that every categorical factor is decoded to a single column with a number ranging from
0 to the number of levels (in the order indicated by the user when specifying the factor).
The continuous factors are normalized between -1 and 1.

However, even the decoding process is not necessary, even when dealing with
categorical factors. When specifying also `decoded=False`, the original encoded
design matrix is passed to the cost function.

>>> @cost_fn(denormalize=False, decoded=False)
>>> def cost(Y)
>>>    # Your cost function here
>>>    return []

There are two attention points when dealing with encoded design matrices.
First, the column index of the factor changes depending on how many preceding
categorical factors there are, and how many levels each categorical factor has.
To easily retrieve the new indices, compute

>>> effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
>>> colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))

The column index in the encoded design for the i\ :sup:`th`\  factor is now `colstart[i]`, 
or `colstart[i]` until `colstart[i+1]` for a categorical factor. 

Second, since the categorical factors are encoded, the exact encoding must be known in
order to use them for cost computations. By default, the factors are encoded using
effect encoding. Another option is to specify manual specify the encoding. More information
on categorical variable encoding in :ref:`cust_cat_encoding`.

Update formulas
---------------

When developing custom metrics for the split\ :sup:`k`\ -plot design
algorithm, make sure to consider developing update formulas.
See :ref:`cust_metric` for more information.

Bayesian variance ratios
------------------------

Try to only use a Bayesian approach as a last resort, and definitely
do not specify too many sets of variance ratios. For each set of
variance ratios, a seperate metric must be computed per evaluation,
making it computationally heavy.

