Welcome to PyOptEx documentation!
===================================

**Date**: |date|

**Version**: |release|

.. role:: raw-html(raw)
   :format: html

PyOptEx (or Python Optimal Experiments) is a package designed to create
optimal designs for experiments with Python. The package designed to be
as intuitive as possible, making it accessible also to non-statisticians. 
The package is fully open source and can be used for any purpose.

The package is designed for both engineers, and design of experiment
researchers. Engineers can use the precreated functions
to generate designs for their problems. Researchers can easily develop
new metrics (criteria) and test them.

To generate experimental designs, there are two main options:

* :raw-html:`<strong>Split<sup>k</sup>-plot</strong>`: These designs
  can be used to create fully randomized
  designs with a fixed number of runs, or a split-plot design with any number
  of strata of hard-to-change factors. The optimization algorithm is the 
  coordinate-exchange algorithm and it uses the update formulas described in
  `Born and Goos (2025) <https://www.sciencedirect.com/science/article/pii/S0167947324001129>`_.
* :raw-html:`Cost-optimal designs`: These designs can be used to optimize
  for a given cost function. Instead of specifying the number of runs, the
  user specifies a cost function and a maximum budget and the algorithm
  decides automatically on the number of runs, randomization structure, etc.

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
