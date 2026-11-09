"""
Module for the SAMS regressor.
"""

import math
import warnings
from functools import reduce

import numpy as np
import plotly.graph_objects as go
import ruptures as rpt
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from tqdm import tqdm

from ....utils.comp import timeout
from ....utils.design import obs_var_from_Zs
from ....utils.model import _term_groups, identityY2X, model2encnames, sample_model_dep_onebyone
from ...mixins.fit_mixin import MultiRegressionMixin
from .accept import ExponentialAccept
from .bnb.sams_bnb import SamsBnB
from .entropy import entropies, entropies_approx
from .models.mixed_lm_model import MixedLMModel
from .models.ols_model import OlsModel
from .plot import plot_raster
from .simulation import simulate_all, simulate_sams


class SamsRegressor(MultiRegressionMixin):
    """
    Regression model selection using the SAMS procedure. This implements
    the :py:class:`MultiRegressionMixin <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin>`,
    and outputs multiple good fitting models. SAMS was originally divised in
    `Wolters and Bingham (2012) <https://www.tandfonline.com/doi/abs/10.1198/TECH.2011.08157>`_
    and adapted here to include any dependency matrix and random effects.

    .. note::
        It also includes all parameters and attributes from
        :py:class:`MultiRegressionMixin <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin>`

    .. note::
        A more detailed guide on SAMS can be found at :ref:`a_cust_sams`.

    .. note::
        SAMS samples models as term groups: the columns of one model term,
        such as the dummy columns of a categorical main effect or interaction,
        are added and removed together, and `model_size` counts term groups.
        The groups are derived from a `Y2X` created with
        :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`. For any other
        `Y2X`, every column is its own group, and a categorical factor raises a
        warning that its dummy columns are sampled separately.

    Attributes
    ----------
    dependencies : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    mode : None or 'weak' or 'strong'
        The heredity mode during sampling.
    forced_model : np.array(1d)
        Any terms that must be included in the model. Commonly np.array([0], dtype=np.int)
        is used to force the intercept when the intercept is the first column in
        the normalized, encoded model matrix. This model must itself fulfill the
        heredity constraints.
    model_size : int
        The number of term groups in the overfitted models. Defaults to the
        number of term groups that gives about N / 3 columns (N the number of
        runs) at the average size of the non-forced groups: at least three more
        than the forced term groups, at most as many as fit in fewer than
        N columns when the largest groups are taken, and at most the forced
        groups plus half of the free groups (rounded up), where the number of
        different models is largest. The overfitted model includes the forced
        term groups, and its size must thus be larger than their number.
    nb_models : int or 'all'
        The number of unique models to accept during the sams procedure.
        If 'all' then all possible models are fitted.
    skipn : int, float or 'auto'
        The number of worst-fitting models to skip during
        branch-and-bound and entropy calculations. When specified as a float,
        it must be a number between 0 and 1 to indicate a fraction.
        Defaults to 'auto' which uses the elbow method and an additional 1% safety
        margin. Any int must be smaller than `nb_models`.
    est_ratios : None or np.array(1d)
        The estimated variance ratios to be used during SAMS. These
        ratios are used to make the SAMS procedure computationally feasible
        in mixed models. For every random effect, a ratio should be provided.
        Defaults to 1 for each random effect if None is specified.
    allow_duplicate_sample : bool
        Whether or not to allow duplicate samples to be stored in the final results of
        the SAMS sampling procedure.
    max_cluster : int
        The maximum number of clusters to try when specifying 'auto'
        as `ncluster`. Atleast three are required for the elbow method. This
        number is inclusive.
    ncluster : None or int or 'auto'
        The number of clusters to fit on the raster plot using the Hamming
        distance. If None, no kmeans clustering is performed. If 'auto',
        every number of clusters between 1 and `max_cluster` is tried and
        the best is selected using the elbow method.
    topn_bnb : int
        The number of top submodels for a fixed size to retrieve for entropy
        calculations.
    nterms_bnb : None or int or iterable(int)
        The fixed sizes of submodels to apply the branch-and-bound algorithm
        on, in term groups including the forced term groups. If None, every size
        from one more than the forced term groups to `model_size` - 2 (inclusive)
        is tested as recommended by the original paper. If an int, every size from
        one more than the forced term groups until the specified number (exclusive)
        is tested. If an iterable, only the values from the iterable are tested.
    bnb_timeout : int
        The maximum number of seconds to run the branch-and-bound algorithm for.
        Clear submodels in the raster plot will not require much time in
        the branch-and-bound algorithm. Therefore, if the branch-and-bound
        algorithm would require too much time, most likely low entropy models
        are the result and the computation can be halted prematurely. Defaults
        to three minutes.
    entropy_sampler : func(dep, model_size, N, forced, mode)
        The sampler to use when generating random hereditary models. It works in
        term groups: `dep` is the (G, G) dependency matrix between the G term groups,
        `model_size` is in term groups and `forced` holds term group indices. See the
        documentation on customizing SAMS for an indication on which sampler to use.
    entropy_sampling_N : int
        The number of random samples to draw using the sampler to compute
        the theoretical frequencies of the submodels.
    entropy_model_order : dict(str: ('lin' or 'tfi' or 'quad'))
        The order of the terms in the model. Please read the warning in
        the documentation on customizing SAMS.
    tqdm : bool
        Whether to use tqdm to track the progress
    sams_model\\_ : :py:class:`Model <pyoptex.analysis.estimators.sams.models.model.Model>`
        A SAMS model used in sampling and fitting data during the SAMS procedure.
    results\\_ : np.array(1d)
        A numpy array with a special datatype where each element contains
        'model' (np.int64, size `model_size`): the term group indices of the model
        (see `sams_model_.term_groups`), 'coeff' (np.float64, one per encoded column):
        the coefficients, NaN for columns not in the model, and 'metric' (np.float64).
        Results contains `nb_models` elements.
        These are the returned models from the SAMS procedure.
    models\\_ : list(np.array(1d))
        The list of models as column indices, ordered by entropy.
    entropies\\_ : np.array(1d)
        The entropy of each exported model in `models\\_`. In
        case of multiple clusters, the entropies are calculated
        within the respective cluster.
    selection_metrics\\_ : np.array(1d)
        Alias for `entropies\\_`.
    frequencies\\_ : np.array(1d)
        The occurence frequency of each submodel in `models\\_`
    kmeans\\_ : None or :py:class:`sklearn.cluster.Kmeans`
        A kmeans object used to cluster the raster plot. Added
        a parameter `skips` equal to 5% of the cluster size to
        be skipped for entropy calculations.
    metric_name\\_ : str
        The name of the selection metric.
    """

    def __init__(
        self,
        factors=(),
        Y2X=identityY2X,
        random_effects=(),
        dependencies=None,
        mode=None,
        forced_model=None,
        model_size=None,
        nb_models=10000,
        skipn="auto",
        est_ratios=None,
        allow_duplicate_sample=False,
        max_cluster=8,
        ncluster=None,
        topn_bnb=4,
        nterms_bnb=None,
        bnb_timeout=180,
        entropy_sampler=sample_model_dep_onebyone,
        entropy_sampling_N=10000,
        entropy_model_order=None,
        tqdm=True,
    ):
        """
        Initializes the class

        Parameters
        ----------
        factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
            A list of factors to be used during fitting. It contains
            the categorical encoding, continuous normalization, etc.
        Y2X : func(Y)
            The function to transform a design matrix Y to a model matrix X.
        random_effects : list(str)
            The names of any random effect columns. Every random effect
            is interpreted as a string column and encoded using
            effect encoding.
        dependencies : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.
        mode : None or 'weak' or 'strong'
            The heredity mode during sampling.
        forced_model : np.array(1d)
            Any terms that must be included in the model. Commonly np.array([0], dtype=np.int64)
            is used to force the intercept when the intercept is the first column in
            the normalized, encoded model matrix. This model must itself fulfill the
            heredity constraints.
        model_size : int
            The number of term groups in the overfitted models. Defaults to the
            number of term groups that gives about N / 3 columns (N the number of
            runs) at the average size of the non-forced groups: at least three more
            than the forced term groups, at most as many as fit in fewer than
            N columns when the largest groups are taken, and at most the forced
            groups plus half of the free groups (rounded up), where the number of
            different models is largest. The overfitted model includes the forced
            term groups, and its size must thus be larger than their number.
        nb_models : int or 'all'
            Th number of unique models to accept during the sams procedure.
            If 'all', then all possible models will be fitted.
        skipn : int, float or 'auto'
            The number of worst-fitting models to skip during
            branch-and-bound and entropy calculations. When specified as a float,
            it must be a number between 0 and 1 to indicate a fraction.
            Defaults to 'auto' which uses the elbow method and an additional 1% safety
            margin. Any int must be smaller than `nb_models`.
        est_ratios : None or np.array(1d)
            The estimated variance ratios to be used during SAMS. These
            ratios are used to make the SAMS procedure computationally feasible
            in mixed models. For every random effect, a ratio should be provided.
            Defaults to 1 for each random effect if None is specified.
        allow_duplicate_sample : bool
            Whether or not to allow duplicate samples to be stored in the final results of
            the SAMS sampling procedure.
        max_cluster : int
            The maximum number of clusters to try when specifying 'auto'
            as `ncluster`. Atleast three are required for the elbow method. This
            number is inclusive.
        ncluster : None or int or 'auto'
            The number of clusters to fit on the raster plot using the Hamming
            distance. If None, no kmeans clustering is performed. If 'auto',
            every number of clusters between 1 and `max_cluster` is tried and
            the best is selected using the elbow method.
        topn_bnb : int
            The number of top submodels for a fixed size to retrieve for entropy
            calculations.
        nterms_bnb : None or int or iterable(int)
            The fixed sizes of submodels to apply the branch-and-bound algorithm
            on, in term groups including the forced term groups. If None, every size
            from one more than the forced term groups to `model_size` - 2 (inclusive)
            is tested as recommended by the original paper. If an int, every size from
            one more than the forced term groups until the specified number (exclusive)
            is tested. If an iterable, only the values from the iterable are tested.
        bnb_timeout : int
            The maximum number of seconds to run the branch-and-bound algorithm for.
            Clear submodels in the raster plot will not require much time in
            the branch-and-bound algorithm. Therefore, if the branch-and-bound
            algorithm would require too much time, most likely low entropy models
            are the result and the computation can be halted prematurely. Defaults
            to three minutes.
        entropy_sampler : func(dep, model_size, N, forced, mode)
            The sampler to use when generating random hereditary models. It works in
            term groups: `dep` is the (G, G) dependency matrix between the G term groups,
            `model_size` is in term groups and `forced` holds term group indices. See the
            documentation on customizing SAMS for an indication on which sampler to use.
        entropy_sampling_N : int
            The number of random samples to draw using the sampler to compute
            the theoretical frequencies of the submodels.
        entropy_model_order : dict(str: ('lin' or 'tfi' or 'quad'))
            The order of the terms in the model. Please read the warning in
            the documentation on customizing SAMS.
        tqdm : bool
            Whether to use tqdm to track the progress
        """
        super().__init__(factors, Y2X, random_effects)

        # Store variables
        self.dependencies = dependencies
        self.mode = mode
        self.model_size = model_size
        self.nb_models = nb_models
        self.skipn = skipn
        self.est_ratios = est_ratios
        self.allow_duplicate_sample = allow_duplicate_sample
        self.topn_bnb = topn_bnb
        self.nterms_bnb = nterms_bnb
        self.bnb_timeout = bnb_timeout
        self.forced_model = forced_model
        self.max_cluster = max_cluster
        self.ncluster = ncluster
        self.entropy_sampler = entropy_sampler
        self.entropy_sampling_N = entropy_sampling_N
        self.entropy_model_order = entropy_model_order
        self.tqdm = tqdm

    def _regr_params(self, X, y):
        """
        Callback function to dynamically alter the inputted variables.
        By default simply references them.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with the original data.
        y : pd.Series or np.array(1d)
            The output variable.
        """
        super()._regr_params(X, y)

        # Set some default values (the model size and BnB sizes are set in _fit, once the term groups are known)
        self._est_ratios = np.ones(len(self._re)) if len(self._re) > 0 and self.est_ratios is None else self.est_ratios

    def _validate_fit(self, X, y):
        """
        Validate the inputted parameters before fitting the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data.
        y : pd.Series or np.array(1d)
            The output variable.
        """
        super()._validate_fit(X, y)

        # Validate dependencies and mode
        assert self.mode in (None, "weak", "strong"), "The drop-mode must be None, weak or strong"
        if self.mode in ("weak", "strong"):
            assert self.dependencies is not None, "Must specify dependency matrix if using weak or strong heredity"
            assert len(self.dependencies.shape) == 2, "Dependencies must be a 2D array"
            assert self.dependencies.shape[0] == self.dependencies.shape[1], "Dependency matrix must be square"

        # TODO: validate forced_model hereditary

        # Validate SAMS inputs (the forced model is compared in term groups, in _fit)
        if self.model_size is not None and self.forced_model is None:
            assert self.model_size > 0, "The overfitted model size must be a positive number"
        assert self.nb_models == "all" or self.nb_models > 0, (
            'Must have at least one model to simulate, nb_models must be larger than zero or "all"'
        )
        assert self.skipn == "auto" or isinstance(self.skipn, (int, float)), (
            'Skipn must be "auto" or an integer or a float'
        )
        if isinstance(self.skipn, int) and self.nb_models != "all":
            assert 0 <= self.skipn < self.nb_models, "Cannot skip all SAMS models, skipn must be smaller than nb_models"
        if isinstance(self.skipn, float):
            assert 0 <= self.skipn < 1, "If skipn is a floating point, it must be a fraction between [0, 1)"
        if self.est_ratios is not None:
            assert len(self.est_ratios) == len(self._re), (
                "Every random effect must have an estimated ratio when specified, in the same order"
            )
        assert self.topn_bnb > 0, (
            "Must select at least one submodel for each fixed size, topn_bnb must be larger than 0"
        )
        if self.nterms_bnb is not None and isinstance(self.nterms_bnb, int):
            assert self.nterms_bnb > 0, "When an integer is specified for nterms_bnb, it must be larger than zero"
        assert self.bnb_timeout > 0, (
            "Must specify a strictly positive number of seconds for the branch-and-bound to run"
        )
        if self.forced_model is not None:
            assert isinstance(self.forced_model, np.ndarray), "The forced model must be an integer array"
            assert np.issubdtype(self.forced_model.dtype, np.integer), "The forced model must be an integer array"
        assert self.max_cluster >= 3, "The maximum number of clusters for auto selection must be larger than three"
        if self.ncluster is not None:
            assert self.ncluster == "auto" or isinstance(self.ncluster, (int, np.integer)), (
                'ncluster must be None, "auto" or an integer'
            )
            if self.ncluster != "auto":
                assert self.ncluster > 0, "The number of clusters must be larger than or equal to one"

        # Validate model order for entropy calculations
        if self.entropy_model_order is not None:
            # Define the required ordering
            model_types_ord = {"quad": 2, "tfi": 1, "lin": 0}

            # Reorder the model types based on the factors
            assert all(str(f.name) in self.entropy_model_order for f in self._factors), (
                "All factors must have an entropy model order specified"
            )
            entropy_model_order = {str(f.name): self.entropy_model_order[str(f.name)] for f in self._factors}

            # Assert the ordering
            assert np.all(np.diff([model_types_ord[typ] for typ in entropy_model_order.values()]) <= 0), (
                "Model types must be ordered quad > tfi > lin for entropy calculations"
            )

            # TODO: Perform a random validation of the model to make sure it is correct in Y2X

    def _topn_selection(self, results, sizes, nterms, topn=4, timeout_sec=180):
        """
        Selects the top n submodels of fixed sizes in the results using
        the branch-and-bound algorithm. For each size in `sizes`, the
        `topn` submodels of that size are extracted from the results.

        Parameters
        ----------
        results : np.array(2d)
            The sampled models as term group indices, one row per model
            (the 'model' field of `results_`).
        sizes : iterable(int)
            An iterable of ints with the fixed sizes of the submodels.
        nterms : int
            The number of term groups (`sams_model_.n_term_groups`).
        topn : int
            The number of submodels with each size to extract.
        timeout_sec : int
            The maximum time to use for a single iteration of the branch-and-bound
            algorithm. If it runs too long, chances are low that any high
            entropy models are returned.

        Returns
        -------
        models : list(np.array(1d))
            The list of top submodels of each size as an integer array.
        frequencies : np.array(1d)
            The frequency of each submodel in the results.
        timed_out : bool
            Whether a size hit the timeout, which stops the remaining sizes.
        """
        # Initialize results
        models = list()
        counts = np.zeros(len(sizes) * topn, dtype=np.int64)
        count_idx = 0
        timed_out = False

        # Compute BnB
        for _i, size in tqdm(enumerate(sizes), total=len(sizes), disable=(not self.tqdm)):
            # Compute the result with a timeout
            result = timeout(
                SamsBnB(size, results, nterms, self.mode, self.sams_model_.dep, self.sams_model_.forced).top,
                topn,
                timeout=timeout_sec,
            )

            # Check for a result within the timeout
            if result is not None:
                m, f = result
                for idx in range(len(m)):
                    if f[idx] > 0:
                        models.append(m[idx][0])
                        counts[count_idx] = f[idx]
                        count_idx += 1
            else:
                timed_out = True
                break

        # Compute frequencies
        frequencies = counts[: len(models)] / len(results)

        return models, frequencies, timed_out

    def _unique_submodels(self, models, nterms):
        """
        Extracts the unique models from the resulting set.
        If a model :math:`A + B + A*B` has the highest entropy,
        any model :math:`A`, :math:`B`, :math:`A + B`, :math:`A + A*B`,
        or :math:`B + A*B`, with a lower entropy, is ignored.
        In other words, any model with a lower entropy, completely present
        in any higher entropy model is discarded.

        Parameters
        ----------
        models : list(np.array(1d))
            The list of models, as term group indices.
        nterms : int
            The number of term groups (`sams_model_.n_term_groups`).

        Returns
        -------
        keep : np.array(1d)
            A boolean array whether the model should be kept or not.
        """

        # Create submodels array
        submodels = np.zeros((len(models), nterms), dtype=np.bool_)
        submodels[0, models[0]] = True
        j = 1

        # Initialize return mask
        keep = np.ones(len(models), dtype=np.bool_)

        for i in range(1, len(models)):
            # If submodel present in higher models
            if np.any(np.all(submodels[:j, models[i]], axis=1)):
                # Drop it
                keep[i] = False
            else:
                # Add it to the good submodels
                submodels[j, models[i]] = True
                j += 1

        return keep

    def _entropy(self, X, y, submodels, freqs):
        """
        Computes the entropy of the different submodels with their frequencies.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        submodels : list(np.array(1d))
            The list of top submodels for each size.
        freq : np.array(1d)
            The frequencies of these submodels in the raster plot.

        Returns
        -------
        entropies : np.array(1d)
            The entropies of these models.
        """
        if self.entropy_model_order is None or self.mode != "weak":
            # Compute approximated entropies (the forced groups are an empty array when nothing is forced)
            entropy = entropies_approx(
                submodels,
                freqs,
                self._model_size,
                self.sams_model_.dep,  # submodels are in group space
                self.mode,
                self.sams_model_.forced,
                self.entropy_sampling_N,
                self.entropy_sampler,
            )

        else:
            # Count the factors of each order as the entropy formula sees them: a categorical factor counts once
            # when its main-effect columns (right after the intercept in the partial response surface model)
            # form one term group, and once per dummy column otherwise
            model_counts = {e: 0 for e in ("quad", "tfi", "lin")}
            start = 1
            for f, et in zip(self._factors, self.effect_types_, strict=True):
                ncols = 1 if et == 1 else et - 1
                one_group = len(np.unique(self.sams_model_.labels[start : start + ncols])) == 1
                model_counts[self.entropy_model_order[str(f.name)]] += 1 if one_group else ncols
                start += ncols
            model_counts = [model_counts.get(e, 0) for e in ("quad", "tfi", "lin")]

            # Compute entropy (based on model order)
            entropy = entropies(submodels, freqs, self._model_size, model_counts)

        return entropy

    def _too_small_message(self, n_unique=None):
        """
        The reason for too few models, naming the number of unique models.

        Parameters
        ----------
        n_unique : None or int
            The number of unique models. If None, the unique sampled models are counted.

        Returns
        -------
        message : str
            The message.
        """
        if n_unique is None:
            n_unique = len(np.unique(self.results_["model"], axis=0))
        n_forced = len(self.sams_model_.forced)
        n_half = n_forced + math.ceil((self.sams_model_.n_term_groups - n_forced) / 2)
        return (
            f"the number of unique models of {self._model_size} term groups is {n_unique}, so the model space is "
            "too small for SAMS. The number of possible models is largest at about half of the free term groups: "
            f"use a model_size near {n_half}, a smaller skipn or a smaller ncluster, or fit the models directly."
        )

    def _fit(self, X, y):
        """
        Internal fit function for the SAMS regressor.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        # Some final validation
        if self.forced_model is not None:
            assert np.all(self.forced_model < self.n_encoded_features_), (
                "The forced model must have integers smaller than the number of parameters in X"
            )
        if self.mode is not None:
            assert self.dependencies.shape[0] == X.shape[1], "Must specify a dependency for each term"

        # Term groups from the model of Y2X: the columns of a term (e.g. the dummy columns of a categorical factor)
        labels = _term_groups(self._Y2X, self._factors)
        categorical = [str(f.name) for f in self._factors if not f.is_continuous]
        if labels is None and categorical:
            warnings.warn(
                f"The dummy columns of the categorical factors {categorical} are sampled separately, because Y2X "
                "was not built with model2Y2X, so its term groups are unknown",
                UserWarning,
                stacklevel=3,
            )

        # Compute SAMS results (the dependencies are only used with heredity)
        dep = self.dependencies if self.mode is not None else None
        kwargs = dict(forced=self.forced_model, mode=self.mode, dep=dep, labels=labels)
        if len(self._re) == 0:
            self.sams_model_ = OlsModel(X, y, **kwargs)
        else:
            V = obs_var_from_Zs(self.Zs_, len(X), self._est_ratios)
            self.sams_model_ = MixedLMModel(X, y, V=V, **kwargs)

        # Model size in term groups: by default about N / 3 columns at the average size of the non-forced groups
        if self.model_size is None:
            n_forced = len(self.sams_model_.forced)
            sizes = np.array([len(g) for g in self.sams_model_.term_groups])
            free = np.delete(sizes, self.sams_model_.forced)
            n_free = round((len(X) / 3 - np.sum(sizes[self.sams_model_.forced])) / free.mean()) if free.size else 0
            self._model_size = n_forced + n_free

            # At most as many groups as check_size allows: the largest groups stay below N columns
            cap = int(np.sum(np.cumsum(np.sort(sizes)[::-1]) < len(X)))
            self._model_size = min(self._model_size, cap)

            # At most the forced groups plus half of the free groups, where the number of different models is largest
            self._model_size = min(self._model_size, n_forced + math.ceil(free.size / 2))

            if n_forced + 3 > cap:
                raise ValueError(
                    f"The default model_size is at least {n_forced + 3} term groups (three more than the {n_forced} "
                    f"forced), but the largest {n_forced + 3} term groups do not fit in fewer than the {len(X)} runs, "
                    "so the model space is too small for SAMS with these runs. Set model_size and nterms_bnb, "
                    "or fit the models directly."
                )

            # At least three more than the forced groups, which leaves a branch-and-bound size, within the groups
            self._model_size = min(max(self._model_size, n_forced + 3), self.sams_model_.n_term_groups)
        else:
            self._model_size = self.model_size
        if self._model_size == self.sams_model_.n_term_groups:
            raise ValueError(
                f"model_size={self._model_size} includes all term groups: {self._too_small_message(n_unique=1)}"
            )

        # BnB submodel sizes are in term groups, starting after the forced groups
        nterms_bnb = self._model_size - 1 if self.nterms_bnb is None else self.nterms_bnb
        self._nterms_bnb = (
            range(len(self.sams_model_.forced) + 1, nterms_bnb) if isinstance(nterms_bnb, int) else nterms_bnb
        )

        accept = ExponentialAccept(T0=(X.shape[0]) * np.var(y) / 10, rho=0.95, kappa=4)
        if self.nb_models == "all":
            self.results_ = simulate_all(self.sams_model_, self._model_size, tqdm=self.tqdm)
        else:
            self.results_ = simulate_sams(
                self.sams_model_,
                self._model_size,
                accept_fn=accept,
                nb_models=self.nb_models,
                allow_duplicate=self.allow_duplicate_sample,
                tqdm=self.tqdm,
            )

        # Sort the results
        idx = np.argsort(self.results_["metric"])
        results = self.results_[idx]

        # Cannot skip all models
        if isinstance(self.skipn, int) and self.skipn >= len(results):
            raise ValueError(
                f"Cannot skip skipn={self.skipn} of the {len(results)} sampled models: {self._too_small_message()}"
            )

        # Skip bad part of the data (the change point detection needs at least three models)
        if self.skipn == "auto" and len(results) < 3:
            self._skipn = 0
        elif self.skipn == "auto":
            # Compute the difference in derivative
            slope = np.diff(results["metric"])
            bkps = rpt.KernelCPD(kernel="linear", min_size=0).fit_predict(slope, pen=np.var(slope) * 1000)

            # Extract the skip
            if len(bkps) == 1:
                self._skipn = 0
            else:
                # Take the last breakpoint
                self._skipn = bkps[-2] + int(0.01 * (len(results) - bkps[-2]))  # Add a safety margin for steady state
        elif isinstance(self.skipn, float):
            self._skipn = int(self.skipn * len(results))
        else:
            self._skipn = self.skipn
        results = results[self._skipn :]
        if len(results) == 0:
            raise ValueError(f"No models remain after skipping {self._skipn}: {self._too_small_message()}")
        if isinstance(self.ncluster, (int, np.integer)) and len(results) < self.ncluster:
            raise ValueError(
                f"Only {len(results)} models remain after skipping {self._skipn}, fewer than "
                f"ncluster={self.ncluster}: {self._too_small_message()}"
            )

        # Possibly cluster
        if self.ncluster is None:
            # Perform branch and bound
            submodels, freq, timed_out = self._topn_selection(
                results["model"], self._nterms_bnb, self.sams_model_.n_term_groups, self.topn_bnb, self.bnb_timeout
            )

            # Compute entropies
            self.entropies_ = self._entropy(X, y, submodels, freq)

            # Set default value
            self.kmeans_ = None

        else:
            # Expand models to integer cluster (for kmeans)
            results_cluster = np.zeros((len(results), self.sams_model_.n_term_groups))
            np.put_along_axis(results_cluster, results["model"], 1, axis=1)

            # Auto detect the number of clusters
            if self.ncluster == "auto":
                # Fit kmeans (at most one cluster per distinct model)
                max_cluster = min(self.max_cluster, len(np.unique(results["model"], axis=0)))
                kmeans_dists = np.zeros(max_cluster)
                for nc in range(1, max_cluster + 1):
                    kmeans = KMeans(n_init="auto", n_clusters=nc).fit(results_cluster)
                    kmeans_dists[nc - 1] = kmeans.inertia_

                # Elbow rule for cluster selection, which needs at least three values
                if max_cluster < 3:
                    ncluster = 1
                else:
                    kmeans_elbow = np.diff(kmeans_dists, n=2)
                    ncluster = np.argmax(kmeans_elbow) + 2
            else:
                # Force the number of clusters
                ncluster = self.ncluster
                kmeans_dists = None

            # Fit kmeans on selected number of clusters
            self.kmeans_ = KMeans(n_init="auto", n_clusters=ncluster).fit(results_cluster)
            self.kmeans_.skips = np.zeros(ncluster, dtype=np.int64)
            self.kmeans_.dists = kmeans_dists

            # Perform model select on each cluster
            m_, f_, e_ = [], [], []
            timed_out = False
            for i in range(0, ncluster):
                # Select cluster i
                cluster_i = self.kmeans_.labels_ == i
                ncluster_i = np.sum(cluster_i)

                # Compute skip
                skipn = int(0.05 * ncluster_i)
                self.kmeans_.skips[i] = skipn

                # Perform branch and bound
                results_ = results["model"][cluster_i][skipn:]
                submodels, freq, timed_out_i = self._topn_selection(
                    results_, self._nterms_bnb, self.sams_model_.n_term_groups, self.topn_bnb, self.bnb_timeout
                )
                timed_out = timed_out or timed_out_i
                m_.extend(submodels)
                f_.append(freq)

                # Compute entropies
                entropies_ = self._entropy(X, y, submodels, freq)
                e_.append(entropies_)

            # Combine the different clusters
            submodels = m_
            freq = np.concatenate(f_)
            self.entropies_ = np.concatenate(e_)

        # No submodels: an empty range of sizes, or a timeout on the first size
        if len(submodels) == 0:
            n_forced = len(self.sams_model_.forced)
            fixes = [f"a model_size of at least {n_forced + 3} term groups", "a larger nterms_bnb"]
            reason = ""
            if timed_out:
                reason = f" within bnb_timeout={self.bnb_timeout} seconds"
                fixes.append("a larger bnb_timeout")
            raise ValueError(
                f"The branch-and-bound found no submodels{reason} for model_size={self._model_size} and "
                f"nterms_bnb={self.nterms_bnb}. The submodel sizes run from one more than the {n_forced} forced "
                "term groups up to nterms_bnb (exclusive, by default model_size - 1). Use "
                f"{', '.join(fixes[:-1])} or {fixes[-1]}."
            )

        # Sort models according to entropy
        best_idx = np.argsort(-self.entropies_, kind="stable")
        self.entropies_ = self.entropies_[best_idx]
        self.models_ = [submodels[i] for i in best_idx]
        self.frequencies_ = freq[best_idx]

        # Take only the unique submodels
        submodels = self._unique_submodels(self.models_, self.sams_model_.n_term_groups)
        self.models_ = [model for accept, model in zip(submodels, self.models_, strict=True) if accept]
        self.frequencies_ = self.frequencies_[submodels]
        self.entropies_ = self.entropies_[submodels]

        # Expand group indices to column indices for downstream use
        self.models_ = [self.sams_model_.expand(m) for m in self.models_]

        # Set required parameters
        self.selection_metrics_ = self.entropies_
        self.metric_name_ = "Entropy"

        return self

    def plot_selection(self, ntop=5, model=None):
        """
        Creates a raster plot of the fitted SAMS procedure.

        Parameters
        ----------
        ntop : int
            The number of top model terms to indicate in the raster plot.
        model : pd.DataFrame
            The dataframe of the model used in
            :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>` used to
            label the raster plot.

        Returns
        -------
        fig : :py:class:`plotly.graph_objects.Figure`
            The Plotly Figure object of the raster plot.
        """
        assert self.is_fitted, "You must fit the regressor before plotting the selection plot"

        # Create default term labels
        if model is not None:
            terms = model2encnames(model, self.effect_types_)
        else:
            terms = [f"x{i}" for i in range(self.n_encoded_features_)]

        # Extract top raster terms
        raster_terms = reduce(np.union1d, self.models_[:ntop])

        # Expand forced groups to their columns (raster is in column space)
        forced = self.sams_model_.expand(self.sams_model_.forced)

        # Check for amount of clusters (the elbow needs at least three cluster counts)
        if self.kmeans_ is not None and self.kmeans_.dists is not None and len(self.kmeans_.dists) >= 3:
            # Create the plot
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"colspan": 2}, None],
                    [{}, {}],
                ],
                shared_yaxes=True,
                column_widths=[0.8, 0.2],
                row_heights=[0.3, 0.7],
            )

            # Plot the cluster selection (for the cluster counts tried)
            dists = self.kmeans_.dists
            fig.add_trace(
                go.Scatter(x=np.arange(1, len(dists) + 1), y=dists / dists[0], showlegend=False),
                row=1,
                col=1,
            )
            elbow = np.diff(dists, n=2)
            fig.add_trace(go.Bar(x=np.arange(2, len(dists)), y=elbow / np.max(elbow), showlegend=False), row=1, col=1)
            fig.update_xaxes(title="nb clusters", row=1)
            fig.update_yaxes(title="", row=1)

            # Plot the raster
            fig = plot_raster(
                self.results_,
                terms,
                self._skipn,
                "r2(adj)",
                forced,
                raster_terms,
                self.kmeans_,
                (fig, (2, 1), (2, 2)),
            )

        else:
            # Plot simple raster
            fig = plot_raster(self.results_, terms, self._skipn, "r2(adj)", forced, raster_terms, self.kmeans_)

        return fig
