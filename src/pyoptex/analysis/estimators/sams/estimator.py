"""
Module for the SAMS regressor.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from functools import reduce
from sklearn.cluster import KMeans
from tqdm import tqdm

from ....utils.design import obs_var_from_Zs
from ....utils.model import identityY2X, sample_model_dep_onebyone
from ....utils.comp import timeout
from ...mixins.fit_mixin import MultiRegressionMixin
from .accept import ExponentialAccept
from .bnb.sams_bnb import SamsBnB
from .entropy import entropies, entropies_approx
from .models.ols_model import OlsModel
from .models.mixed_lm_model import MixedLMModel
from .plot import plot_raster
from .simulation import simulate_sams


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

    Attributes
    ----------
    dependencies : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    mode : None or 'weak' or 'strong'
        The heredity mode during sampling.
    forced_model : np.array(1d)
        Any terms that must be included in the model. Commonly np.array([0], dtype=np.int\_)
        is used to force the intercept when the intercept is the first column in
        the normalized, encoded model matrix. This model must itself fulfill the
        heredity constraints.
    model_size : int
        The size of the overfitted models. Defaults to the number of runs
        divided by three. The overfitted model includes the forced model,
        and its size must thus be larger than the forced model.
    nb_models : int
        Th number of unique models to accept during the sams procedure.
    skipn : int or 'auto'
        The number of worst-fitting models to skip during
        branch-and-bound and entropy calculations. Defaults to 'auto'
        which uses the elbow method and an additional 1% safety
        margin. Any int must be smaller than `nb_models`.
    est_ratios : None or np.array(1d)
        The estimated variance ratios to be used during SAMS. These
        ratios are used to make the SAMS procedure computationally feasible
        in mixed models. For every random effect, a ratio should be provided.
        Defaults to 1 for each random effect if None is specified.
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
        on. If None, every size from one to the `model_size` - 2 (inclusive) is tested
        as recommended by the original paper. If an int, every size from 
        one until the specified number is tested. If an iterable, only the
        values from the iterable are tested.
    bnb_timeout : int
        The maximum number of seconds to run the branch-and-bound algorithm for.
        Clear submodels in the raster plot will not require much time in
        the branch-and-bound algorithm. Therefore, if the branch-and-bound
        algorithm would require too much time, most likely low entropy models
        are the result and the computation can be halted prematurely. Defaults
        to three minutes.
    entropy_sampler : func(dep, model_size, N, forced, mode)
        The sampler to use when generating random hereditary models. See the
        documentation on customizing SAMS for an indication on which sampler to use.
    entropy_sampling_N : int
        The number of random samples to draw using the sampler to compute
        the theoretical frequencies of the submodels.
    entropy_model_order : dict(str: ('lin' or 'tfi' or 'quad'))
        The order of the terms in the model. Please read the warning in
        the documentation on customizing SAMS.
    tqdm : bool
        Whether to use tqdm to track the progress
    sams_model\_ : :py:class:`Model <pyoptex.analysis.estimators.sams.models.model.Model>`
        A SAMS model used in sampling and fitting data during the SAMS procedure.
    results\_ : np.array(1d)
        A numpy array with a special datatype where each element contains
        two arrays of size `model_size` ('model', np.int\_), ('coeff', np.float64),
        and one scalar ('metric', np.float64). Results contains `nb_models` elements.
        These are the returned models from the SAMS procedure.
    models\_ : list(np.array(1d))
        The list of models, ordered by entropy.
    entropies\_ : np.array(1d)
        The entropy of each exported model in `models\_`. In
        case of multiple clusters, the entropies are calculated
        within the respective cluster.
    selection_metrics\_ : np.array(1d)
        Alias for `entropies\_`.
    frequencies\_ : np.array(1d)
        The occurence frequency of each submodel in `models\_`
    kmeans\_ : None or :py:class:`sklearn.cluster.Kmeans`
        A kmeans object used to cluster the raster plot. Added 
        a parameter `skips` equal to 5% of the cluster size to
        be skipped for entropy calculations.
    metric_name\_ : str  
        The name of the selection metric.
    """

    def __init__(self, factors=(), Y2X=identityY2X, random_effects=(),
                    dependencies=None, mode=None, forced_model=None,
                    model_size=None, nb_models=10000, skipn='auto', est_ratios=None,
                    max_cluster=8, ncluster=None,
                    topn_bnb=4, nterms_bnb=None, bnb_timeout=180,
                    entropy_sampler=sample_model_dep_onebyone, entropy_sampling_N=10000, 
                    entropy_model_order=None,
                    tqdm=True):
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
            Any terms that must be included in the model. Commonly np.array([0], dtype=np.int\_)
            is used to force the intercept when the intercept is the first column in
            the normalized, encoded model matrix. This model must itself fulfill the
            heredity constraints.
        model_size : int
            The size of the overfitted models. Defaults to the number of runs
            divided by three. The overfitted model includes the forced model,
            and its size must thus be larger than the forced model.
        nb_models : int
            Th number of unique models to accept during the sams procedure.
        skipn : int or 'auto'
            The number of worst-fitting models to skip during
            branch-and-bound and entropy calculations. Defaults to 'auto'
            which uses the elbow method and an additional 1% safety
            margin. Any int must be smaller than `nb_models`.
        est_ratios : None or np.array(1d)
            The estimated variance ratios to be used during SAMS. These
            ratios are used to make the SAMS procedure computationally feasible
            in mixed models. For every random effect, a ratio should be provided.
            Defaults to 1 for each random effect if None is specified.
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
            on. If None, every size from one to the `model_size` - 2 (inclusive) is tested
            as recommended by the original paper. If an int, every size from 
            one until the specified number is tested. If an iterable, only the
            values from the iterable are tested.
        bnb_timeout : int
            The maximum number of seconds to run the branch-and-bound algorithm for.
            Clear submodels in the raster plot will not require much time in
            the branch-and-bound algorithm. Therefore, if the branch-and-bound
            algorithm would require too much time, most likely low entropy models
            are the result and the computation can be halted prematurely. Defaults
            to three minutes.
        entropy_sampler : func(dep, model_size, N, forced, mode)
            The sampler to use when generating random hereditary models. See the
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
        self._model_size = self.model_size if self.model_size is not None else int(len(X) / 3)

        # Set some default values
        self._nterms_bnb = self._model_size - 1 if self.nterms_bnb is None else self.nterms_bnb
        self._nterms_bnb = range(len(self.forced_model) + 1, self._nterms_bnb) \
                                if isinstance(self._nterms_bnb, int) else self._nterms_bnb
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
        assert self.mode in (None, 'weak', 'strong'), 'The drop-mode must be None, weak or strong'
        if self.mode in ('weak', 'strong'):
            assert self.dependencies is not None, 'Must specify dependency matrix if using weak or strong heredity'
            assert len(self.dependencies.shape) == 2, 'Dependencies must be a 2D array'
            assert self.dependencies.shape[0] == self.dependencies.shape[1], 'Dependency matrix must be square'

        # TODO: validate forced_model hereditary

        # Validate SAMS inputs
        if self.model_size is not None:
            if self.forced_model is None:
                assert self.model_size > 0, 'The overfitted model size must be a positive number'
            else:
                assert self.model_size > len(self.forced_model), 'The overfitted model size must be at least one larger than the forced model' 
        assert self.nb_models > 0, 'Must have at least one model to simulate, nb_models must be larger than zero'
        assert self.skipn == 'auto' or isinstance(self.skipn, int), 'Skipn must be "auto" or an integer'
        if self.skipn != 'auto':
            assert 0 <= self.skipn < self.nb_models, 'Cannot skip all SAMS models, skipn must be smaller than nb_models'
        if self.est_ratios is not None:
            assert len(self.est_ratios) == len(self._re), 'Every random effect must have an estimated ratio when specified, in the same order'
        assert self.topn_bnb > 0, 'Must select at least one submodel for each fixed size, topn_bnb must be larger than 0'
        if self.nterms_bnb is not None:
            if isinstance(self.nterms_bnb):
                assert self.nterms_bnb > 0, 'When an integer is specified for nterms_bnb, it must be larger than zero'
        assert self.bnb_timeout > 0, 'Must specify a strictly positive number of seconds for the branch-and-bound to run'
        if self.forced_model is not None:
            assert isinstance(self.forced_model, np.ndarray), 'The forced model must be an integer array'
            assert np.issubdtype(self.forced_model.dtype, np.integer), 'The forced model must be an integer array'
        assert self.max_cluster >= 3, 'The maximum number of clusters for auto selection must be larger than three'
        if self.ncluster is not None:
            assert self.ncluster == 'auto' or isinstance(self.ncluster, int), 'ncluster must be None, "auto" or an integer'
            if self.ncluster != 'auto':
                assert self.ncluster > 0, 'The number of clusters must be larger than or equal to one'

        # Validate model order for entropy calculations
        if self.entropy_model_order is not None:
            # Define the required ordering
            model_types_ord = {'quad': 2, 'tfi': 1, 'lin': 0}

            # Reorder the model types based on the factors
            assert all(str(f.name) in self.entropy_model_order.keys() for f in self._factors), 'All factors must have an entropy model order specified'
            entropy_model_order = {str(f.name): self.entropy_model_order[str(f.name)] for f in self._factors}
            
            # Assert the ordering
            assert np.all(np.diff([model_types_ord[typ] for typ in entropy_model_order.values()]) <= 0), 'Model types must be ordered quad > tfi > lin for entropy calculations'

            # TODO: Perform a random validation of the model to make sure it is correct in Y2X

    def _topn_selection(self, results, sizes, nterms, topn=4, timeout_sec=180):
        """
        Selects the top n submodels of fixed sizes in the results using
        the branch-and-bound algorithm. For each size in `sizes`, the 
        `topn` submodels of that size are extracted from the results.

        Parameters
        ----------
        results : np.array(1d)
            A numpy array with a special datatype where each element contains
            two arrays of size `model_size` ('model', np.int\_), ('coeff', np.float64),
            and one scalar ('metric', np.float64). Results contains `nb_models` elements.
        sizes : iterable(int)
            An iterable of ints with the fixed sizes of the submodels.
        nterms : int
            The total number of fixed effects in the encoded, normalized
            model matrix (=X.shape[1] after encoding and normalization).
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
        """
        # Initialize results
        models = list()
        counts = np.zeros(len(sizes) * topn, dtype=np.int_)

        # Compute BnB
        for i, size in tqdm(enumerate(sizes), total=len(sizes), disable=(not self.tqdm)):
            # Compute the result with a timeout
            result = timeout(
                SamsBnB(size, results, nterms, self.mode, self.dependencies, self.forced_model).top, 
                topn, 
                timeout=timeout_sec
            )

            # Check for a result within the timeout
            if result is not None:
                m, f = result
                models.extend([i for i, _ in m])
                counts[i*topn:(i+1)*topn] = f
            else:
                break

        # Compute frequencies
        frequencies = counts[:len(models)] / len(results)

        return models, frequencies

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
            The list of models
        nterms : int
            The total number of fixed effects in the encoded, normalized
            model matrix (=X.shape[1] after encoding and normalization).

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
        if self.entropy_model_order is None or self.mode != 'weak':
            # Compute approximated entropies
            entropy = entropies_approx(
                submodels, freqs, self._model_size, self.dependencies, self.mode,
                self.forced_model, self.entropy_sampling_N, self.entropy_sampler
            )

        else:
            # Compute the number of factors for each
            model_counts = {e: 0 for e in ('quad', 'tfi', 'lin')}
            for mt, et in zip(self.entropy_model_order.values(), self.effect_types_):
                model_counts[mt] += (1 if et == 1 else et - 1)
            model_counts = [model_counts.get(e, 0) for e in ('quad', 'tfi', 'lin')]

            # Compute entropy (based on model order)
            entropy = entropies(submodels, freqs, self._model_size, model_counts)

        return entropy

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
        assert np.all(self.forced_model < self.n_encoded_features_), 'The forced model must have integers smaller than the number of parameters in X'
        if self.mode is not None:
            assert self.dependencies.shape[0] == X.shape[1], 'Must specify a dependency for each term'

        # Compute SAMS results
        if len(self._re) == 0:
            self.sams_model_ = OlsModel(X, y, forced=self.forced_model, mode=self.mode, dep=self.dependencies)
        else:
            V = obs_var_from_Zs(self.Zs_, len(X), self._est_ratios)
            self.sams_model_ = MixedLMModel(X, y, forced=self.forced_model, mode=self.mode, dep=self.dependencies, V=V)
        accept = ExponentialAccept(T0=(X.shape[0])*np.var(y)/10, rho=0.95, kappa=4)
        self.results_ = simulate_sams(self.sams_model_, self._model_size, accept_fn=accept, nb_models=self.nb_models,
                                        tqdm=self.tqdm)

        # Sort the results
        idx = np.argsort(self.results_['metric'])
        results = self.results_[idx]

        # Skip bad part of the data
        if self.skipn == 'auto':
            # Compute the difference in derivative
            slope = np.diff(results['metric'])
            bkps = rpt.KernelCPD(kernel='linear', min_size=0).fit_predict(slope, pen=np.var(slope)*1000)

            # Extract the skip
            if len(bkps) == 1:
                skipn = 0
            else:
                # Take the last breakpoint
                skipn = bkps[-2] + int(0.01*(len(results) - bkps[-2])) # Add a safety margin for steady state
        else:
            skipn = self.skipn
        results = results[skipn:]

        # Possibly cluster
        if self.ncluster is None:
            # Perform branch and bound
            submodels, freq = self._topn_selection(
                results['model'], self._nterms_bnb, self.n_encoded_features_, self.topn_bnb, self.bnb_timeout
            )

            # Compute entropies
            self.entropies_ = self._entropy(X, y, submodels, freq)

            # Set default value
            self.kmeans_ = None

        else:
            # Expand models to integer cluster (for kmeans)
            results_cluster = np.zeros((len(results), self.n_encoded_features_))
            np.put_along_axis(results_cluster, results['model'], 1, axis=1)

            # Auto detect the number of clusters
            if self.ncluster == 'auto':
                # Fit kmeans
                kmeans_dists = np.zeros(self.max_cluster)
                for nc in range(1, self.max_cluster+1):
                    kmeans = KMeans(n_init='auto', n_clusters=nc).fit(results_cluster)
                    kmeans_dists[nc-1] = kmeans.inertia_

                # Elbow rule for cluster selection
                kmeans_elbow = np.diff(kmeans_dists, n=2)
                ncluster = np.argmax(kmeans_elbow) + 2
            else:
                # Force the number of clusters
                ncluster = self.ncluster
                kmeans_dists = None

            # Fit kmeans on selected number of clusters
            self.kmeans_ = KMeans(n_init='auto', n_clusters=ncluster).fit(results_cluster)
            self.kmeans_.skips = np.zeros(ncluster, dtype=np.int_)
            self.kmeans_.dists = kmeans_dists

            # Perform model select on each cluster
            m_, f_, e_ = [], [], []
            for i in range(0, ncluster):
                # Select cluster i
                cluster_i = (self.kmeans_.labels_ == i)
                ncluster_i = np.sum(cluster_i)

                # Compute skip
                skipn = int(0.05*ncluster_i)
                self.kmeans_.skips[i] = skipn

                # Perform branch and bound
                results_ = results['model'][cluster_i][skipn:]
                submodels, freq = topn_selection(
                    results_, self._nterms_bnb, self.n_encoded_features_, self.topn_bnb, self.bnb_timeout
                )
                m_.extend(submodels)
                f_.append(freq)

                # Compute entropies
                entropies_ = self._entropy(X, y, submodels, freq)
                e_.append(entropies_)

            # Combine the different clusters
            submodels = m_
            freq = np.concatenate(f_)
            self.entropies_ = np.concatenate(e_)

        # Sort models according to entropy
        best_idx = np.argsort(self.entropies_)[::-1]
        self.entropies_ = self.entropies_[best_idx]
        self.models_ = [submodels[i] for i in best_idx]
        self.frequencies_ = freq[best_idx]

        # Take only the unique submodels
        submodels = self._unique_submodels(self.models_, self.n_encoded_features_)
        self.models_ = [model for accept, model in zip(submodels, self.models_) if accept]
        self.frequencies_ = self.frequencies_[submodels]
        self.entropies_ = self.entropies_[submodels]

        # Set required parameters
        self.selection_metrics_ = self.entropies_
        self.metric_name_ = 'Entropy'        

        return self

    def plot_selection(self, ntop=5):
        """
        Creates a raster plot of the fitted SAMS procedure.

        Parameters
        ----------
        ntop : int
            The number of top model terms to indicate in the raster plot.

        Returns
        -------
        fig : :py:class:`plotly.graph_objects.Figure`
            The Plotly Figure object of the raster plot.
        """
        assert self.is_fitted, 'You must fit the regressor before plotting the selection plot'

        # Extract top raster terms
        raster_terms = reduce(np.union1d, self.models_[:ntop])

        # Check for amount of clusters
        if self.kmeans_ is not None and self.kmeans_.dists is not None:

            # Create the plot
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"colspan": 2}, None],
                    [{}, {}],
                ],
                shared_yaxes=True, column_widths=[0.8, 0.2], row_heights=[0.3, 0.7]
            )

            # Plot the cluster selection
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, self.max_cluster+1), 
                    y=self.kmeans_.dists / self.kmeans_.dists[0], 
                    showlegend=False
                ),
                row=1, col=1
            )
            elbow = np.diff(self.kmeans_.dists, n=2)
            fig.add_trace(
                go.Bar(x=np.arange(1+1, self.max_cluster), y=elbow / np.max(elbow), showlegend=False),
                row=1, col=1
            )
            fig.update_xaxes(title='nb clusters', row=1)
            fig.update_yaxes(title='', row=1)

            # Plot the raster
            fig_ = plot_raster(
                self.results_, [f'x{i}' for i in range(self.n_encoded_features_)],
                self.skipn, 'r2(adj)', self.forced_model, 
                raster_terms, self.kmeans_, (fig, (2, 1), (2, 2))
            )

        else:
            # Plot simple raster
            fig = plot_raster(
                self.results_, [f'x{i}' for i in range(self.n_encoded_features_)],
                self.skipn, 'r2(adj)', self.forced_model, 
                raster_terms, self.kmeans_
            )

        return fig
