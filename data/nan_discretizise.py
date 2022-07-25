import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import OneHotEncoder


class NanDiscretizer:
    """
    Bin continuous data into intervals.
    Adapted version of sklearn.preprocessing.kBinsDiscretizer
    without checks, ignoring nan vals, and including 1% bins on the edges

    Parameters
    ----------
    n_bins : int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.

    encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
        Method used to encode the transformed result.

        onehot
            Encode the transformed result with one-hot encoding
            and return a sparse matrix. Ignored features are always
            stacked to the right.
        onehot-dense
            Encode the transformed result with one-hot encoding
            and return a dense array. Ignored features are always
            stacked to the right.
        ordinal
            Return the bin identifier encoded as an integer value.

    strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        Strategy used to define the widths of the bins.

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D k-means
            cluster.

    Attributes
    ----------
    n_bins_ : int array, shape (n_features,)
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    bin_edges_ : array of arrays, shape (n_features, )
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    See Also
    --------
     sklearn.preprocessing.Binarizer : Class used to bin values as ``0`` or
        ``1`` based on a parameter ``threshold``.
    """

    def __init__(self, n_bins=5, *, encode='onehot', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Data to be discretized.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
        """
        valid_encode = ('onehot', 'onehot-dense', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_encode, self.encode))
        valid_strategy = ('uniform', 'quantile', 'kmeans', 'quantile_outlier')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))

        n_features = X.shape[1]
        n_bins = np.full(n_features, self.n_bins, dtype=np.int)
        self.col_names = X.columns
        X = X.values  # DataFrame to ndarray

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]

            # Newcode: Remove nans and 0
            # column = column[~np.isnan(column)]
            column = column[column != 0]

            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn("Feature %d is constant and will be "
                              "replaced with 0." % jj)
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == 'uniform':
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)

            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            # Newcode
            elif self.strategy == 'quantile_outlier':
                """Quantiles that reserve the outer quantiles for 1% outliers"""
                # get 1% outlier borders
                col_sorted = np.sort(column)
                outlier_min_val = col_sorted[int(column.shape[0] * 0.01)]
                outlier_max_val = col_sorted[int(column.shape[0] * 0.99)]
                # remove outliers from data
                data = column[column < outlier_max_val]
                data = data[outlier_min_val < data]

                # set n_bins[jj] so that each bucket can have at least 2000 datapoints, or max = n_bins
                # n_bins[jj] = min(n_bins[jj], max(2, data.shape[0]//2500) + 2)

                # get n_bins-2 quantiles for the inlier points
                quantiles = np.linspace(0, 100, n_bins[jj] - 1)
                norm_bins = np.asarray(np.percentile(data, quantiles))
                # add the 2 outlier borders as outer bins
                norm_bins = np.insert(norm_bins, 0, col_min)
                bin_edges[jj] = np.append(norm_bins, col_max)

            elif self.strategy == 'kmeans':
                from sklearn.cluster import KMeans

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                def log2_transform(data):
                    """transform positive and negative data to log2 scale"""
                    signs = np.sign(data)
                    return np.power(abs(data), (1/2)) * signs

                def log2_reverse_transform(data):
                    """reverse transform positive and negative data from log2 scale"""
                    signs = np.sign(data)
                    return (abs(data))**2 * signs

                # 1D k-means procedure
                km = KMeans(n_clusters=n_bins[jj], init=init, n_init=1)
                data = column[:, None]  # Newcode: Transform data to mitigate outlier influence
                centers = km.fit(data).cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                # Newcode: revert data Transformation
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ('quantile', 'kmeans'):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warningsXt = X.values.warn('Bins whose width are too small (i.e., <= '
                                               '1e-8) in feature %d are removed. Consider '
                                               'decreasing the number of bins.' % jj)
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        if 'onehot' in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i + 1) for i in self.n_bins_],  # Newcode: Added 1 col for 0
                sparse=self.encode == 'onehot')
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_)), dtype=int))

        return self

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        Xt : pandas.DataFrame
            Data in the binned space.
        """

        n_features = self.n_bins_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, X.shape[1]))

        bin_edges = self.bin_edges_
        value_indices = []  # list of pd.DataFrame.index entries of all features, marking non-nan/zero vals
        discretized_cols = []
        for jj in range(X.shape[1]):
            # Newcode: remove zeros, remember index, convert to ndarray
            column = X.iloc[:, jj]
            zero_idx = column[column == 0].index
            column = column[column != 0]  # remove zeros
            value_idx = column.index  # remember index
            column = column.values  # convert to ndarray

            # Values which are close to a bin edge are susceptible to numeric
            # instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation. See documentation of
            # numpy.isclose for an explanation of ``rtol`` and ``atol``.
            rtol = 1.e-5
            atol = 1.e-8
            eps = atol + rtol * np.abs(column)
            column = np.digitize(column + eps, bin_edges[jj][1:])
            np.clip(column, 0, self.n_bins - 1, out=column)

            # join discretized, nan and zeros together again
            col = pd.Series(column, index=value_idx, name=self.col_names[jj])
            col = col.append(pd.Series(self.n_bins, index=zero_idx))
            discretized_cols.append(col.sort_index())

        Xt = pd.concat(discretized_cols, axis=1)

        if self.encode == 'ordinal':
            Xt.columns = self.col_names
            return Xt

        # encode to ndarray
        Xt = self._encoder.transform(Xt).toarray()

        # add column names
        col_names = []
        for col in range(n_features):
            for bin in range(self.n_bins):
                bmin = f'{self.bin_edges_[col][bin]:.0f}' if bin != 0 else '-inf'
                bmax = f'{self.bin_edges_[col][bin + 1]:.0f}' if bin != self.n_bins - 1 else 'inf'
                col_names.append(self.col_names[col] + '_' + bmin + '_' + bmax)
            col_names.append(self.col_names[col] + '_0')

        Xt = pd.DataFrame(Xt, columns=col_names, index=X.index)
        return Xt
