import abc
import numpy as np
import warnings


class BaseBinner(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def splits(self):
        raise NotImplementedError

    @splits.setter
    def splits(self, values):
        raise NotImplementedError

    @abc.abstractproperty
    def values(self):
        raise NotImplementedError

    @values.setter
    def values(self, values):
        raise NotImplementedError

    @abc.abstractproperty
    def counts(self, counts):
        raise NotImplementedError

    @counts.setter
    def counts(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, values, targets):
        raise NotImplementedError

    @abc.abstractproperty
    def is_fit(self):
        raise NotImplementedError

    @is_fit.setter
    def is_fit(self, is_fit):
        raise NotImplementedError

    def transform(self, values, **kwargs):
        """
        #     See output of 'fit_transform()'
        #     :param feature_values:
        #     :param class_index:
        #     :return:
        #     """
        if not self.is_fit:
            raise AssertionError("FeatureBinner has to be fit to the data first.")

        class_index = kwargs.get('class_index', 1)
        idx = np.searchsorted(self.splits, values, side='right')

        # searchsorted on nan values gives wrong idx when input is nan
        # as it assumes the nan value is "right" of a nan split
        idx[np.where(np.isnan(values))[0]] -= 1

        if class_index:
            return np.asarray(self.values)[idx][:, class_index]
        else:
            return np.asarray(self.values)[idx]

    def fit_transform(self, feature_values, target_values, **kwargs):
        """
        :param feature_values: list or array of the feature values
        :param target_values: list or array of the corresponding labels
        :param class_index: Index of the corresponding class in the conditional probability vector for each bucket.
               Defaults to 1 (as mostly used for binary classification)
        :return: list of cond_proba_buckets with corresponding conditional probabilities P( T | x in X )
                 for a given example with value x in bin with range X to have label T and list of conditional probabilities for each value to be of class T
        """
        self.fit(feature_values, target_values)
        return self.transform(feature_values, **kwargs)

    def add_bin(self, right_bin_edge, bin_value):

        if right_bin_edge in self.splits:
            warnings.warn("Bin edge already exists.", UserWarning)
            return self

        # use np.searchsorted instead of digitize as the latter
        # gives the wrong result for purely non-numeric splits
        # see corresponding test for the issue
        idx = np.searchsorted(self.splits, right_bin_edge, side='right')
        self.splits = np.insert(self.splits, idx, right_bin_edge).tolist()
        self.values = np.insert(self.values, [idx], bin_value, axis=0).tolist()

        return self

    def _set_counts(self, values):
        binned_values = np.digitize(values, self.splits, right=True)
        self._counts = [(binned_values == bv).sum() for bv in range(len(self.splits))]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        sp = np.asarray(self.splits)
        vals = np.asarray(self.values)
        non_na_sp = sp[~np.isnan(sp)]
        non_na_val = vals[~np.isnan(sp)]
        na_val = vals[np.isnan(sp)].flatten()

        non_na_str = ["<= {}: {}".format(split, val) for split, val in zip(non_na_sp, non_na_val)]
        non_na_str += ["NaN: {}".format(na_val)]

        return "\n".join(non_na_str)

