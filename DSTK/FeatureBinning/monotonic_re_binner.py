from __future__ import division
import numpy as np
import pandas as pd
import pdb
import scipy.stats as st
from DSTK.FeatureBinning.base_binner import BaseBinner
from DSTK.FeatureBinning.conditional_inference_binner import ConditionalInferenceBinner


class MonotonicBinner(BaseBinner):
    """
    This binner strives for monotonicity by removing splits established by ConditionalInferenceBinner and
    DecisionTreeBinner

    This merges the bin(s) of choice with the bin(s) *directly to the right*
    """
    @property
    def values(self):
        return self._values

    @property
    def splits(self):
        return self._splits

    @property
    def counts(self):
        return self._counts

    @property
    def is_fit(self):
        return self._is_fit

    @values.setter
    def values(self, values):
        self._values = values

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    @counts.setter
    def counts(self, counts):
        self._counts = counts

    @is_fit.setter
    def is_fit(self, is_fit):
        self._is_fit = is_fit

    def __init__(self, binner, **kwargs):

        self.name = binner.name
        self._splits = list(binner.splits)
        self._values = list(binner.values)
        self._counts = list(binner.counts)
        self._deprecated_splits = list()
        self._special_values = list(binner.special_values)

        self.min_splits = kwargs.get('min_splits', 2)
        self.max_splits_to_pop = kwargs.get('max_splits_to_pop', 1)
        self.idx_to_remove = kwargs.get('idx_to_remove', None)

        self._is_fit = False

    def fit(self):

        assert ~self._check_bin_linearity(return_as='bool'), "Binner is already monotonic"

        # If user has provided specific indices to remove, override all other settings and just handle those
        if self.idx_to_remove is not None:
            self.idx_to_remove = np.atleast_1d(self.idx_to_remove)
            for idx in self.idx_to_remove:
                self._pop_split(idx)
            return self

        while (~self._check_bin_linearity(return_as='bool')) & (len(self._deprecated_splits) < self.max_splits_to_pop) & (len(self._splits) > self.min_splits):
            to_pop = self._find_splits_to_pop()
            self._pop_split(to_pop[0])

        return self

    def _find_splits_to_pop(self):
        return self._check_bin_linearity(return_as='which')

    def _pop_split(self, idx):

        # No removing the uppermost bin
        if not np.isfinite(self._splits[idx]):
            idx -= 1

        # Pop
        removed_splits = self._splits.pop(idx)
        removed_vals = self._values.pop(idx)
        removed_counts = self._counts.pop(idx)

        # Re-compute
        self._values[idx] = self._re_weight_cond_prob(self._counts[idx], removed_counts, self._values[idx], removed_vals)
        self._counts[idx] += removed_counts

        self._deprecated_splits.append(removed_splits)

    @staticmethod
    def _re_weight_cond_prob(n1, n2, v1, v2):

        new_distrib = np.array(v1)*n1 + np.array(v2)*n2
        return list(new_distrib / new_distrib.sum())

    def _check_bin_linearity(self, return_as='bool'):
        '''
        Check whether the bins in binner increase or decrease monotonically. Returns either boolean T/F, or the # of bins
        that deviate from monotonic

        :param binner: BaseBinner instance
        :param min_splits: number of splits necessary to check for monotonicity (default 2)
        :param return_as: Return boolean of whether the bins increase/decrease monotonically (default T)
        :return:
        '''

        if len(self.splits) <= self.min_splits:
            return False

        sp = np.asarray(self.splits)
        vals = np.asarray(self.values)
        diffs = np.diff(vals[np.asarray([(~np.isnan(s)) & (s not in self._special_values) for s in sp])][:, 1])

        if return_as == 'bool':
            return (diffs > 0).all() or (diffs < 0).all()

        if return_as == 'sum':
            return (len(diffs) - np.abs(np.sign(diffs).sum())) / 2

        if return_as == 'which':
            return np.ravel(np.argwhere(np.sign(diffs) != np.sign(diffs[0])) + 1)













