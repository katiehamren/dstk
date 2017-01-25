import sklearn.datasets as ds
from DSTK.FeatureBinning.conditional_inference_binner import ConditionalInferenceBinner
from DSTK.FeatureBinning.monotonic_re_binner import MonotonicBinner
from DSTK.FeatureBinning.decision_tree_binner import DecisionTreeBinner
import numpy as np
import pandas as pd
import pdb

cancer_ds = ds.load_breast_cancer()
cancer_data = cancer_ds['data']
cancer_target = cancer_ds['target']

cancer_df = pd.DataFrame(cancer_data, columns=cancer_ds['feature_names'])

col = 'mean radius'
data = cancer_df[col].values

# cancer data is perfectly monotonic, so fake the target
fake_target = cancer_target.copy()
bin_idx = (data.values > 11.75) & (data.values <= 13.08)
fake_target[bin_idx] = 0.0

assert_str = \
"""
<= 11.75: [ 0.02  0.98]
<= 15.0399999619: [ 0.6194332  0.3805668]
<= 15.2700004578: [ 0.7  0.3]
<= 16.8400001526: [ 0.84090909  0.15909091]
<= inf: [ 0.99152542  0.00847458]
NaN: [ 0.55711775  0.44288225]
"""


def test_ctree_rebinning():

    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, fake_target)

    redo = MonotonicBinner(cib)
    redo.fit()

    np.testing.assert_equal(redo.splits, [11.75, 13.109999656677246, 15.270000457763672, 16.84000015258789, np.PINF, np.NaN])
    np.testing.assert_equal(redo.values, [[0.02, 0.98],
                                         [0.9914529914529915, 0.008547008547008548],
                                         [0.31428571428571428, 0.68571428571428572],
                                         [0.8409090909090909, 0.1590909090909091],
                                         [0.9915254237288136, 0.00847457627118644],
                                         [0.5571177504393673, 0.4428822495606327]])
    np.testing.assert_equal(redo.counts, [150, 117, 140, 44, 118, 0])


def test_ctree_manual_rebinning():

    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, fake_target)

    redo = MonotonicBinner(cib, idx_to_remove=1)
    redo.fit()

    np.testing.assert_equal(redo.splits, [11.75, 15.039999961853027, 15.270000457763672, 16.84000015258789, np.PINF, np.NaN])
    np.testing.assert_equal(redo.values, [[0.02, 0.98],
                                         [0.61943319838056676, 0.38056680161943318],
                                         [0.7, 0.3],
                                         [0.8409090909090909, 0.1590909090909091],
                                         [0.9915254237288136, 0.00847457627118644],
                                         [0.5571177504393673, 0.4428822495606327]])
    np.testing.assert_equal(redo.counts, [150, 247, 10, 44, 118, 0])


def test_string_repr():
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, fake_target)

    redo = MonotonicBinner(cib, idx_to_remove=1)
    redo.fit()

    assert str(redo) == assert_str


def test_decisiontree_rebinning():

    dtree = DecisionTreeBinner('test_dim_{}'.format(col), max_leaf_nodes=4)
    dtree.fit(data, fake_target)

    redo = MonotonicBinner(dtree)
    redo.fit()

    np.testing.assert_equal(redo.splits, [11.755000114440918, 13.094999313354492, np.PINF, np.NaN])
    np.testing.assert_equal(redo.values, [[0.02, 0.98],
                                          [0.62256809338521402, 0.37743190661478598],
                                          [0.9506172839506173, 0.04938271604938271],
                                          [0.55711775043936729, 0.44288224956063271]])
    np.testing.assert_equal(redo.counts, [150, 257, 162, 0])


def test_decisiontree_manual_rebinning():

    dtree = DecisionTreeBinner('test_dim_{}'.format(col), max_leaf_nodes=4)
    dtree.fit(data, fake_target)

    redo = MonotonicBinner(dtree, idx_to_remove=1)
    redo.fit()

    np.testing.assert_equal(redo.splits, [11.755000114440918, 15.274999618530273, np.PINF, np.NaN])
    np.testing.assert_equal(redo.values, [[0.02, 0.98],
                                         [0.61943319838056676, 0.38056680161943318],
                                         [0.7, 0.3],
                                         [0.8409090909090909, 0.1590909090909091],
                                         [0.9915254237288136, 0.00847457627118644],
                                         [0.5571177504393673, 0.4428822495606327]])
    np.testing.assert_equal(redo.counts, [150, 247, 10, 44, 118, 0])





