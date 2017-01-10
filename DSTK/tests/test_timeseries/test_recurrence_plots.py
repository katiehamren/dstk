import DSTK.Timeseries.recurrence_plots as rp
import DSTK.Timeseries._recurrence_map as rm
import numpy as np


def test_recurrence_map():
    ts = np.arange(0, 1, .1)
    np.testing.assert_almost_equal(rm.recurrence_map(ts, ts),
                                   [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                    [0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                    [0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                    [0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                    [0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                                    [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3],
                                    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2],
                                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1],
                                    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])


def test_recurrence_dist():
    ts = np.arange(0, 1, .1)
    np.testing.assert_almost_equal(rp.poincare_recurrence_dist(ts, ts),
                                   [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                    [0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                    [0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                    [0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                    [0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                                    [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3],
                                    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2],
                                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1],
                                    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])


def test_recurrence_plot():
    ts = np.arange(0, 1, .1)
    np.testing.assert_array_equal(rp.poincare_map(ts, threshold=0.15),
                                  [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])


def test_recurrence_plot_two_series():
    ts = np.arange(0, 1, .1)
    ts2 = np.sin(2 * np.pi * ts)
    np.testing.assert_array_equal(rp.poincare_map(ts, ts2, threshold=0.15),
                                  [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
