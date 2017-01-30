import matplotlib.pyplot as plt
import numpy as np
import pdb


def plot_binner(binner, **kwargs):
    """
    Possible kwargs:
        - class_labels
        - title
        - annotate
        - ticksize
        - fontsize
    :return:
    """
    _plot_bucket_values(binner.splits, binner.values, title=kwargs.pop('title', binner.name), **kwargs)


def _plot_bucket_values(splits, values,
                        title=None, class_labels={0: '0', 1: '1'}, annotate=False, ticksize=16, fontsize=16,
                        labelsize=18):
    class_0 = [val[0] for val in values]
    class_1 = [val[1] for val in values]
    sp = np.asarray(splits)
    non_na = sp[~np.isnan(sp)]
    non_na = np.insert(non_na, 0, np.NINF)
    label = ['({0:6.2f}, {1:6.2f}]'.format(tup[0], tup[1]) for tup in zip(non_na[:-1], non_na[1:])] + ['nan']
    ind = np.arange(len(class_0))
    w = 0.5
    plt.bar(ind, class_0, w, label=class_labels[0])
    plt.bar(ind, class_1, w, bottom=class_0, color='g', label=class_labels[1])
    plt.xticks(ind + w / 2., label, size=ticksize, rotation=75)
    plt.yticks(size=ticksize)
    plt.legend(fontsize=fontsize)
    if title:
        plt.title(title, size=labelsize)
    plt.xlabel('bucket', size=labelsize)
    plt.ylabel('bucket value', size=labelsize)
