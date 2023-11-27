import json
import pathlib

import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()
import pickle

from Interpretability.utils import get_reports_path, get_cache_path
from matplotlib.ticker import FuncFormatter

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


def to_filename(s, extension):
    return "".join(x if (x.isalnum() or x in "._-") else '_' for x in s) + "." + extension


def create_figure(feature_name, weighted_sum, weight_total, report_dir, filetype):
    assert filetype in ('png', 'pdf')

    mean_by_head = weighted_sum / weight_total
    print(mean_by_head)
    n_layers, n_heads = mean_by_head.shape
    if n_layers == 12 and n_heads == 12:
        plt.figure(figsize=(3,2.5))
        ax1 = plt.subplot2grid((100, 100), (0, 0), colspan=99, rowspan=99) #heatmap
    else:
        raise NotImplementedError

    xtick_labels = [str(i) for i in range(1, n_heads + 1)]
    ytick_labels = [str(i) for i in range(1, n_layers + 1)]
    heatmap = sns.heatmap((mean_by_head).tolist(), center=0.0, ax=ax1,
                          square=True, linecolor='#D0D0D0',
                          cmap=LinearSegmentedColormap.from_list('rg', ["#F14100", "white", "#3D4FC4"], N=256),
                          xticklabels=xtick_labels,
                          yticklabels=ytick_labels)
    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('#D0D0D0')
        spine.set_linewidth(0.1)
    plt.setp(heatmap.get_yticklabels(), fontsize=7)
    plt.setp(heatmap.get_xticklabels(), fontsize=7)
    heatmap.tick_params(axis='x', pad=1, length=2)
    heatmap.tick_params(axis='y', pad=.5, length=2)
    heatmap.yaxis.labelpad = 3
    heatmap.invert_yaxis()
    heatmap.set_facecolor('#E7E6E6')
    ax1.set_xlabel('Head', size=8)
    ax1.set_ylabel('Layer', size=8)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
    fname = report_dir / to_filename(feature_name, filetype)
    print('Saving', fname)
    plt.savefig(fname, format=filetype)
    plt.close()


if __name__ == "__main__":

    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp_name', help='Name of experiment')
    # args = parser.parse_args()
    # print(args)
    exp_name='edge_features_Variability_mean_codebert_python_noneighbor'
    filetype = 'pdf'

    cache_path = get_cache_path() / f'{exp_name}.pickle'
    report_dir = get_reports_path() / exp_name
    print(cache_path)
    print(report_dir)
    pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)

    feature_to_weighted_sum, weight_total = pickle.load(open(cache_path, "rb"))
    # with open(report_dir / 'args.json', 'w') as f:
    #     json.dump(vars(args), f)
    # print(args)
    for feature_name, weighted_sum in feature_to_weighted_sum.items():
        create_figure(feature_name, weighted_sum, weight_total, report_dir, filetype=filetype)