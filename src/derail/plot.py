import argparse
import contextlib
import functools
import os
import sys

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import seaborn as sns


## We were using sns.boxplot for plotting, which makes
#  matplotlib's boxplot look better.
#  However, the original boxplot boxes are the quartiles, and
#  we wanted it to be the confidence interval - we thus modify
#  `matplotlib.axes.Axes.bxp` so that the boxes represent the confidence
#  interval.
#  Another problem, however, is that sns.boxplot did not support
#  confidence intervals (simply because it is not included in the call
#  parameters), so we override `sns.categorical._BoxPlotter.draw_boxplot`.
#  While this is hacky and there are more principled ways of solving this,
#  it should work just fine for this use case.

import matplotlib.axes
old_bxp = matplotlib.axes.Axes.bxp

def new_bxp(self, bxpstats, *args, **kwargs):
    for stats in bxpstats:
        stats['q1'] = stats['cilo']
        stats['q3'] = stats['cihi']
    return old_bxp(self, bxpstats, *args, **kwargs)

matplotlib.axes.Axes.bxp = new_bxp


def new_draw_boxplot(self, ax, kws):
    """Use matplotlib to draw a boxplot on an Axes."""
    vert = self.orient == "v"

    props = {}
    for obj in ["box", "whisker", "cap", "median", "flier"]:
        props[obj] = kws.pop(obj + "props", {})

    # Begin added lines
    n = len(self.plot_data)
    usermedians = kws.pop('usermedians', [None] * n)
    conf_intervals = kws.pop('conf_intervals', [None] * n)
    # End added lines

    for i, group_data in enumerate(self.plot_data):

        if self.plot_hues is None:

            # Handle case where there is data at this level
            if group_data.size == 0:
                continue

            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = np.asarray(sns.utils.remove_na(group_data))

            # Handle case where there is no non-null data
            if box_data.size == 0:
                continue

            artist_dict = ax.boxplot(box_data,
                                     vert=vert,
                                     patch_artist=True,
                                     positions=[i],
                                     widths=self.width,
                                     # Begin added lines
                                     usermedians=usermedians[i],
                                     conf_intervals=conf_intervals[i],
                                     # End added lines
                                     **kws)
            color = self.colors[i]
            self.restyle_boxplot(artist_dict, color, props)
        else:
            # Draw nested groups of boxes
            offsets = self.hue_offsets
            for j, hue_level in enumerate(self.hue_names):

                # Add a legend for this hue level
                if not i:
                    self.add_legend_data(ax, self.colors[j], hue_level)

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = np.asarray(remove_na(group_data[hue_mask]))

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                center = i + offsets[j]
                artist_dict = ax.boxplot(box_data,
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[center],
                                         widths=self.nested_width,
                                         **kws)
                self.restyle_boxplot(artist_dict, self.colors[j], props)
                # Add legend data, but just for one set of boxes

sns.categorical._BoxPlotter.draw_boxplot = new_draw_boxplot


STYLES = {
    "paper": {
        "axes.facecolor": "lightgray",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "image.cmap": "GnBu",
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 10,
        "legend.fontsize": 10,
        "ytick.labelsize": 10,
    },
    "heatmap": {
        "axes.linewidth": 0.2,
        "figure.figsize": (8, 6),
        "figure.subplot.bottom": 0.00,
        "figure.subplot.left": 0.00,
        "figure.subplot.right": 1.00,
        "figure.subplot.top": 1.00,
        "hatch.linewidth": 0.1,
        "xtick.labelsize": 9,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    },
    "heatmap_full": {
        "figure.figsize": (5.4, 4.8),
    },
    "heatmap_drlhp": {
        "figure.figsize": (2.5, 5.0),
    },
    "taskplots": {
        "figure.figsize": (10, 3.5),
        "figure.subplot.bottom": 0.07,
        "figure.subplot.left": 0.05,
        "figure.subplot.right": 0.99,
        "figure.subplot.top": 0.99,
        "hatch.linewidth": 0.1,
        "xtick.labelsize": 9,
    },

    "algoplots": {
        "boxplot.capprops.linewidth" : 0.01,
        "boxplot.medianprops.linewidth" : 0.01,
        "boxplot.whiskerprops.linewidth" : 0.01,
        "figure.figsize": (10, 3.5),
        "figure.subplot.bottom": 0.09,
        "figure.subplot.left": 0.05,
        "figure.subplot.right": 0.99,
        "figure.subplot.top": 0.99,
        "hatch.linewidth": 0.1,
        "xtick.labelsize": 10,
    },
}


@contextlib.contextmanager
def setup_styles(styles):
    styles = [STYLES[style] for style in styles]
    with plt.style.context(styles):
        yield


def heatmap(data, row_labels, col_labels, fig, ax=None,
            **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, vmin=-0.1, vmax=1.0)

    vertical_bar = False
    if vertical_bar:
        im_ratio = data.shape[0]/data.shape[1]
        plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    else:
        im_ratio = data.shape[1]/data.shape[0]
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="2%", pad=0.05, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="horizontal")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-35, ha="right",
             rotation_mode="anchor")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=-00, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, threshold_data=None, dx=0.0, dy=0.0, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold_data is None:
        threshold_data = data

    data = np.ma.masked_invalid(data)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(threshold_data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(threshold_data[i, j]) > threshold)])
            text = im.axes.text(j + dx, i + dy, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def bootstrap(data, stat_fn, n_samples=1000):
    return np.array([stat_fn(np.random.choice(data, size=len(data))) for _ in range(n_samples)])

def empirical_ci(arr, alpha=95.0):
    percentiles = 50 - alpha / 2, 50, 50 + alpha / 2
    return np.percentile(arr, percentiles)

def bootstrap_ci(data, n_samples=1000, alpha=95.0):
    return empirical_ci(bootstrap(data, np.mean, n_samples=n_samples), alpha=alpha)
    

def normalize(x, a, b):
    return (x - a + 1e-8) / (b - a + 1e-8)


def make_boxplot(*, all_scores, entry, data_column, x_column, x_values, base_folder, y_lims=None, mean_labels=False, show=False, color=None):
    entry_scores = all_scores[all_scores[data_column] == entry]
    present_x_values = np.unique(entry_scores[x_column].values)
    x_values = [x for x in x_values if x in present_x_values]

    scores_per_x = entry_scores.groupby([x_column])['Return'].apply(list).to_dict()
    score_means_per_x = entry_scores.groupby([x_column])['Return'].mean().to_dict()
    usermedians = []
    conf_intervals = []
    for x in x_values:
        lower, median, upper = bootstrap_ci(scores_per_x[x])
        usermedians.append(score_means_per_x[x])
        conf_intervals.append([lower, upper])

    usermedians = np.array(usermedians)[:, None]
    conf_intervals = np.array(conf_intervals)[:, None, :]

    ax1 = sns.boxplot(x=x_column, y='Return', color=color, order=x_values, hue_order=x_values, data=entry_scores, showmeans=False, meanline=False, whis=(0, 100), linewidth=0.8, usermedians=usermedians, conf_intervals=conf_intervals)

    ax2 = sns.swarmplot(x=x_column, y='Return', order=x_values, data=entry_scores, color='k', size=3.2)

    if y_lims is not None:
        plt.ylim(*y_lims)

    ax1.set_xlabel('')
    ax2.set_xlabel('')

    ax1.set_ylabel('')
    ax2.set_ylabel('')

    if mean_labels:
        cmap = matplotlib.cm.get_cmap('winter')

        get_label = lambda s : str(round(s, 2))
        get_color = lambda s : cmap(s)

        upper_labels = [str(round(norm_means[task, algo], 2)) for task in a_task_names]
        colors = [cmap(norm_means.get((task, algo), 0.0)) for task in a_task_names]

        y_pos = -0.33
        for x_pos, _ in enumerate(a_task_names):
            score = norm_means.get((task, algo), 0.0)
            label = get_label(score)
            color = get_color(score)
            breakpoint()
            ax1.text(
                x_pos,
                y_pos,
                label,
                transform=ax1.get_xaxis_transform(),
                horizontalalignment='center',
                color=color,
                weight='bold',
            )


    for ext in ('pdf', 'png'):
        folder = os.path.join(base_folder, ext)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f'{entry}.{ext}'))

    if show:
        plt.show()
    plt.clf()

def process_results(
    *,
    algo_specs,
    task_specs,
    results_file,
    heatmap_styles=None,
    show=False,
    base_folder='',
    color=None,
):
    if heatmap_styles is None:
        heatmap_styles = ['heatmap']

    base_folder = os.path.join('plots/', base_folder)

    algo_names, algos = zip(*algo_specs)
    to_algo_display_name = {res_name : display_name for display_name, res_name in algo_specs}

    task_names, tasks = zip(*task_specs)
    to_task_display_name = {res_name : display_name for display_name, res_name in task_specs}

    names = ['Task', 'Algorithm', 'Return']
    df = pd.read_csv(results_file, names=names, sep=' ')

    df['Algorithm'] = df['Algorithm'].map(lambda a : to_algo_display_name.get(a, a))
    df['Task'] = df['Task'].map(lambda a : to_task_display_name.get(a, a))

    algo_names = [algo for algo in algo_names if algo in df['Algorithm'].values]
    task_names = [task for task in task_names if task in df['Task'].values]

    task_random = df[df['Algorithm'] == 'Random'].drop('Algorithm', axis=1).groupby('Task').mean().to_dict()['Return']

    if 'expert' in df['Algorithm'].values:
        task_max = df[df['Algorithm'] == 'Expert'].drop('Algorithm', axis=1).groupby('Task').mean().to_dict()['Return']
    else:
        task_max = df.drop('Algorithm', axis=1).groupby('Task').max().to_dict()['Return']

    task_algo_pairs = df.groupby(['Task', 'Algorithm']).mean().to_dict()['Return'].keys()

    def norm_val(task, algo, val):
        return normalize(val, task_random[task], task_max[task]) if (task, algo) in task_algo_pairs else np.nan

    all_scores = df.copy()
    all_scores['Return'] = df.apply(lambda row : norm_val(row['Task'], row['Algorithm'], row['Return']), axis=1)

    all_scores = all_scores[all_scores['Algorithm'].map(lambda a : a in algo_names)]

    limit_negative_scores = True
    if limit_negative_scores:
        all_scores.loc[all_scores['Return'] < -1.0, 'Return'] = -1.0

    mean_scores_dict = all_scores.groupby(['Task', 'Algorithm']).mean().to_dict()['Return']
    mean_or_nan = lambda t, a: mean_scores_dict[t, a] if (t, a) in mean_scores_dict else np.nan
    mean_scores = np.array([[mean_or_nan(task, algo) for algo in algo_names] for task in task_names])

    def make_heatmap():
        fig, ax = plt.subplots()

        is_task_rows = True
        if is_task_rows:
            im = heatmap(mean_scores, task_names, algo_names, fig=fig, ax=ax)
        else:
            im = heatmap(mean_scores.T, algo_names, task_names, fig=fig, ax=ax)

        valfmt = matplotlib.ticker.FuncFormatter(lambda x, _ : f'{x:.2f}' if isinstance(x, np.float64) else '')
        texts = annotate_heatmap(im, valfmt=valfmt)

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            os.makedirs(base_folder, exist_ok=True)
            plt.savefig(os.path.join(base_folder, f'heatmap.{ext}'))

        if show:
            plt.show()

        plt.clf()
        plt.close(fig)


    make_taskplot = functools.partial(make_boxplot, all_scores=all_scores, data_column='Task', x_column='Algorithm', x_values=algo_names, base_folder=os.path.join(base_folder, 'taskplots'), color=color)
    make_algoplot = functools.partial(make_boxplot, all_scores=all_scores, data_column='Algorithm', x_column='Task', x_values=task_names, base_folder=os.path.join(base_folder, 'algoplots'), color=color, y_lims=(-1.05, 1.05))

    with setup_styles(['paper'] + list(heatmap_styles)):
        make_heatmap()

    with setup_styles(['paper', 'taskplots']):
        for task in task_names:
            make_taskplot(entry=task, show=show)

    with setup_styles(['paper', 'algoplots']):
        for algo in algo_names:
            make_algoplot(entry=algo, show=show)


def get_type_kwargs(typ):
    if typ == 'full':
        return get_full_kwargs()
    elif typ == 'noise':
        return get_noise_kwargs()
    else:
        return get_drlhp_kwargs()

def get_full_kwargs():
    return dict(
        algo_specs=[
            ('Expert', 'expert'),
            ('Random', 'random'),
            ('PPO', 'ppo'),
            ('BC', 'bc'),
            ('GAIL_IM', 'gail_im'),
            ('GAIL_SB', 'gail_sb'),
            ('GAIL_FU', 'gail_fu'),
            ('AIRL_FU', 'airl_fu'),
            ('AIRL_IM_SA', 'airl_im_sa'),
            ('AIRL_IM_SO', 'airl_im_so'),
            ('DRLHP_SA', 'drlhp_sa'),
            ('DRLHP_SO', 'drlhp_so'),
            ('MaxEnt_IRL', 'maxent_irl'),
            ('MCE_IRL', 'mce_irl'),
        ],
        task_specs=[
            ('RiskyPath', 'RiskyPath'),
            ('EarlyTerm+', 'EarlyTermPos'),
            ('EarlyTerm-', 'EarlyTermNeg'),
            ('Branching', 'Branching'),
            ('InitShift', 'InitShift'),
            ('NoisyObs', 'NoisyObs'),
            ('Parabola', 'Parabola'),
            ('LargestSum', 'LargestSum'),
            ('ProcGoal', 'ProcGoal'),
            ('Sort', 'Sort'),
        ],
        base_folder='full',
        heatmap_styles=['heatmap', 'heatmap_full'],
    )

def get_noise_kwargs():
    return dict(
        algo_specs=[
            ('DRLHP_SA', 'preferences'),
        ],
        task_specs=[
            ('L=5', 'noisy_obs_v7'),
            ('L=50', 'noisy_obs_v8'),
            ('L=500', 'noisy_obs_v9'),
        ],
        base_folder='noise',
        color='#44BB44',
    )

def get_drlhp_kwargs():
    return dict(
        algo_specs=[
            ('DRLHP_SA', 'drlhp_sa'),
            ('DRLHP_SLOW', 'drlhp_slow'),
            ('DRLHP_GREEDY', 'drlhp_eps'),
            ('DRLHP_BONUS', 'drlhp_rnd'),
        ],
        task_specs=[
            ('RiskyPath', 'RiskyPath'),
            ('EarlyTerm+', 'EarlyTermPos'),
            ('EarlyTerm-', 'EarlyTermNeg'),
            ('Branching', 'Branching'),
            ('InitShift', 'InitShift'),
            ('NoisyObs', 'NoisyObs'),
            ('Parabola', 'Parabola'),
            ('LargestSum', 'LargestSum'),
            ('ProcGoal', 'ProcGoal'),
            ('Sort', 'Sort'),
        ],
        base_folder='drlhp',
        heatmap_styles=['heatmap', 'heatmap_drlhp'],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='results-last.csv')
    parser.add_argument('-s', '--show', action='store_true', default=False)
    parser.add_argument('-t', '--type', type=str, default='full')
    args = parser.parse_args()

    process_results(
        **get_type_kwargs(args.type),
        results_file=args.file,
        show=args.show,
    )


if __name__ == '__main__':
    main()
