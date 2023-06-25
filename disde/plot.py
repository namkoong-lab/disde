import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def grid_plots(num_plots, num_cols=5, xsize=3, ysize=3, **kwargs):
    num_rows = int(np.ceil(num_plots / num_cols))
    fig,ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * xsize, num_rows * ysize), **kwargs)
    return fig,ax

def get_grid_ax(idx, axes):
    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            return axes[idx]
        num_cols = axes.shape[1]
        return axes[idx // num_cols][idx % num_cols]
    else:
        return axes

def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None, 
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir=None, balance=False):
    
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    for i in range(3):
        ax[i].set_box_aspect(1)
        ax[i].set_xlim(0,1)
    fig.suptitle("Calibration: {}".format(name), fontsize="x-large")

    if p_weights is None: p_weights = np.ones_like(prop_p)
    if q_weights is None: q_weights = np.ones_like(prop_q)

    p_sample_weights = p_weights.copy()
    q_sample_weights = q_weights.copy()
    if balance:
        p_sample_weights = p_sample_weights / p_sample_weights.sum()
        q_sample_weights = q_sample_weights / q_sample_weights.sum()

    conf_scores, bin_edges = np.histogram(np.concatenate([1-prop_p, prop_q]),bins=nbins, density=True, 
                                          weights=np.concatenate([p_sample_weights, 
                                                                 q_sample_weights]),
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2

    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
    
    ax[0].plot(bin_mids, nanmask * conf_scores / (conf_scores + conf_scores[::-1]), color='green')
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Proportion correct')
    ax[0].set_xlabel('Predicted probability')
    ax[0].set_title('Prop calibration: combined')

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_p]),bins=nbins, weights=p_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[1].plot(bin_mids, conf_scores, color='green')
    ax[1].set_title('Density: P')
    ax[1].set_xlabel('Predicted probability of Q')
    ax[1].set_ylim(bottom=0)

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_q]),bins=nbins, weights=q_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[2].plot(bin_mids, conf_scores, color='green')
    ax[2].set_title('Density: Q')
    ax[2].set_xlabel('Predicted probability of Q')
    ax[2].set_ylim(bottom=0)

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + '/' + name.replace(' ', '_')+'_calibration.pdf')

def plot_balance(eval_data, w, plot_titles=None, grid_plots_kwargs=dict(xsize=2.5, ysize=2)):
    x_weighted_0 = (w[eval_data.t==0] * eval_data.x[eval_data.t==0].T).T.mean(0) / w[eval_data.t==0].mean(0)
    x_weighted_1 = (w[eval_data.t==1] * eval_data.x[eval_data.t==1].T).T.mean(0) / w[eval_data.t==1].mean(0)
    x_unweighted_0 = eval_data.x[eval_data.t==0].mean(0)
    x_unweighted_1 = eval_data.x[eval_data.t==1].mean(0)

    num_plots = len(x_weighted_0)
    fig,ax = grid_plots(num_plots, **grid_plots_kwargs)
    if plot_titles is None:
        plot_titles = range(num_plots)
    for i, plot_title in zip(range(num_plots), plot_titles):
        this_ax = get_grid_ax(i, ax)
        pd.Series([x_unweighted_0[i], x_weighted_0[i], x_weighted_1[i], x_unweighted_1[i]]).plot(ax=this_ax, marker='.', title=plot_title)
        this_ax.get_xaxis().set_ticks([])
    fig.tight_layout()


