import numpy as np
import pandas as pd
from disde.dataset import Dataset, subset_dataset, check_binary_values
from disde.model import Model
from disde.fit_predict import CrossFitIdx, cross_fit_predict 
import matplotlib.pyplot as plt

class DistShiftDecomp:
    def __init__(self, eprp=None, esrp=None, esrq=None, eqrq=None):
        self.eprp = eprp
        self.esrp = esrp
        self.esrq = esrq
        self.eqrq = eqrq
    def __getstate__(self):
        return self.__dict__
    def __setstate(self, d):
        self.__dict__.update(d)
    def series(self):
        return pd.Series([self.eprp, self.esrp, self.esrq, self.eqrq], index=['eprp','esrp','esrq','eqrq'])
    def plot(self, **kwargs):
        self.series().plot(marker='o', **kwargs)
    def __str__(self):
        return str(self.series())
    def __repr__(self):
        return repr(self.series())

def get_overlap_weights(data: Dataset, pi, alpha=None):
    '''
    Returns weights where the shared space has density proportional to
        p(x)q(x) / (p(x)+q(x))
    where p(x) is the density of P (X when T=0) and
          q(x) is the density of Q (X when T=1).
    '''
    a = alpha
    if a is None:
        a = data.t.mean()
    t = data.t
    denom = a * (1-pi) + pi * (1-a)
    w = t * a * (1-pi) / denom + (1-t) * (1-a) * pi / denom
    return w

def get_min_weights(data: Dataset, pi):
    '''
    Gets weights where the shared space has density proportional to
        min(p(x), q(x))
    where p(x) is the density of P (X when T=0) and
          q(x) is the density of Q (X when T=1).

    Letting a=E[T], since the propensity score 
        pi=aq/(aq+(1-a)p), 
    the weights from P to S are proportional to min(1, q/p), and 
    the weights from Q to S are proportional to min(1, p/q).
    '''
    t = data.t
    a = data.t.mean()
    with np.errstate(all='ignore'):
        q_over_p = pi / (1-pi) * (1-a) / a
        p_over_q = 1 / q_over_p
        w_min = t * np.clip(p_over_q, 0, 1) + (1-t) * np.clip(q_over_p, 0, 1)
    return w_min

def get_clipped_ate_weights(data: Dataset, pi, clip_eps=0.001):
    '''
    Gets weights where the shared space is basically like the ATE
    except it doesn't have support where dP/dQ or dQ/dP are below 
    a certain threshold
    '''
    assert 0 < clip_eps < 0.5
    t = data.t
    with np.errstate(all='ignore'):
        l = pi / (1-pi)
        ate_w = np.nan_to_num(l * (1-t) + (1/l) * t)
        clip_mask = ((l > clip_eps) & (l < 1/clip_eps)).astype(int)
    return ate_w * clip_mask

def get_decomp(data: Dataset, w):
    return DistShiftDecomp(eprp=data.wate(1-data.t),
                           esrp=data.wate((1-data.t)*w),
                           esrq=data.wate(data.t*w),
                           eqrq=data.wate(data.t))

def get_overlap_decomp(data: Dataset, pi):
    w = get_overlap_weights(data, pi)
    return get_decomp(data, w)

def fit_overlap_decomp(data: Dataset, prop_model: Model, 
        cf: CrossFitIdx, return_extras=False):
    '''Currently we have not thought about sample weighting'''
    check_binary_values(data.t)
    pi = cross_fit_predict(data, prop_model, cf)
    a = data.t.mean()
    w = get_overlap_weights(data, pi)
    res = get_decomp(data, w)
    if return_extras:
        extras = {'w':w, 'props':pi, 'a': a}
        return res, extras
    return res

def fit_eval_model_and_get_eval_dataset(orig_data,
        eval_model,
        rg=np.random.default_rng(1),
        prop_test=0.2,
        return_extras=False):
    '''
    fit an evaluation model, return performance dataset
    with source val and test
    
    Arguments:
      orig_data:  Dataset with t
      eval_model: implements fit(x,y) and predict(x)
    '''

    check_binary_values(orig_data.t)

    rands = rg.binomial(1, 1-prop_test, len(orig_data)).astype(bool)
    train_mask = rands & (orig_data.t==0)
    test_mask = ~rands & (orig_data.t==0)
    prop_mask = test_mask | (orig_data.t==1)

    eval_model.fit(orig_data.x[train_mask], orig_data.y[train_mask])
    eval_preds = eval_model.predict(orig_data.x)
    all_eval_data = Dataset(x=orig_data.x,
                            y=eval_preds == orig_data.y,
                            t=orig_data.t)
    eval_data = subset_dataset(all_eval_data, prop_mask)
    if return_extras:
        masks = {'train': train_mask, 'test': test_mask, 'prop': prop_mask}
        return eval_data, masks
    return eval_data

def plot_decomp_bar(decomp: DistShiftDecomp, labels=['Source','Target'], ax=None, figsize=(5.5,4), termlabeltext=['','','']):
    if ax is None:
        fig,ax=plt.subplots(1,1,figsize=figsize)
    ax.spines['top'].set(alpha=0.2)
    ax.spines['right'].set(alpha=0.2)

    width = 0.5
    barcolor = '#4878d0'
    ax.bar(0, decomp.eprp, width, color=barcolor)
    ax.bar(1, decomp.eqrq, width, color=barcolor)

    terms = [decomp.esrp-decomp.esrq, 
            decomp.esrq-decomp.eqrq,
            decomp.eprp-decomp.esrp]
    termlabels = ['Y|X shift     ', 'X shift (S to Q)', 'X shift (P to S)']
    termsum = np.concatenate([np.zeros(1),np.cumsum(terms)])+decomp.eqrq

    ax.bar(2.7, [terms[2]], width, bottom=[termsum[2]],
           label=termlabels[2]+termlabeltext[1], color='#AA0000')
    ax.bar(2.7, [terms[1]], width, bottom=[termsum[1]],
           label=termlabels[1]+termlabeltext[0], color='red')
    ax.bar(2.7, [terms[0]], width, bottom=[termsum[0]],
           label=termlabels[0]+termlabeltext[2], color='orange')

    ax.axhline(decomp.eqrq, linestyle='dashed', color='black', alpha=0.4, linewidth=1.2)
    ax.axhline(decomp.eprp, linestyle='dashed', color='black', alpha=0.4, linewidth=1.2)
    ax.legend()
    ax.set_xticks([0,1,2.7],labels+['Difference'])
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.legend(loc='lower right')

    plt.annotate(text='', xy=(2.15,decomp.eqrq),
                 xytext=(2.15,decomp.eprp),
                 arrowprops=dict(arrowstyle='<|-|>',
                                 color='black',
                                 mutation_scale=15,
                                 alpha=0.8))

    ax.text(1.9, (decomp.eqrq+decomp.eprp)/2+0.002, 'Accuracy\ndegradation', rotation='vertical', va='center', ha='center')
    return ax

def plot_decomp_line(decomp, ax, arrow_width=3.5, y0=-0.15, y1=-0.25, termsize=13):
    ax.plot([0,1,2,3],decomp.values,marker='o')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['$E_P[R_P(X)]$','$E_S[R_P(X)]$','$E_S[E_Q(X)]$','$E_Q[R_Q(X)]$'])
    ax.annotate('X shift\n(P to S)', xy=(0.2, y0), xytext=(0.2, y1-0.07),
                ha='center', va='bottom', xycoords='axes fraction',  size=termsize,
                bbox=dict(boxstyle='square', fc='0.9', pad=0.4, lw=0),
                arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=.5'.format(arrow_width), lw=1.0))

    ax.annotate('Y|X shift', xy=(0.5, y0), xytext=(0.5, y1),
                ha='center', va='bottom', xycoords='axes fraction',  size=termsize,
                bbox=dict(boxstyle='square', fc='0.9', pad=0.4, lw=0),
                arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=.5'.format(arrow_width), lw=1.0))

    ax.annotate('X shift\n(S to Q)', xy=(0.8, y0), xytext=(0.8, y1-0.07),
                ha='center', va='bottom', xycoords='axes fraction', size=termsize,
                bbox=dict(boxstyle='square', fc='0.9', pad=0.4, lw=0),
                arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=.5'.format(arrow_width), lw=1.0))


