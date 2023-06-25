import numpy as np
from numpy.random import RandomState
import copy

class Dataset:
    def __init__(self, x=None, y=None, t=None, w=None):
        self.x=x
        self.y=y
        self.w=w      # sample weight
        self.t=t

        assert x is None or isinstance(x, np.ndarray)
        assert y is None or isinstance(y, np.ndarray)
        assert w is None or isinstance(w, np.ndarray)
        assert t is None or isinstance(t, np.ndarray)

        # verify length is consistent
        _ = self.__len__()
        
    def _check_equal_len(self, cur_len, x):
        if x is None:
            return cur_len
        if cur_len is None:
            return len(x)
        assert len(x) == cur_len
        return cur_len
    
    # a bit of a hack: gives you 1's if it would have been none
    def sample_weight(self):
        if self.w is None:
            return 1
        return self.w
    
    def E(self, z):
        if self.w is None:
            return z.sum() / len(z)
        else:
            return (z * self.w).sum() / self.w.sum()
    
    def wate(self, w):
        return self.E(self.y * w) / self.E(w)
    
    def __len__(self):
        res = None
        res = self._check_equal_len(res, self.x)
        res = self._check_equal_len(res, self.y)
        res = self._check_equal_len(res, self.t)
        res = self._check_equal_len(res, self.w)
        return res
    def __copy__(self):
        return copy.deepcopy(self)
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d


def check_binary_values(t):
    assert (np.sort(np.unique(np.array(t))) == np.array([0,1])).all()

def _subset(x, idxs):
    if x is None:
        return None
    return x[idxs]

def subset_dataset(data: Dataset, idxs):
    return Dataset(x=subset(data.x, idxs), 
                   y=subset(data.y, idxs), 
                   w=subset(data.w, idxs),
                   t=subset(data.t, idxs))

def bootstrap(data: Dataset, rg=RandomState(), 
              replace=True, 
              num_samples=None):
    if num_samples is None:
        num_samples = len(data)
    if replace:
        new_idxs = rg.choice(np.arange(num_samples), size=num_samples)
    else:
        new_idxs = rg.permutation(len(data))[:num_samples]
    return subset_dataset(data, new_idxs)

def balance_weights_t_inplace(data: Dataset):
    assert data.t is not None
    
    if data.w is None:
        data.w = np.ones(len(data))
    vals, counts = np.unique(data.t, return_inverse=True)

    for i,v in enumerate(vals):
        value_mask = data.t==v
        old_weight_sum = data.w[value_mask].sum()
        data.w[value_mask] /= old_weight_sum

    # have total weight sum to number of elements
    data.w = data.w / data.w.sum() * len(data.w)
    
def balance_weights_t(data: Dataset):
    new_data = copy.deepcopy(data)
    balance_weights_t_inplace(new_data)
    return new_data

def subset(x, idxs):
    if x is None:
        return None
    return x[idxs]

def concat_dataset(data1, data2):
    new_vals = {}
    for attr in ['x','y','t']:
        if getattr(data1, attr) is None or getattr(data2, attr) is None:
            pass
        new_vals[attr] = np.concatenate([getattr(data1, attr), getattr(data2, attr)])
    return Dataset(**new_vals)
