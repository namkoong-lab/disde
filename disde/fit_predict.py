import copy
import numpy as np
from numpy.random import RandomState
from typing import Optional, List
from collections import namedtuple
from disde.dataset import Dataset, subset_dataset
from disde.model import Model

'''
Deal with cross-fitting
'''

TrainTestIdxs = namedtuple('TrainTestIdxs',['train','test'])

class CrossFitIdx:
    def __init__(self, N, num_splits, rg=RandomState(283402420), remap=True):
        self.N=N
        self.num_splits=num_splits
        if hasattr(N, '__len__'):
            self.map = np.concatenate([np.arange(n) for n in N])
        else:
            self.map = np.arange(N)
        if remap:
            rg.shuffle(self.map)
    def subset(self, idxs):
        res = CrossFitIdx(len(self.map[idxs]), self.num_splits)
        res.map = self.map[idxs]
        return res
    def get_idxs(self, split_num, train_vs_test=None):
        labels = self.map % self.num_splits
        '''
        Args:
            split (int):
            train_vs_test (string): either 'train' or 'test'
                with the idea that we test on K-1 splits
                and test on the remaining 1
        '''
        if train_vs_test is None:
            return TrainTestIdxs(
                test=np.where(labels == split_num)[0],
                train=np.where(labels != split_num)[0]
            )
        if train_vs_test == 'test':
            return np.where(labels == split_num)[0]
        elif train_vs_test == 'train':
            return np.where(labels != split_num)[0]
        else:
            raise ValueError("argument train_vs_test={} must be either 'train' or 'test'".format(train_vs_test))
    def __iter__(self):
        return CrossFitIdxIterator(self)
    def copy(self):
        return copy.deepcopy(self)
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d 

class CrossFitIdxIterator:
    def __init__(self, cf_idx: CrossFitIdx):
        self.cf_idx = cf_idx
        self._idx = 0
    def __next__(self):
        if self._idx >= self.cf_idx.num_splits:
            raise StopIteration
        res = self.cf_idx.get_idxs(self._idx, None)
        self._idx += 1
        return res

def cross_predict(data: Dataset,
                models: List[Model], 
                cf: CrossFitIdx):
    if cf is None and len(models)==1:
        return models[0].predict(data)
    else:
        test_idxs_l = []
        preds_l = []
        # probably better to write directly in-place
        # to sliced final result rather than concat results
        for model,idxs in zip(models,cf):
            test_data = subset_dataset(data, idxs.test) 
            test_idxs_l.append(idxs.test)
            preds_l.append(model.predict(test_data))
        # rearrange predictions
        test_idxs = np.concatenate(test_idxs_l)
        perm = np.argsort(test_idxs)

        # the concat here won't work for all ... things. 
        preds = np.concatenate(preds_l)[perm] 
        return preds

def cross_fit_predict(data: Dataset,
                      model: Model, 
                      cf: Optional[CrossFitIdx]=None,
                      return_extras: bool=False):
    assert cf.N == len(data)
    if cf is None:
        return model.predict(data)
    else:
        test_idxs_l = []
        preds_l = []
        in_sample_preds = []
        in_sample_targets = []
        models = []
        # probably better to write directly in-place
        # to sliced final result rather than concat results
        for idxs in cf:
            train_data = subset_dataset(data, idxs.train) 
            test_data = subset_dataset(data, idxs.test) 
            test_idxs_l.append(idxs.test)
            model.fit(train_data)
            preds_l.append(model.predict(test_data))
            if return_extras:
                in_sample_preds.append(model.predict(train_data))
                in_sample_targets.append(train_data.y)
                models.append(copy.deepcopy(model))
        # rearrange predictions
        test_idxs = np.concatenate(test_idxs_l)
        perm = np.argsort(test_idxs)

        # the concat here won't work for all ... things. 
        preds = np.concatenate(preds_l)[perm] 
        if return_extras:
            return {
                    'predictions': preds,
                    'in_sample_preds':in_sample_preds,
                    'in_sample_targets':in_sample_targets,
                    'models':models,
                    'test_idxs': test_idxs_l}
        return preds

def combine_cf(cf1, cf2):
    assert cf1.num_splits == cf2.num_splits
    new_cf = CrossFitIdx(cf1.N+cf2.N, cf1.num_splits)
    new_cf.map[:cf1.N] = cf1.map
    new_cf.map[cf1.N:] = cf2.map + cf1.N
    return new_cf
