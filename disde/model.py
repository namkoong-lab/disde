from disde.dataset import Dataset
import numpy as np
from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data: Dataset):
        pass
    @abstractmethod
    def predict(self, data: Dataset):
        pass

class SKLearnPropModel(Model):
    def __init__(self, model, clip=0):
        self.model = model
        self.clip = clip
    def fit(self, data: Dataset):
        assert data.t is not None
        self.model.fit(data.x, data.t, sample_weight=data.w)
    def predict(self, data):
        return np.clip(self.model.predict_proba(data.x)[:,1],self.clip,1-self.clip)

class SKLearnNoTreatmentOutcomeModel(Model):
    def __init__(self, model):
        self.model = model
    def fit(self, data: Dataset):
        self.model.fit(data.x, data.y, sample_weight=data.w)
    def predict(self, data):
        return self.model.predict(data.x)

