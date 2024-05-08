import torch
from torch.nn import functional as F
from abc import ABC, abstractmethod

class VICRegLoss(ABC):
    def __init__(self, expander):
        self.expander = expander

    @abstractmethod
    def calculate_variance(self, expanded_representation):
        pass

    @abstractmethod
    def calculate_covariance(self, expanded_representation):
        pass

    @abstractmethod
    def calculate_invariance(self, expanded_representation_yhat, expanded_representation_y):
        pass




