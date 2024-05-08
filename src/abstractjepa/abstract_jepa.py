import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import copy

class JEPA(nn.Module, ABC):
    def __init__(self, context_encoder, predictor, loss_calculators=None, target_encoder=None):
        super(JEPA, self).__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        if loss_calculators:
            self.loss_calculators = loss_calculators
        if not target_encoder:
            self.target_encoder = copy.deepcopy(context_encoder)
        else:
            self.target_encoder = target_encoder

    @abstractmethod
    def encode_x(self, x):
        pass

    @abstractmethod
    def encode_y(self, y):
        pass

    @abstractmethod
    def predict_encoded_y(self, encoded_x, z):
        pass

    @abstractmethod
    def get_loss(self, encoded_y, predicted_encoded_y, z=None):
        pass
