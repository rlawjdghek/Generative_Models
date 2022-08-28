from abc import ABC, abstractmethod

import torch
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
    @abstractmethod
    def set_input(self): pass
    @abstractmethod
    def train(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad