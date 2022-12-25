from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanilaLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logit_real, logit_gene):
        loss_real = torch.mean(F.softplus(-logit_real))
        loss_gene = torch.mean(F.softplus(logit_gene))
        return 0.5 * (loss_real + loss_gene)
class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logit_real, logit_gene):
        loss_real = torch.mean(F.relu(1.0 - logit_real))
        loss_gene = torch.mean(F.relu(1.0 + logit_gene))
        return 0.5 * (loss_real + loss_gene)
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

            