"""
Loss functions package
"""
from .hungarian_loss import HungarianPointLoss, HungarianPointLossPyTorch

__all__ = [
    'HungarianPointLoss',
    'HungarianPointLossPyTorch'
]
