"""
Training package
"""
from .trainer import CoordinateAdapterTrainer, create_optimizer_and_scheduler
from .config import Config, get_config, CONFIG_PRESETS

__all__ = [
    'CoordinateAdapterTrainer',
    'create_optimizer_and_scheduler',
    'Config',
    'get_config',
    'CONFIG_PRESETS'
]
