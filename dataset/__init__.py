"""Dataset loader registry. Add a dataset by writing a module with a
BaseDatasetLoader subclass decorated with @register, then import it below.
"""
from dataset.base import datasets, get_loader_class  # noqa: F401

from dataset import gsm8k, math
