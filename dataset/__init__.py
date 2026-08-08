"""Dataset loader registry.

Importing this package registers every loader defined in it. To add a new
dataset (TriviaQA, HotpotQA, HumanEval, MBPP, ...):

1. Write a new module here with a `BaseDatasetLoader` subclass, decorated
   with `@register` (see gsm8k.py / math.py for the pattern).
2. Add one import line below.

Nothing else in the project needs to change — run.py discovers datasets
through `available_datasets()` / `get_loader_class()`, never by name.
"""
from dataset.base import available_datasets, get_loader_class  # noqa: F401

from dataset import gsm8k, math  # noqa: F401,E402
