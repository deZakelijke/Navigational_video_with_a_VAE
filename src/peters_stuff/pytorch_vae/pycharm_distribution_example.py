

from typing import Callable
import torch
from torch.distributions import Distribution

VariationalFunction = Callable[[torch.Tensor], Distribution]

class MyModel(VariationalFunction):

    def __call__(self, x) -> Distribution:
        pass


def myfunc(model: VariationalFunction):
    pass


myfunc(MyModel())
