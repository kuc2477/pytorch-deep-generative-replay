import abc
from torch import nn
import utils
import random


# ============
# Base Classes
# ============

class GenerativeMixin(object):
    def sample(self, size):
        raise NotImplementedError


class ReplayTrainable(nn.Module, metaclass=abc.ABCMeta):
    """

    """

    @abc.abstractmethod
    def train_a_batch(self, x, y):
        raise NotImplementedError

    def train_with_replay(self, dataset, scholar=None,
                          importance_of_new_task=.5,
                          iteration=1000, batch_size=32):
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda()
        ))

        for _ in range(iteration):
            # decide from where to sample the training data.
            from_replay = (
                random.random() > importance_of_new_task and
                scholar is not None
            )

            # sample the training data.
            x, y = (
                scholar.sample(batch_size) if from_replay else
                next(data_loader)
            )

            # train the model with a batch.
            self.train_a_batch(x, y)

    def _is_on_cuda(self):
        return iter(self.parameters()).next().is_cuda


# ===============
# Generic Classes
# ===============


class Solver(ReplayTrainable):
    """

    """
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def train_a_batch(self, x, y):
        self.optimizer.zero_grad()
        loss = self.criterion(self.forward(x), y)
        loss.backward()
        self.optimizer.step()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class Generator(GenerativeMixin, ReplayTrainable):
    """

    """


# ================
# Concrete Classes
# ================

class CNNSolver(Solver):
    pass


class WGANGenerator(Generator):
    pass


class Scholar(GenerativeMixin, ReplayTrainable):
    # TODO: NOT IMPLEMENTED YET
    pass
