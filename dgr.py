import abc
import utils
import random
from random import choice
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable


# ============
# Base Classes
# ============

class GenerativeMixin(object):
    """Mixin which defines a sampling iterface for a generative model."""
    def sample(self, size):
        raise NotImplementedError


class BatchTrainable(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class which defines a generative-replay based training
    interface for a model.

    """
    @abc.abstractmethod
    def train_a_batch(self, x, y):
        raise NotImplementedError


# ==============================
# Deep Generative Replay Modules
# ==============================

class Generator(GenerativeMixin, BatchTrainable):
    """Abstract generator module of a scholar module"""


class Solver(BatchTrainable):
    """Abstract solver module of a scholar module"""
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def solve(self, x):
        scores = self(x)
        _, predictions = torch.max(scores, 1)
        return predictions

    def train_a_batch(self, x, y):
        # run the model and backpropagate the errors
        self.optimizer.zero_grad()
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        loss.backward()
        self.optimizer.step()

        # calculate the training precision
        _, predicted = scores.max(1)
        precision = (y == predicted).sum().data[0] / x.size(0)
        return {'loss': loss.data[0], 'precision': precision}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class Scholar(GenerativeMixin, nn.Module):
    """Scholar for Deep Generative Replay"""
    def __init__(self, label, generator, solver):
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(
            self, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32,
            generator_iterations=2000,
            generator_training_callbacks=None,
            solver_iterations=1000,
            solver_training_callbacks=None):

        # train the generator of the scholar.
        self._train_batch_trainable_with_replay(
            self.generator, dataset, scholar,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=generator_iterations,
            training_callbacks=generator_training_callbacks,
        )

        # train the solver of the scholar.
        self._train_batch_trainable_with_replay(
            self.solver, dataset, scholar,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=solver_iterations,
            training_callbacks=solver_training_callbacks,
        )

    def sample(self, size):
        x = self.generator.sample(size)
        y = self.solver.solve(x)
        return x.data, y.data

    def _train_batch_trainable_with_replay(
            self, trainable, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32, iterations=1000,
            training_callbacks=None):

        # scholar and previous datasets cannot be given at the same time.
        mutex_condition_infringed = all([
            scholar is not None,
            bool(previous_datasets)
        ])
        assert not mutex_condition_infringed, (
            'scholar and previous datasets cannot be given at the same time'
        )

        # create data loaders.
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda()
        ))
        previous_datasets = previous_datasets or []
        previous_loaders = [
            iter(utils.get_data_loader(d, batch_size, cuda=self._is_on_cuda()))
            for d in previous_datasets
        ]
        # define a tqdm progress bar.
        progress = tqdm(range(1, iterations+1))

        for batch_index in progress:
            # decide from where to sample the training data.
            from_scholar = (
                random.random() > importance_of_new_task and
                scholar is not None
            )
            from_previous_datasets = (
                random.random() > importance_of_new_task and
                previous_datasets
            )

            # sample the training data.
            x, y = (
                scholar.sample(batch_size) if from_scholar else
                next(choice(previous_loaders)) if from_previous_datasets else
                next(data_loader)
            )
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)

            # train the model with a batch.
            result = trainable.train_a_batch(x, y)

            # fire the callbacks on each iteration.
            for callback in (training_callbacks or []):
                callback(trainable, progress, batch_index, result)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
