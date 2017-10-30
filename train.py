from torch import optim
from torch import nn
from tqdm import tqdm


def _generator_training_callback(loss_log_interval, image_log_interval):
    # TODO: NOT IMPLEMENTED YET
    pass


def _solver_training_callback(loss_log_interval, eval_log_interval):
    # TODO: NOT IMPLEMENTED YET
    pass


def train(model, train_datasets, test_datasets,
          generator_iterations=2000,
          solver_iterations=1000,
          batch_size=32,
          test_size=1024,
          sample_size=36,
          lr=1e-03, weight_decay=1e-05,
          loss_log_interval=30,
          eval_log_interval=50,
          image_log_interval=100,
          cuda=False):
    # TODO: NOT IMPLEMENTED YET
    pass
