# pytorch-deep-generative-replay
PyTorch implementation of [Continual Learning with Deep Generative Replay, NIPS 2017](https://arxiv.org/abs/1705.08690)

![model](./arts/model.png)


## Results
WIP


## Installation
```shell
$ git clone https://github.com/kuc2477/pytorch-deep-generative-replay && cd pytorch-deep-generative-replay
$ pip install -r requirements.txt
```

## CLI

### Run Experiments
```shell
# Run a visdom server and conduct a desired experiment.
$ python -m visdom.server &
$ ./main.py --train \
    --experiment=[ permutated-mnist | svhn-mnist | mnist-svhn ] \
    --replay-mode=[ exect-replay | generative-replay | none ]
```

### Sample Images from Learned Scholars
```shell
$ # Run the command below and visit the "samples" directory.
$ ./main.py --test \
    --experiment=[ permutated-mnist | svhn-mnist | mnist-svhn ] \
    --replay-mode=[ exect-replay | generative-replay | none ]
```

### Usage
```shell
$ ./main.py --help
$ usage: PyTorch implementation of Deep Generative Replay [-h]
                                                        [--experiment {permutated-mnist,svhn-mnist,mnist-svhn}]
                                                        [--mnist-permutation-number MNIST_PERMUTATION_NUMBER]
                                                        [--mnist-permutation-seed MNIST_PERMUTATION_SEED]
                                                        --replay-mode
                                                        {exect-replay,generative-replay,none}
                                                        [--generator-z-size GENERATOR_Z_SIZE]
                                                        [--generator-c-channel-size GENERATOR_C_CHANNEL_SIZE]
                                                        [--generator-g-channel-size GENERATOR_G_CHANNEL_SIZE]
                                                        [--solver-depth SOLVER_DEPTH]
                                                        [--solver-reducing-layers SOLVER_REDUCING_LAYERS]
                                                        [--solver-channel-size SOLVER_CHANNEL_SIZE]
                                                        [--generator-c-updates-per-g-update GENERATOR_C_UPDATES_PER_G_UPDATE]
                                                        [--generator-iterations GENERATOR_ITERATIONS]
                                                        [--solver-iterations SOLVER_ITERATIONS]
                                                        [--importance-of-new-task IMPORTANCE_OF_NEW_TASK]
                                                        [--lr LR]
                                                        [--weight-decay WEIGHT_DECAY]
                                                        [--batch-size BATCH_SIZE]
                                                        [--test-size TEST_SIZE]
                                                        [--sample-size SAMPLE_SIZE]
                                                        [--image-log-interval IMAGE_LOG_INTERVAL]
                                                        [--eval-log-interval EVAL_LOG_INTERVAL]
                                                        [--loss-log-interval LOSS_LOG_INTERVAL]
                                                        [--checkpoint-dir CHECKPOINT_DIR]
                                                        [--sample-dir SAMPLE_DIR]
                                                        [--no-gpus]
                                                        (--train | --test)

optional arguments:
  -h, --help            show this help message and exit
  --experiment {permutated-mnist,svhn-mnist,mnist-svhn}
  --mnist-permutation-number MNIST_PERMUTATION_NUMBER
  --mnist-permutation-seed MNIST_PERMUTATION_SEED
  --replay-mode {exect-replay,generative-replay,none}
  --generator-z-size GENERATOR_Z_SIZE
  --generator-c-channel-size GENERATOR_C_CHANNEL_SIZE
  --generator-g-channel-size GENERATOR_G_CHANNEL_SIZE
  --solver-depth SOLVER_DEPTH
  --solver-reducing-layers SOLVER_REDUCING_LAYERS
  --solver-channel-size SOLVER_CHANNEL_SIZE
  --generator-c-updates-per-g-update GENERATOR_C_UPDATES_PER_G_UPDATE
  --generator-iterations GENERATOR_ITERATIONS
  --solver-iterations SOLVER_ITERATIONS
  --importance-of-new-task IMPORTANCE_OF_NEW_TASK
  --lr LR
  --weight-decay WEIGHT_DECAY
  --batch-size BATCH_SIZE
  --test-size TEST_SIZE
  --sample-size SAMPLE_SIZE
  --image-log-interval IMAGE_LOG_INTERVAL
  --eval-log-interval EVAL_LOG_INTERVAL
  --loss-log-interval LOSS_LOG_INTERVAL
  --checkpoint-dir CHECKPOINT_DIR
  --sample-dir SAMPLE_DIR
  --no-gpus
  --train
  --test
```

## Notes
- I couldn't find the supplementary materials which the authors mentioned to contain the experimental details. Thus, I arbitrarily chosed a 4-convolutional-layer CNN as a default solver model. Please let me know where I can find the supplementary materials if you know it.

## Reference
- [Continual Learning with Deep Generative Replay, arxiv:1705.08690](https://arxiv.org/abs/1705.08690)


## Author
Ha Junsoo / [@kuc2477](https://github.com/kuc2477) / MIT License
