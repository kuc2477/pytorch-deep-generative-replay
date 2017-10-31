import os.path
import copy
from torch import optim
from torch import nn
import utils
import visual


def train(scholar, train_datasets, test_datasets, replay_mode,
          generator_lambda=10.,
          generator_c_updates_per_g_update=5,
          generator_iterations=2000,
          solver_iterations=1000,
          importance_of_new_task=.5,
          batch_size=32,
          test_size=1024,
          sample_size=36,
          lr=1e-03, weight_decay=1e-05,
          beta1=.5, beta2=.9,
          loss_log_interval=30,
          eval_log_interval=50,
          image_log_interval=100,
          sample_log_interval=300,
          sample_log=False,
          sample_dir='./samples',
          checkpoint_dir='./checkpoints',
          collate_fn=None,
          cuda=False):
    # define solver criterion and generators for the scholar model.
    solver_criterion = nn.CrossEntropyLoss()
    solver_optimizer = optim.Adam(
        scholar.solver.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_g_optimizer = optim.Adam(
        scholar.generator.generator.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_c_optimizer = optim.Adam(
        scholar.generator.critic.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )

    # set the criterion, optimizers, and training configurations for the
    # scholar model.
    scholar.solver.set_criterion(solver_criterion)
    scholar.solver.set_optimizer(solver_optimizer)
    scholar.generator.set_lambda(generator_lambda)
    scholar.generator.set_generator_optimizer(generator_g_optimizer)
    scholar.generator.set_critic_optimizer(generator_c_optimizer)
    scholar.generator.set_critic_updates_per_generator_update(
        generator_c_updates_per_g_update
    )
    scholar.train()

    # define the previous scholar who will generate samples of previous tasks.
    previous_scholar = None
    previous_datasets = None

    for task, train_dataset in enumerate(train_datasets, 1):
        # define callbacks for visualizing the training process.
        generator_training_callbacks = [_generator_training_callback(
            loss_log_interval=loss_log_interval,
            image_log_interval=image_log_interval,
            sample_log_interval=sample_log_interval,
            sample_log=sample_log,
            sample_dir=sample_dir,
            sample_size=sample_size,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=generator_iterations,
            batch_size=batch_size,
            replay_mode=replay_mode,
            env=scholar.name,
        )]
        solver_training_callbacks = [_solver_training_callback(
            loss_log_interval=loss_log_interval,
            eval_log_interval=eval_log_interval,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=solver_iterations,
            batch_size=batch_size,
            test_size=test_size,
            test_datasets=test_datasets,
            replay_mode=replay_mode,
            cuda=cuda,
            collate_fn=collate_fn,
            env=scholar.name,
        )]

        # train the scholar with generative replay.
        scholar.train_with_replay(
            train_dataset,
            scholar=previous_scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            generator_iterations=generator_iterations,
            generator_training_callbacks=generator_training_callbacks,
            solver_iterations=solver_iterations,
            solver_training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

        previous_scholar = (
            copy.deepcopy(scholar) if replay_mode == 'generative-replay' else
            None
        )
        previous_datasets = (
            train_datasets[:task] if replay_mode == 'exact-replay' else
            None
        )

    # save the model after the experiment.
    print()
    utils.save_checkpoint(scholar, checkpoint_dir)
    print()
    print()


def _generator_training_callback(
        loss_log_interval,
        image_log_interval,
        sample_log_interval,
        sample_log,
        sample_dir,
        current_task,
        total_tasks,
        total_iterations,
        batch_size,
        sample_size,
        replay_mode,
        env):

    def cb(generator, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            '<Training Generator> '
            'task: {task}/{tasks} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss => '
            'g: {g_loss:.4} / '
            'w: {w_dist:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            trained=batch_size * batch_index,
            total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            g_loss=result['g_loss'],
            w_dist=-result['c_loss'],
        ))

        # log the losses of the generator.
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(
                result['g_loss'], 'generator g loss', iteration, env=env
            )
            visual.visualize_scalar(
                -result['c_loss'], 'generator w distance', iteration, env=env
            )

        # log the generated images of the generator.
        if iteration % image_log_interval == 0:
            visual.visualize_images(
                generator.sample(sample_size).data,
                'generated samples ({replay_mode})'
                .format(replay_mode=replay_mode), env=env,
            )

        # log the sample images of the generator
        if iteration % sample_log_interval == 0 and sample_log:
            utils.test_model(generator, sample_size, os.path.join(
                sample_dir,
                env + '-sample-logs',
                str(iteration)
            ), verbose=False)

    return cb


def _solver_training_callback(
        loss_log_interval,
        eval_log_interval,
        current_task,
        total_tasks,
        total_iterations,
        batch_size,
        test_size,
        test_datasets,
        cuda,
        replay_mode,
        collate_fn,
        env):

    def cb(solver, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            '<Training Solver>    '
            'task: {task}/{tasks} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss: {loss:.4} | '
            'prec: {prec:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            trained=batch_size * batch_index,
            total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            loss=result['loss'],
            prec=result['precision'],
        ))

        # log the loss of the solver.
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(
                result['loss'], 'solver loss', iteration, env=env
            )

        # evaluate the solver on multiple tasks.
        if iteration % eval_log_interval == 0:
            names = ['task {}'.format(i+1) for i in range(len(test_datasets))]
            precs = [
                utils.validate(
                    solver, test_datasets[i], test_size=test_size,
                    cuda=cuda, verbose=False, collate_fn=collate_fn,
                ) if i+1 <= current_task else 0 for i in
                range(len(test_datasets))
            ]
            title = 'precision ({replay_mode})'.format(replay_mode=replay_mode)
            visual.visualize_scalars(
                precs, names, title,
                iteration, env=env
            )

    return cb
