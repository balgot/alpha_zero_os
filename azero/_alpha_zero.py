# Modified from `open_spiel` repo: https://github.com/deepmind/open_spiel
"""A basic AlphaZero implementation."""

import datetime
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np
import wandb
import pyspiel

from azero._model import Model, TrainInput, Losses
from azero._eval import AlphaZeroEvaluator

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero.alpha_zero import (
    JOIN_WAIT_DELAY,
    Buffer,
    Config,
    watcher,
    _play_game,
    data_logger,
    stats,
    spawn
)


def _init_model_from_config(config: Config, is_win_loose=True):
    """Randomly initialize a new model based on the config."""
    return Model.build_model(
            config.nn_model,
            config.observation_shape,
            config.output_size,
            config.nn_width,
            config.nn_depth,
            config.weight_decay,
            config.learning_rate,
            config.path,
            is_win_loose=is_win_loose
    )


def _init_bot(config: Config, game, evaluator_, evaluation):
    """Initializes a bot."""
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    return mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True
    )


def update_checkpoint(logger, queue, model, az_evaluator):
    """Read the queue for a checkpoint to load, or an exit signal."""
    path = None
    while True:  # Get the last message, ignore intermediate ones.
        try:
            path = queue.get_nowait()
        except spawn.Empty:
            break
    if path:
        logger.print("Found path:", path)
        logger.print("Inference cache:", az_evaluator.cache_info())
        logger.print("Loading checkpoint", path)
        model.load_checkpoint(path)
        az_evaluator.clear_cache()
    elif path is not None:  # Empty string means stop this process.
        logger.print("Found enpty string, finishing")
        return False
    return True


@watcher
def actor(*, config, game, logger, queue, is_win_loose=True):
    """An actor process runner that generates games and returns trajectories."""

    # create the (random) model for this process
    logger.print("Initializing model")
    model = _init_model_from_config(config, is_win_loose=is_win_loose)

    # create bots for this process
    logger.print("Initializing bots")
    az_evaluator = AlphaZeroEvaluator(game, model)
    bots = [
        _init_bot(config, game, az_evaluator, False)
        for _ in range(game.num_players())
    ]
    logger.print(f"Intialized {len(bots)} bots.")

    for game_num in itertools.count():
        # if there is empty string in the queue, this returns false
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return
        _res = _play_game(logger, game_num, game, bots, config.temperature, config.temperature_drop)
        queue.put(_res)


@watcher
def evaluator(*, game, config, logger, queue, is_win_loose=True):
    """A process that plays the latest checkpoint vs standard MCTS."""

    results = Buffer(config.evaluation_window)
    logger.print("Initializing model")
    model = _init_model_from_config(config, is_win_loose=is_win_loose)

    logger.print("Initializing bots")
    az_evaluator = AlphaZeroEvaluator(game, model)
    random_evaluator = mcts.RandomRolloutEvaluator()

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return

        az_player = game_num % 2
        difficulty = (game_num // 2) % config.eval_levels
        max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
        bots = [
            _init_bot(config, game, az_evaluator, True),
            mcts.MCTSBot(game, config.uct_c, max_simulations, random_evaluator, solve=True, verbose=False, dont_return_chance_node=True)
        ]
        if az_player == 1:
            bots = list(reversed(bots))

        trajectory = _play_game(logger, game_num, game, bots, temperature=1, temperature_drop=0)
        results.append(trajectory.returns[az_player])
        queue.put((difficulty, trajectory.returns[az_player]))

        logger.print(
            f"AlphaZero: {trajectory.returns[az_player]}, "
            f"MCTS: {trajectory.returns[1 - az_player]}, "
            f"AlphaZero avg/{len(results)} games: {np.mean(results.data):.3f}"
        )


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger, checkpoint=None, is_win_loose=True, start_step=1):
    """A learner that consumes the replay buffer and trains the network."""

    logger.also_to_stdout = True
    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse

    if checkpoint is None:
        logger.print(f"Initializing model: {config.nn_model}({config.nn_width}, {config.nn_depth})")
        model = _init_model_from_config(config, is_win_loose=is_win_loose)
        save_path = model.save_checkpoint(0)
        logger.print("Initial checkpoint:", save_path)
        broadcast_fn(save_path)
    else:
        logger.print(f"Loading model from checkpoint: {checkpoint}")
        model = Model.from_checkpoint(checkpoint)
        broadcast_fn(checkpoint)

    logger.print("Model size:", model.num_trainable_variables, "variables")
    model.print_trainable_variables()

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)
    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
    total_trajectories = 0
    _returns = []

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_trajectories():
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            _returns.append(p1_outcome)

            replay_buffer.extend(
                TrainInput(
                    s.observation,
                    s.legals_mask,
                    s.policy,
                    p1_outcome
                )
                for s in trajectory.states
            )

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []
        for _ in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            losses.append(model.update(data))

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        save_path = model.save_checkpoint(step if step % config.checkpoint_freq == 0 else -1)

        losses = sum(losses, Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

    last_time = time.time() - 60
    for step in range(start_step, start_step+config.max_steps):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()


        logger.print(f"Collecting trajectories (step={step})")
        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print("Step:", step)
        logger.print(f"Collected {num_states:5} states from {num_trajectories:3} games, {num_states / seconds:.1f} states/s.")
        logger.print(f"Buffer size: {len(replay_buffer)}. States seen: {replay_buffer.total_seen}")

        save_path, losses = learn(step)

        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        wandb.log({
            "step": step,
            "total_states": replay_buffer.total_seen,
            "states_per_s": num_states / seconds,
            "states_per_s_actor": num_states / (config.actors * seconds),
            "total_trajectories": total_trajectories,
            "trajectories_per_s": num_trajectories / seconds,
            "game_length": game_lengths.as_dict,
            "game_length_hist": game_lengths_hist.data,
            "eval": {
                "count": evals[0].total_seen,
                "results": [sum(e.data) / len(e) if e else 0 for e in evals],
                "mean": np.mean([sum(e.data) / len(e) if e else 0 for e in evals])
            },
            "returns": {
                "mean": np.mean(_returns),
                "std": np.std(_returns),
                "max": max(_returns),
                "min": min(_returns),
                "data": _returns
            },
            "loss": {
                "policy": losses.policy,
                "value": losses.value,
                "l2reg": losses.l2,
                "sum": losses.total,
            },
        })
        data_log.write({
                "step": step,
                "total_states": replay_buffer.total_seen,
                "states_per_s": num_states / seconds,
                "states_per_s_actor": num_states / (config.actors * seconds),
                "total_trajectories": total_trajectories,
                "trajectories_per_s": num_trajectories / seconds,
                "queue_size": 0,  # Only available in C++.
                "game_length": game_lengths.as_dict,
                "game_length_hist": game_lengths_hist.data,
                "value_accuracy": [v.as_dict for v in value_accuracies],
                "value_prediction": [v.as_dict for v in value_predictions],
                "eval": {
                        "count": evals[0].total_seen,
                        "results": [sum(e.data) / len(e) if e else 0 for e in evals],
                },
                "batch_size": batch_size_stats.as_dict,
                "batch_size_hist": [0, 1],
                "loss": {
                        "policy": losses.policy,
                        "value": losses.value,
                        "l2reg": losses.l2,
                        "sum": losses.total,
                },
                "cache": {  # Null stats because it's hard to report between processes.
                        "size": 0,
                        "max_size": 0,
                        "usage": 0,
                        "requests": 0,
                        "requests_per_s": 0,
                        "hits": 0,
                        "misses": 0,
                        "misses_per_s": 0,
                        "hit_rate": 0,
                },
        })
        logger.print()
        broadcast_fn(save_path)


def alpha_zero(
        config: 'dict | Config',
        is_win_loose: bool = True,
        checkpoint: 'str | None' = None,
        start_step = 1
    ):

    """Start all the worker processes for a full alphazero setup."""

    # load the game, prepare config
    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions()
    )

    print("Starting game", config.game)

    if checkpoint is None:
        path = config.path
        if not path:
            path = tempfile.mkdtemp(prefix="az-{}-{}-".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
            config = config._replace(path=path)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            sys.exit("{} isn't a directory".format(path))
        print("Writing logs and checkpoints to:", path)
        with open(os.path.join(config.path, "config.json"), "w") as fp:
            fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")
    else:
        print("Resuming from checkpoint stored at:", checkpoint)

    # spawn generators
    actors = [spawn.Process(actor, kwargs={"game": game, "config": config, "num": i, "is_win_loose": is_win_loose}) for i in range(config.actors)]
    evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config, "num": i, "is_win_loose": is_win_loose}) for i in range(config.evaluators)]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    # run the training process
    try:
        learner(
            game=game,
            config=config,
            actors=actors,
            evaluators=evaluators,
            broadcast_fn=broadcast,
            is_win_loose=is_win_loose,
            checkpoint=checkpoint,
            start_step=start_step
        )
    except (KeyboardInterrupt, EOFError):
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        broadcast("")
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(JOIN_WAIT_DELAY)
        for proc in evaluators:
            proc.join()
