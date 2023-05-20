"""Simple AlphaZero tic tac toe example."""

from absl import app
from absl import flags
import itertools, statistics, collections
import numpy as np

import sys
sys.path.append("azero")
from azero import spawn, load_mcts_bot, load_trained_bot, alpha_zero, Config
import pyspiel


flags.DEFINE_string("path", "./logs", "Where to save checkpoints.")
FLAGS = flags.FLAGS


def main(unused_argv):
    # configuration
    config = Config(
        game="tic_tac_toe",
        path=FLAGS.path,
        learning_rate=0.001,
        weight_decay=1e-4,
        train_batch_size=256,
        replay_buffer_size=2**14,
        replay_buffer_reuse=4,
        max_steps=10,
        checkpoint_freq=25,

        actors=1,
        evaluators=1,
        uct_c=1,
        max_simulations=20,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="mlp", # resnet
        nn_width=512, # 128
        nn_depth=8, # 2
        observation_shape=None,
        output_size=None,

        quiet=True,
    )


    def _play_one(game, fst_fn, snd_fn):
        state = game.new_initial_state()
        for idx in itertools.count():
          if state.is_terminal():
            break
          _player = fst_fn if idx % 2 == 0 else snd_fn
          action = _player(state)
          state.apply_action(action)
        return state.returns()

    def _random(state):
       return np.random.choice(state.legal_actions())


    # run the training algortihm
    game = pyspiel.load_game(config.game)
    checkpoint = "./logs/checkpoint--1"
    alpha_zero(config, is_win_loose=True, checkpoint=checkpoint, start_step=10)

    # evaluate trained bot
    trained, chck_path = load_trained_bot(config, FLAGS.path, -1)
    checkpoint = chck_path
    mcts_bot = load_mcts_bot(config)

    for opponent, _name in [(mcts_bot.step, "mcts"), (_random, "random"), (trained.step, "itself")]:
      scores = [[], []]  # as first/second
      for g in range(100):
          bot_first = g % 2 == 0
          if bot_first:
            my, _ = _play_one(game, trained.step, opponent)
            scores[0].append(my)
          else:
            _, my = _play_one(game, opponent, trained.step)
            scores[1].append(my)

      print("=================")
      print(_name)
      print("=================")
      print("Total:", np.mean(np.concatenate(scores)))
      print(collections.Counter(np.concatenate(scores)))

      print("\tFirst:", np.mean(scores[0]))
      print("\t", collections.Counter(scores[0]))

      print("\tSecond:", np.mean(scores[1]))
      print("\t", collections.Counter(scores[1]))
      print("\n\n")

if __name__ == "__main__":
  import wandb
  wandb.init()
  with spawn.main_handler():
    app.run(main)