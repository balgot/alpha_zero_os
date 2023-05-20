# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple AlphaZero tic tac toe example.

Take a look at the log-learner.txt in the output directory.

If you want more control, check out `alpha_zero.py`.
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero.evaluator import AlphaZeroEvaluator
from open_spiel.python.utils import spawn
# import sys
# sys.path.append("azero")
# import azero as alpha_zero
# from azero._utils import spawn
import pyspiel

flags.DEFINE_string("path", "./logs", "Where to save checkpoints.")
FLAGS = flags.FLAGS


def main(unused_argv):
    config = alpha_zero.Config(
        game="tic_tac_toe",
        path=FLAGS.path,
        learning_rate=0.01,
        weight_decay=1e-4,
        train_batch_size=128,
        replay_buffer_size=2**14,
        replay_buffer_reuse=4,
        max_steps=3,
        checkpoint_freq=25,

        actors=4,
        evaluators=4,
        uct_c=1,
        max_simulations=20,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="resnet", # resnet
        nn_width=64, # 128
        nn_depth=4, # 2
        observation_shape=None,
        output_size=None,

        quiet=True,
    )
    alpha_zero.alpha_zero(config)

    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions()
    )

    import itertools, statistics, collections

    def _play_one(game, fst_fn, snd_fn):
        state = game.new_initial_state()
        for idx in itertools.count():
          if state.is_terminal():
            break
          _player = fst_fn if idx % 2 == 0 else snd_fn
          action = _player(state)
          state.apply_action(action)
        return state.returns()

    import os
    path = os.path.join(FLAGS.path, "checkpoint--1")
    print("loading checkpoint:", path)
    model = alpha_zero.model_lib.Model.from_checkpoint(path)
    az_evaluator = AlphaZeroEvaluator(game, model)
    bot = alpha_zero._init_bot(config, game, az_evaluator, True)
    mcts_bot = alpha_zero._init_bot(config, game, alpha_zero.mcts.RandomRolloutEvaluator(n_rollouts=200), True)

    import numpy as np
    def _random(state):
       return np.random.choice(state.legal_actions())

    for opponent, _name in [(mcts_bot.step, "mcts"), (_random, "random")]:
       scores = [[], []]  # as first/second
       for g in range(100):
          bot_first = g % 2 == 0
          if bot_first:
             my, _ = _play_one(game, bot.step, opponent)
             scores[0].append(my)
          else:
             _, my = _play_one(game, opponent, bot.step)
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
  # import wandb
  # wandb.init()
  with spawn.main_handler():
    app.run(main)