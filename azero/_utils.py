import pyspiel
import os
from azero._model import Model
from azero._eval import AlphaZeroEvaluator
from azero._alpha_zero import Config, _init_bot, mcts


def load_mcts_bot(config: 'dict | Config', is_eval=True):
    if isinstance(config, dict):
        config = Config(**config)

    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions()
    )
    ev = mcts.RandomRolloutEvaluator()
    return _init_bot(config, game, ev, is_eval)


def load_trained_bot(config: 'dict | Config', path: str, checkpoint: int, is_eval=True):
    if isinstance(config, dict):
        config = Config(**config)

    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions()
    )

    full_path = os.path.join(path, f"checkpoint-{checkpoint}")
    print("Loading bot from:", full_path)
    model = Model.from_checkpoint(full_path)
    az_eval = AlphaZeroEvaluator(game, model)
    return _init_bot(config, game, az_eval, evaluation=is_eval), full_path
